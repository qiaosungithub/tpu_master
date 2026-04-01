import time
import subprocess
import logging
import os
import sys
import json
import fcntl
from multiprocessing import Pool
from collections import defaultdict
import re
from lock_util import _parse_lock_filename, get_lock_time_str, lock_time_seconds_between

# ================= 配置区域 =================

CHECK_INTERVAL = 300
MAX_WORKERS = 10

ZONES = [
    "us-central1-a",
    "us-central1-b",
    "us-central2-b",
    "us-east1-d",
    "us-east5-a",
    "us-east5-b",
    "europe-west4-a",
    "asia-northeast1-b",
]

# 需要并行审计的"人名 / key"
PREFIXES = ["llq", "keya", "dmy", "gzy", "kangyang"]

# 未匹配任何 prefix 的最后一类
OTHER_PREFIX = "__OTHER__"

LOG_PATH = "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/tpu_enforcer.log"
LOCK_DIR = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
LOCK_EXPIRE_SECONDS = 30 * 60

TPU_MANAGER_DIR = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager"
TPU_MANAGER_DATA_PATH = os.path.join(TPU_MANAGER_DIR, "data.json")
MOUNTED_FILE = os.path.join(TPU_MANAGER_DIR, "mounted.json")

# TPU names containing any of these keywords will be skipped by auto register/mount.
AUTO_REGISTER_MOUNT_SKIP_KEYWORDS = ("katelyn", "victor", "zander", "xtiange")

# ===========================================


# ---------- Console coloring: [IDLE] in green, [RESERVED] in yellow ----------
class IdleOnlyFormatter(logging.Formatter):
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    # Matches: [IDLE] <tpu-name>
    IDLE_PATTERN = re.compile(r"(\[IDLE\])\s+([^\s]+)")
    # Matches: [RESERVED] <tpu-name>
    RESERVED_PATTERN = re.compile(r"(\[RESERVED\])\s+([^\s]+)")

    def format(self, record):
        msg = super().format(record)

        def repl_idle(m):
            return (
                f"{self.GREEN}{m.group(1)}{self.RESET} "
                f"{self.GREEN}{m.group(2)}{self.RESET}"
            )

        def repl_reserved(m):
            return (
                f"{self.YELLOW}{m.group(1)}{self.RESET} "
                f"{self.YELLOW}{m.group(2)}{self.RESET}"
            )

        msg = self.IDLE_PATTERN.sub(repl_idle, msg)
        msg = self.RESERVED_PATTERN.sub(repl_reserved, msg)
        return msg


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = "%(asctime)s [%(levelname)s] %(message)s"

    # File: plain text (NO ANSI)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(fmt))

    # Console: ONLY highlight [IDLE] + TPU name
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(IdleOnlyFormatter(fmt))

    logger.addHandler(fh)
    logger.addHandler(ch)


setup_logging()


# ---------------- Utilities ----------------


def should_skip_tpu(name: str, zone: str, state: str) -> bool:
    if state in {
        "PREEMPTED",
        "TERMINATED",
        "CREATING",
        "DELETING",
        "REPAIRING",
        "STOPPED",
    }:
        return True

    # Special ignore
    if name == "kmh-tpuvm-v4-8-4":
        return True

    # Problematic dev machines: keep skip + warn (still white on console per requirement)
    if "kmh-tpuvm-v3-8" in name or "kmh-tpuvm-v4-8-" in name:
        # logging.info(f"[SKIP] problematic dev machine: {name}")
        return True

    return False


def delete_preempted_tpu(tpu_info: dict):
    """
    Delete a single PREEMPTED TPU.
    tpu_info: {"name": ..., "zone": ..., "state": ...}

    Return: (status, name, zone)
      status in {"DELETE_SUCCESS", "DELETE_TIMEOUT", "DELETE_FAIL"}
    """
    name = tpu_info["name"]
    zone = tpu_info["zone"]

    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "delete",
        name,
        "--zone",
        zone,
        "--quiet",
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        logging.info(f"[DELETE] Successfully deleted PREEMPTED TPU: {name} ({zone})")
        return ("DELETE_SUCCESS", name, zone)
    except subprocess.TimeoutExpired:
        logging.info(f"[DELETE] Timeout deleting {name} ({zone})")
        return ("DELETE_TIMEOUT", name, zone)
    except Exception as e:
        logging.info(f"[DELETE] Failed to delete {name} ({zone}): {e}")
        return ("DELETE_FAIL", name, zone)


def list_tpus_in_zone(zone: str):
    """
    List all ACTIVE TPUs (after skip policy) in a zone.
    Also collect PREEMPTED TPUs for deletion.

    Return: (active_tpus, preempted_tpus)
      - active_tpus: list of {name, zone}
      - preempted_tpus: list of {name, zone, state}
    """
    active_results = []
    preempted_results = []
    cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "list",
        "--zone",
        zone,
        "--format=value(name,state)",
    ]

    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name, state = parts[0].strip(), parts[1].strip()

            # Collect PREEMPTED TPUs for deletion
            if state == "PREEMPTED":
                preempted_results.append({"name": name, "zone": zone, "state": state})
                continue

            # apply skip policy for active TPUs
            if should_skip_tpu(name, zone, state):
                continue

            active_results.append({"name": name, "zone": zone})

    except Exception as e:
        logging.info(f"[ALL] {zone}: list failed ({e})")

    return (active_results, preempted_results)


def assign_prefix(name: str, prefixes):
    """
    Assign a TPU name to the first matching prefix (priority by PREFIXES order).
    If no match, return OTHER_PREFIX.
    """
    for pfx in prefixes:
        if pfx in name:
            return pfx
    return OTHER_PREFIX


def collect_recent_reservations():
    """
    Scan all lock files once and return fresh reservations:
      {vm_name: user}
    Expired or invalid lock files are deleted.
    """
    reservations = {}
    now = get_lock_time_str()

    try:
        files = os.listdir(LOCK_DIR)
    except OSError as e:
        logging.info(f"[LOCK] list failed ({e})")
        return reservations

    for file in files:
        parsed = _parse_lock_filename(file)
        full_path = os.path.join(LOCK_DIR, file)
        if parsed is None:
            continue

        user, vm_name, time_str = parsed
        try:
            seconds_ago = lock_time_seconds_between(time_str, now)
        except (ValueError, TypeError):
            # Old/invalid timestamp format, treat as expired.
            try:
                os.remove(full_path)
            except OSError:
                pass
            continue

        if seconds_ago > LOCK_EXPIRE_SECONDS:
            try:
                os.remove(full_path)
            except OSError:
                pass
            continue

        # Keep the first valid reservation encountered for this TPU.
        if vm_name not in reservations:
            reservations[vm_name] = user

    return reservations


# ---------------- Core check ----------------


def check_single_tpu(tpu: dict):
    """
    tpu dict:
      {"name":..., "zone":..., "prefix":...}

    Return:
      (prefix, name, zone, status, message)
    status in {"IDLE","BUSY","SSH_FAIL","TIMEOUT","ERROR"}
    """
    name = tpu["name"]
    zone = tpu["zone"]
    prefix = tpu["prefix"]

    # No PID output requested
    remote_cmd = (
        "PID=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | head -n 1); "
        'if [ -z "$PID" ]; then echo "CHECK_RES:IDLE"; exit 0; fi; '
        'TPU_USER=$(ps -o user= -p "$PID"); '
        'echo "CHECK_RES:BUSY|USER:$TPU_USER"'
    )

    ssh_cmd = [
        "gcloud",
        "compute",
        "tpus",
        "tpu-vm",
        "ssh",
        name,
        "--zone",
        zone,
        "--worker=all",
        "--ssh-flag=-n",
        "--command",
        remote_cmd,
    ]

    try:
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=40)

        if res.returncode != 0:
            msg = f"[{prefix}] [SSH_FAIL] {name}: {res.stderr.strip()}"
            return (prefix, name, zone, "SSH_FAIL", msg)

        users = set()
        saw_check = False
        saw_busy = False

        for line in res.stdout.splitlines():
            if "CHECK_RES:" not in line:
                continue

            saw_check = True
            payload = line.split("CHECK_RES:")[1].strip()

            if payload == "IDLE":
                continue

            if payload.startswith("BUSY"):
                saw_busy = True
                for part in payload.split("|"):
                    if part.startswith("USER:"):
                        users.add(part.split(":", 1)[1])

        if not saw_check:
            msg = f"[{prefix}] [ERROR] {name}: no CHECK_RES in output"
            return (prefix, name, zone, "ERROR", msg)

        if not saw_busy:
            msg = f"[{prefix}] [IDLE] {name} ({zone})"
            return (prefix, name, zone, "IDLE", msg)

        msg = f"[{prefix}] [BUSY] {name} ({zone}) users={sorted(users)}"
        return (prefix, name, zone, "BUSY", msg)

    except subprocess.TimeoutExpired:
        msg = f"[{prefix}] [TIMEOUT] {name} ({zone})"
        return (prefix, name, zone, "TIMEOUT", msg)
    except Exception as e:
        msg = f"[{prefix}] [ERROR] {name}: {e}"
        return (prefix, name, zone, "ERROR", msg)


# ---------------- Task dispatcher ----------------


def process_task(task):
    """
    Dispatch task to appropriate handler based on task type.
    - If task has 'state' key -> delete task
    - Otherwise -> check task
    """
    if "state" in task:
        return delete_preempted_tpu(task)
    else:
        return check_single_tpu(task)


# ---------------- Auto-register & mount helpers ----------------


def _read_mount_state():
    """Return (mounted_set, mounting_set) from mounted.json."""
    if not os.path.exists(MOUNTED_FILE):
        return set(), set()
    try:
        with open(MOUNTED_FILE, "r") as f:
            data = json.load(f)
        return set(data.get("mounted", [])), set(data.get("mounting", []))
    except (OSError, json.JSONDecodeError):
        return set(), set()


def _write_mount_state_locked(mounted, mounting):
    """Write mounted.json with file lock to avoid races between workers."""
    fd = None
    try:
        fd = os.open(MOUNTED_FILE, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        content = json.dumps(
            {"mounted": sorted(mounted), "mounting": sorted(mounting)}, indent=2
        )
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, content.encode())
    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _read_mount_state_locked():
    """Read mounted.json under file lock (for use inside workers)."""
    fd = None
    try:
        fd = os.open(MOUNTED_FILE, os.O_RDWR | os.O_CREAT)
        fcntl.flock(fd, fcntl.LOCK_EX)
        raw = os.read(fd, 1 << 20).decode()
        if not raw.strip():
            data = {}
        else:
            data = json.loads(raw)
        return set(data.get("mounted", [])), set(data.get("mounting", []))
    except (OSError, json.JSONDecodeError):
        return set(), set()
    finally:
        if fd is not None:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def _get_mount_tag(name, mounted_set, mounting_set):
    if name in mounted_set:
        return "[mounted]"
    if name in mounting_set:
        return "[mounting]"
    return "[not mounted]"


def _is_tpu_registered(name):
    """Return True if the TPU full name appears in the tpu_manager data.json."""
    try:
        with open(TPU_MANAGER_DATA_PATH, "r") as f:
            data = json.load(f)
        for tpu_list in data.get("all_tpus", {}).values():
            if name in tpu_list:
                return True
        return False
    except (OSError, json.JSONDecodeError):
        return False


def _should_skip_auto_register_mount(name):
    lowered_name = name.lower()
    return any(keyword in lowered_name for keyword in AUTO_REGISTER_MOUNT_SKIP_KEYWORDS)


def _do_mount_single(args):
    """
    Worker function: register (if needed) + mount one TPU.
    Suppresses stdout. Updates mounted.json atomically when done.
    """
    name, zone = args

    if _should_skip_auto_register_mount(name):
        logging.info(f"[AUTO] Skip auto register/mount for TPU: {name}")
        return

    if TPU_MANAGER_DIR not in sys.path:
        sys.path.insert(0, TPU_MANAGER_DIR)

    devnull = open(os.devnull, "w")
    saved_stdout = sys.stdout

    # Register if needed
    if not _is_tpu_registered(name):
        try:
            sys.stdout = devnull
            from utils.logger import register_tpu_and_write_spreadsheet

            register_tpu_and_write_spreadsheet(name, zone, spot=True)
        except Exception:
            pass
        finally:
            sys.stdout = saved_stdout

    # Mount
    mount_ok = False
    try:
        sys.stdout = devnull
        from utils.operate import mount_disk

        result = mount_disk(name)
        mount_ok = result == "success"
    except Exception:
        pass
    finally:
        sys.stdout = saved_stdout
        devnull.close()

    # Update mounted.json: move from mounting -> mounted (or just remove from mounting)
    mounted, mounting = _read_mount_state_locked()
    mounting.discard(name)
    if mount_ok:
        mounted.add(name)
    _write_mount_state_locked(mounted, mounting)


def _spawn_mount_workers(tpus_to_mount):
    """
    Fork a child process that mounts all given TPUs in parallel.
    Caller must have already marked them as 'mounting' in mounted.json.
    Parent returns immediately.
    """
    logging.info(
        f"[AUTO] Spawning background mounts for {len(tpus_to_mount)} TPU(s)..."
    )

    pid = os.fork()
    if pid != 0:
        # Parent: return immediately
        return

    # Child process: mount all in parallel, then exit
    try:
        # Detach from parent's session and close inherited FDs
        # so wrap_master's subprocess.run() doesn't hang waiting on pipes
        os.setsid()
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull_fd, 0)  # stdin
        os.dup2(devnull_fd, 1)  # stdout
        os.dup2(devnull_fd, 2)  # stderr
        os.close(devnull_fd)
        # Re-setup logging to file only (console handler is now /dev/null)
        for h in logging.getLogger().handlers[:]:
            logging.getLogger().removeHandler(h)
        fh = logging.FileHandler(LOG_PATH)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)

        with Pool(min(MAX_WORKERS, len(tpus_to_mount))) as pool:
            pool.map(_do_mount_single, tpus_to_mount)
    except Exception:
        pass
    finally:
        os._exit(0)


# ---------------- Main runner ----------------


def run_audit_all(prefixes):
    logging.info("=== TPU idle/busy audit start ===")

    # Phase 1: list all TPUs in all zones (parallel by zone)
    with Pool(MAX_WORKERS) as pool:
        zone_results = pool.map(list_tpus_in_zone, ZONES)

    # Separate active TPUs and PREEMPTED TPUs
    all_tpus = []
    all_preempted = []
    for active_list, preempted_list in zone_results:
        all_tpus.extend(active_list)
        all_preempted.extend(preempted_list)

    if not all_tpus and not all_preempted:
        logging.info("No TPU found.")
        return

    # Assign prefix (including OTHER) for active TPUs
    for t in all_tpus:
        t["prefix"] = assign_prefix(t["name"], prefixes)

    # Phase 2: Delete PREEMPTED TPUs + Check active TPUs in SAME pool (parallel)
    all_tasks = []

    # Add delete tasks
    if all_preempted:
        logging.info(
            f"Found {len(all_preempted)} PREEMPTED TPU(s), deleting in parallel with checks..."
        )
        all_tasks.extend(all_preempted)

    # Add check tasks
    all_tasks.extend(all_tpus)

    if not all_tasks:
        logging.info("No tasks to execute.")
        return

    # Execute all tasks in parallel: deletes + checks
    with Pool(MAX_WORKERS) as pool:
        all_results = pool.map(process_task, all_tasks)

    # Separate delete results and check results
    delete_results = []
    check_results = []

    for i, result in enumerate(all_results):
        if i < len(all_preempted):
            # This is a delete result
            delete_results.append(result)
        else:
            # This is a check result
            check_results.append(result)

    # Phase 2.5: scan lock directory once, then mark reserved idle TPUs.
    reservations = collect_recent_reservations()
    if reservations:
        marked_results = []
        for prefix, name, zone, status, msg in check_results:
            if status == "IDLE" and name in reservations:
                user = reservations[name]
                status = "RESERVED"
                msg = f"[{prefix}] [RESERVED] {name} ({zone}) reserved by {user}"
            marked_results.append((prefix, name, zone, status, msg))
        check_results = marked_results

    # Log deletion summary if any
    if delete_results:
        success = sum(1 for r in delete_results if r[0] == "DELETE_SUCCESS")
        failed = len(delete_results) - success
        logging.info(f"[DELETE] Summary: {success} deleted, {failed} failed")

    # Phase 3: summary by prefix (including OTHER if any)
    by_prefix = defaultdict(list)
    for r in check_results:
        by_prefix[r[0]].append(r)

    # Determine which IDLE TPUs need mounting, mark them as "mounting" BEFORE printing
    pre_mounted, pre_mounting = _read_mount_state()
    tpus_to_mount = []
    for prefix, name, zone, status, msg in check_results:
        if (
            status == "IDLE"
            and name not in pre_mounted
            and name not in pre_mounting
            and not _should_skip_auto_register_mount(name)
        ):
            tpus_to_mount.append((name, zone))

    # Mark as "mounting" in mounted.json so the SUMMARY below picks it up
    if tpus_to_mount:
        for name, _zone in tpus_to_mount:
            pre_mounting.add(name)
        _write_mount_state_locked(pre_mounted, pre_mounting)

    # Re-read to get the freshest state (including what we just marked)
    mounted_tpus, mounting_tpus = _read_mount_state()

    logging.info("========== SUMMARY ==========")

    # Ensure we print in requested order + OTHER at the end (only if exists)
    ordered_prefixes = list(prefixes)
    if by_prefix.get(OTHER_PREFIX):
        ordered_prefixes.append(OTHER_PREFIX)

    total_all = 0
    idle_all = 0

    for idx, pfx in enumerate(ordered_prefixes):
        items = by_prefix.get(pfx, [])
        if not items:
            continue

        total = len(items)
        idle = sum(1 for x in items if x[3] == "IDLE")
        reserved = sum(1 for x in items if x[3] == "RESERVED")
        busy = sum(1 for x in items if x[3] == "BUSY")
        bad = sum(1 for x in items if x[3] in {"ERROR", "TIMEOUT", "SSH_FAIL"})

        total_all += total
        idle_all += idle

        # Divider between groups
        if idx != 0:
            logging.info("-------")

        header = "[OTHER]" if pfx == OTHER_PREFIX else f"[{pfx}]"
        reserved_part = f", reserved {reserved}" if reserved else ""
        logging.info(
            f"{header} total {total}, idle {idle}{reserved_part}, busy {busy}, bad {bad}"
        )

        # Per-TPU lines
        for _, name, _, _, msg in items:
            mount_tag = _get_mount_tag(name, mounted_tpus, mounting_tpus)
            logging.info(f"{msg}  {mount_tag}")

    logging.info("-------")
    logging.info(f"[ALL] total {total_all}, idle {idle_all}")

    # Spawn background mount processes (marks already written above)
    if tpus_to_mount:
        _spawn_mount_workers(tpus_to_mount)


if __name__ == "__main__":
    t0 = time.time()
    run_audit_all(PREFIXES)
    logging.info(f"Audit finished in {time.time() - t0:.2f}s")

    # Optional periodic run:
    # while True:
    #     start = time.time()
    #     run_audit_all(PREFIXES)
    #     elapsed = time.time() - start
    #     if elapsed < CHECK_INTERVAL:
    #         time.sleep(CHECK_INTERVAL - elapsed)
