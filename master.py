import time
import subprocess
import logging
import os
import sys
import json
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

# Sort modes for `tou` output grouping
SORT_BY_PREFIX = "prefix"
SORT_BY_TYPE = "type"
VALID_SORTS = (SORT_BY_PREFIX, SORT_BY_TYPE)

# TPU type extraction (e.g. kmh-tpuvm-v5p-64-spot-... -> "v5p-64")
OTHER_TYPE = "__OTHER_TYPE__"
TPU_TYPE_REGEX = re.compile(r"kmh-tpuvm-(v\d+[a-z]?-\d+)")
TYPE_PARSE_REGEX = re.compile(r"v(\d+)([a-z]?)-(\d+)")

LOG_PATH = "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/tpu_enforcer.log"
LOCK_DIR = "/kmh-nfs-ssd-us-mount/code/qiao/tpu_lock"
LOCK_EXPIRE_SECONDS = 30 * 60

TPU_MANAGER_DIR = "/kmh-nfs-ssd-us-mount/code/zhichengjiang/working/xibo_tpu_manager"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_COUNT_FILE = os.path.join(SCRIPT_DIR, ".tpu_timeout_counts.json")
# Delete a TPU only after this many consecutive audit runs where it reports TIMEOUT.
TIMEOUT_DELETE_THRESHOLD = 10
MOUNT_LOG_DIR = os.path.join(SCRIPT_DIR, "mount_logs")

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


# setup_logging() is called from __main__ only — wrap_master imports this module
# just to use format_summary, and we don't want to open the file log on every cache hit.


# ---------------- Utilities ----------------

def _load_timeout_counts() -> dict:
    """Per-TPU consecutive [TIMEOUT] streak (persisted across runs)."""
    try:
        with open(TIMEOUT_COUNT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {str(k): int(v) for k, v in data.items()}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}


def _save_timeout_counts(counts: dict) -> None:
    try:
        with open(TIMEOUT_COUNT_FILE, "w", encoding="utf-8") as f:
            json.dump(counts, f, indent=0, sort_keys=True)
    except OSError:
        pass


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


def extract_tpu_type(name: str) -> str:
    """Extract chip-type-and-size from a TPU name (e.g. v5p-64, v6e-8)."""
    m = TPU_TYPE_REGEX.search(name)
    return m.group(1) if m else OTHER_TYPE


def tpu_type_sort_key(t: str):
    """Sort key for TPU types: by (generation, chip-suffix, size)."""
    if t == OTHER_TYPE:
        return (10**6, "z", 10**6, t)
    m = TYPE_PARSE_REGEX.match(t)
    if not m:
        return (10**6 - 1, "z", 10**6 - 1, t)
    return (int(m.group(1)), m.group(2), int(m.group(3)), t)


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


def _new_record(tpu):
    return {
        "name": tpu["name"],
        "zone": tpu["zone"],
        "prefix": tpu["prefix"],
        "status": None,
        "disk_mounted": None,
        "users": None,        # populated for BUSY
        "reserved_by": None,  # populated for RESERVED (set later in run_audit_all)
        "error_msg": None,    # populated for ERROR / SSH_FAIL
    }


def check_single_tpu(tpu: dict):
    """tpu dict: {"name":..., "zone":..., "prefix":...}. Returns a record dict."""
    name = tpu["name"]
    zone = tpu["zone"]
    record = _new_record(tpu)

    remote_cmd = (
        "PID=$(sudo lsof -t /dev/accel* /dev/vfio/* 2>/dev/null | head -n 1); "
        'if [ -z "$PID" ]; then echo "CHECK_RES:IDLE"; '
        'else TPU_USER=$(ps -o user= -p "$PID"); echo "CHECK_RES:BUSY|USER:$TPU_USER"; fi; '
        '[ -f /home/sqa/.disk_mounted ] && echo "MOUNT_RES:MOUNTED" || echo "MOUNT_RES:NOT_MOUNTED"'
    )

    ssh_cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "ssh", name,
        "--zone", zone, "--worker=all", "--ssh-flag=-n", "--command", remote_cmd,
    ]

    try:
        res = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=40)

        if res.returncode != 0:
            record["status"] = "SSH_FAIL"
            record["error_msg"] = res.stderr.strip()
            return record

        users = set()
        saw_check = False
        saw_busy = False
        disk_mounted = None  # any worker reporting NOT_MOUNTED → False
        for line in res.stdout.splitlines():
            if "MOUNT_RES:" in line:
                payload = line.split("MOUNT_RES:")[1].strip()
                if payload == "NOT_MOUNTED":
                    disk_mounted = False
                elif payload == "MOUNTED" and disk_mounted is None:
                    disk_mounted = True
                continue
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

        record["disk_mounted"] = disk_mounted

        if not saw_check:
            record["status"] = "ERROR"
            record["error_msg"] = "no CHECK_RES in output"
            return record

        if not saw_busy:
            record["status"] = "IDLE"
            return record

        record["status"] = "BUSY"
        record["users"] = sorted(users)
        return record

    except subprocess.TimeoutExpired:
        record["status"] = "TIMEOUT"
        return record
    except Exception as e:
        record["status"] = "ERROR"
        record["error_msg"] = str(e)
        return record


def format_record_msg(record, group_label):
    """Build a single audit-line string from a record + the group label to display."""
    name = record["name"]
    zone = record["zone"]
    status = record["status"]
    base = f"[{group_label}] [{status}]"
    if status == "IDLE":
        disk_str = (
            "MOUNTED" if record["disk_mounted"] is True
            else "NOT_MOUNTED" if record["disk_mounted"] is False
            else "UNKNOWN"
        )
        return f"{base} {name} ({zone}) [{disk_str}]"
    if status == "BUSY":
        return f"{base} {name} ({zone}) users={record['users']}"
    if status == "RESERVED":
        disk_str = (
            "MOUNTED" if record["disk_mounted"] is True
            else "NOT_MOUNTED" if record["disk_mounted"] is False
            else "UNKNOWN"
        )
        return f"{base} {name} ({zone}) reserved by {record['reserved_by']} [{disk_str}]"
    if status in ("TIMEOUT", "TIMEOUT_DELETED"):
        return f"{base} {name} ({zone})"
    if status in ("SSH_FAIL", "ERROR"):
        return f"{base} {name}: {record['error_msg']}"
    return f"{base} {name} ({zone})"


def _within_group_sort_key(r):
    """Per-row sort key inside each type group: IDLE first, then RESERVED/BUSY by user, then bad."""
    status = r["status"]
    if status == "IDLE":
        # MOUNTED first (immediately usable), then NOT_MOUNTED, then UNKNOWN
        mount_rank = 0 if r["disk_mounted"] is True else (1 if r["disk_mounted"] is False else 2)
        return (0, mount_rank, "", r["name"])
    if status == "RESERVED":
        return (1, 0, r.get("reserved_by") or "", r["name"])
    if status == "BUSY":
        users = r.get("users") or []
        first_user = users[0] if users else ""
        return (2, 0, first_user, r["name"])
    # TIMEOUT / SSH_FAIL / ERROR / TIMEOUT_DELETED
    return (3, 0, status, r["name"])


def format_summary(records, sort_by, prefixes):
    """Format a list of records as audit-summary lines (no logging side effect)."""
    if sort_by not in VALID_SORTS:
        raise ValueError(f"sort_by must be one of {VALID_SORTS}, got {sort_by!r}")

    if sort_by == SORT_BY_TYPE:
        by_group = defaultdict(list)
        for r in records:
            by_group[extract_tpu_type(r["name"])].append(r)
        ordered_groups = sorted(by_group.keys(), key=tpu_type_sort_key)
        is_other = lambda g: g == OTHER_TYPE
        # Type mode: within each group, sort IDLE → RESERVED → BUSY → bad,
        # so picking a card just means scanning the top of the relevant section.
        for g in by_group:
            by_group[g].sort(key=_within_group_sort_key)
    else:
        by_group = defaultdict(list)
        for r in records:
            by_group[r["prefix"]].append(r)
        ordered_groups = list(prefixes)
        if by_group.get(OTHER_PREFIX):
            ordered_groups.append(OTHER_PREFIX)
        is_other = lambda g: g == OTHER_PREFIX

    lines = ["========== SUMMARY =========="]
    total_all = 0
    idle_all = 0

    for idx, g in enumerate(ordered_groups):
        items = by_group.get(g, [])
        if not items:
            continue

        total = len(items)
        idle = sum(1 for x in items if x["status"] == "IDLE")
        reserved = sum(1 for x in items if x["status"] == "RESERVED")
        busy = sum(1 for x in items if x["status"] == "BUSY")
        bad = sum(1 for x in items if x["status"] in {"ERROR", "TIMEOUT", "SSH_FAIL", "TIMEOUT_DELETED"})

        total_all += total
        idle_all += idle

        if idx != 0:
            lines.append("-------")

        label = "OTHER" if is_other(g) else g
        reserved_part = f", reserved {reserved}" if reserved else ""
        lines.append(f"[{label}] total {total}, idle {idle}{reserved_part}, busy {busy}, bad {bad}")
        for r in items:
            lines.append(format_record_msg(r, label))

    lines.append("-------")
    lines.append(f"[ALL] total {total_all}, idle {idle_all}")
    return lines


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


def _should_skip_auto_register_mount(name):
    lowered_name = name.lower()
    return any(keyword in lowered_name for keyword in AUTO_REGISTER_MOUNT_SKIP_KEYWORDS)



def _do_mount_single(args):
    """
    Worker function: mount one TPU directly (no registration needed).
    Zone is passed in explicitly so data.json lookup is bypassed entirely.
    Whether the mount succeeded is reflected on the TPU itself (/home/sqa/.disk_mounted).
    stdout+stderr are written to mount_logs/{name}_{zone}.txt for debugging.
    """
    name, zone = args

    os.makedirs(MOUNT_LOG_DIR, exist_ok=True)
    log_path = os.path.join(MOUNT_LOG_DIR, f"{name}_{zone}.txt")
    cmd = ["python", os.path.join(TPU_MANAGER_DIR, "tpu.py"), "mount-disk", name, f"--zone={zone}"]

    def _write_log(content):
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError:
            pass

    if _should_skip_auto_register_mount(name):
        _write_log(f"SKIPPED: {name} matches AUTO_REGISTER_MOUNT_SKIP_KEYWORDS\n")
        return

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1000)
        _write_log(
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )
        if result.returncode == 0:
            logging.info(f"[AUTO] mount-disk {name} ({zone}): SUCCESS")
        else:
            logging.info(
                f"[AUTO] mount-disk {name} ({zone}): FAILED (rc={result.returncode}), "
                f"see {log_path}"
            )
    except subprocess.TimeoutExpired:
        _write_log(f"cmd: {' '.join(cmd)}\nTIMEOUT after 1000s\n")
        logging.info(f"[AUTO] mount-disk {name} ({zone}): TIMEOUT")
    except Exception as e:
        _write_log(f"cmd: {' '.join(cmd)}\nERROR: {e}\n")
        logging.info(f"[AUTO] mount-disk {name} ({zone}): ERROR: {e}")


def _spawn_mount_workers(tpus_to_mount):
    """
    Fork a child process that mounts all given TPUs in parallel.
    Parent returns immediately (non-blocking).
    """
    logging.info(
        f"[AUTO] Spawning background mounts for {len(tpus_to_mount)} TPU(s)..."
    )

    pid = os.fork()
    if pid != 0:
        return

    try:
        os.setsid()
        devnull_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(devnull_fd, 0)
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
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
    """Run one audit pass. Returns a list of TPU records (no formatting/printing of the summary).

    The audit-process status messages (timeout streak, delete summary, AUTO mount spawn)
    still go to the file/console logger as before. Only the per-prefix SUMMARY block
    is no longer printed here — callers format that themselves via `format_summary`,
    so cached records can be re-grouped on demand without re-running the audit.
    """
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
        return []

    # Assign prefix (including OTHER) for active TPUs
    for t in all_tpus:
        t["prefix"] = assign_prefix(t["name"], prefixes)

    # Phase 2: Delete PREEMPTED TPUs + Check active TPUs in SAME pool (parallel)
    all_tasks = []
    if all_preempted:
        logging.info(
            f"Found {len(all_preempted)} PREEMPTED TPU(s), deleting in parallel with checks..."
        )
        all_tasks.extend(all_preempted)
    all_tasks.extend(all_tpus)

    if not all_tasks:
        logging.info("No tasks to execute.")
        return []

    with Pool(MAX_WORKERS) as pool:
        all_results = pool.map(process_task, all_tasks)

    # Separate delete results (tuples) from check results (record dicts)
    delete_results = []
    check_results = []
    for i, result in enumerate(all_results):
        if i < len(all_preempted):
            delete_results.append(result)
        else:
            check_results.append(result)

    # Phase 2.3: Consecutive TIMEOUT counter; delete only after TIMEOUT_DELETE_THRESHOLD streak.
    seen_names = {r["name"] for r in check_results}
    timeout_counts = {k: v for k, v in _load_timeout_counts().items() if k in seen_names}

    for r in check_results:
        if r["status"] == "TIMEOUT":
            c = timeout_counts.get(r["name"], 0) + 1
            timeout_counts[r["name"]] = c
            logging.info(
                f"[TIMEOUT_COUNT] {r['name']} ({r['zone']}): consecutive_timeout_count={c} "
                f"(delete when >= {TIMEOUT_DELETE_THRESHOLD})"
            )
        else:
            timeout_counts.pop(r["name"], None)

    persistent_timeout_tasks = [
        {"name": r["name"], "zone": r["zone"], "state": "TIMEOUT"}
        for r in check_results
        if r["status"] == "TIMEOUT"
        and timeout_counts.get(r["name"], 0) >= TIMEOUT_DELETE_THRESHOLD
    ]
    if persistent_timeout_tasks:
        logging.info(
            f"[TIMEOUT_DELETE] {len(persistent_timeout_tasks)} TPU(s) reached "
            f"{TIMEOUT_DELETE_THRESHOLD} consecutive TIMEOUT(s), deleting: "
            + ", ".join(t["name"] for t in persistent_timeout_tasks)
        )
        with Pool(min(MAX_WORKERS, len(persistent_timeout_tasks))) as pool:
            timeout_del_results = pool.map(delete_preempted_tpu, persistent_timeout_tasks)
        deleted_names = set()
        for del_status, del_name, del_zone in timeout_del_results:
            logging.info(f"[TIMEOUT_DELETE] {del_name} ({del_zone}): {del_status}")
            if del_status == "DELETE_SUCCESS":
                deleted_names.add(del_name)
                timeout_counts.pop(del_name, None)
        _save_timeout_counts(timeout_counts)
        # Update status for successfully deleted TPUs (mutate records in place)
        for r in check_results:
            if r["name"] in deleted_names:
                r["status"] = "TIMEOUT_DELETED"
    else:
        _save_timeout_counts(timeout_counts)

    # Phase 2.5: scan lock directory once, then mark reserved idle TPUs (mutate in place).
    reservations = collect_recent_reservations()
    if reservations:
        for r in check_results:
            if r["status"] == "IDLE" and r["name"] in reservations:
                r["status"] = "RESERVED"
                r["reserved_by"] = reservations[r["name"]]

    # Log deletion summary if any
    if delete_results:
        success = sum(1 for r in delete_results if r[0] == "DELETE_SUCCESS")
        failed = len(delete_results) - success
        logging.info(f"[DELETE] Summary: {success} deleted, {failed} failed")

    # Collect IDLE/RESERVED TPUs whose disk is not yet mounted and schedule background mounts.
    tpus_to_mount = [
        (r["name"], r["zone"])
        for r in check_results
        if r["status"] in ("IDLE", "RESERVED") and r["disk_mounted"] is False
        and not _should_skip_auto_register_mount(r["name"])
    ]
    if tpus_to_mount:
        _spawn_mount_workers(tpus_to_mount)

    return check_results


if __name__ == "__main__":
    setup_logging()
    import argparse
    parser = argparse.ArgumentParser(description="TPU idle/busy audit (one round).")
    parser.add_argument(
        "--sort",
        choices=VALID_SORTS,
        default=SORT_BY_PREFIX,
        help="Group output by name prefix (default, e.g. [gzy] [llq]) or by TPU type (e.g. [v5p-64] [v6e-8]).",
    )
    parser.add_argument(
        "--records-out",
        default=None,
        metavar="PATH",
        help="If set, also dump the raw audit records as JSON to this path (consumed by wrap_master cache).",
    )
    args = parser.parse_args()
    t0 = time.time()
    records = run_audit_all(PREFIXES)

    if args.records_out and records:
        try:
            with open(args.records_out, "w", encoding="utf-8") as f:
                json.dump({"ts": time.time(), "records": records}, f)
        except OSError as e:
            logging.warning(f"Failed to write records JSON to {args.records_out}: {e}")

    for line in format_summary(records, args.sort, PREFIXES):
        logging.info(line)
    logging.info(f"Audit finished in {time.time() - t0:.2f}s")

    # Optional periodic run:
    # while True:
    #     start = time.time()
    #     run_audit_all(PREFIXES)
    #     elapsed = time.time() - start
    #     if elapsed < CHECK_INTERVAL:
    #         time.sleep(CHECK_INTERVAL - elapsed)
