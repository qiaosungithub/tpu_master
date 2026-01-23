import time
import subprocess
import logging
from multiprocessing import Pool
from collections import defaultdict
import re

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

# 需要并行审计的“人名 / key”
PREFIXES = ["llq", "keya", "dmy", "gzy", "kangyang"]

# 未匹配任何 prefix 的最后一类
OTHER_PREFIX = "__OTHER__"

LOG_PATH = "/kmh-nfs-ssd-us-mount/code/qiao/work/tpu_dls/tpu_enforcer.log"

# ===========================================


# ---------- Console coloring: ONLY [IDLE] + TPU name ----------
class IdleOnlyFormatter(logging.Formatter):
    GREEN = "\033[32m"
    RESET = "\033[0m"

    # Matches: [IDLE] <tpu-name>
    IDLE_PATTERN = re.compile(r"(\[IDLE\])\s+([^\s]+)")

    def format(self, record):
        msg = super().format(record)

        def repl(m):
            return (
                f"{self.GREEN}{m.group(1)}{self.RESET} "
                f"{self.GREEN}{m.group(2)}{self.RESET}"
            )

        return self.IDLE_PATTERN.sub(repl, msg)


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
        "PREEMPTED", "TERMINATED", "CREATING",
        "DELETING", "REPAIRING", "STOPPED"
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


def list_tpus_in_zone(zone: str):
    """
    List all ACTIVE TPUs (after skip policy) in a zone.
    Return: list of {name, zone}
    """
    results = []
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "list",
        "--zone", zone,
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

            # apply skip policy
            if should_skip_tpu(name, zone, state):
                continue

            results.append({"name": name, "zone": zone})

    except Exception as e:
        logging.info(f"[ALL] {zone}: list failed ({e})")

    return results


def assign_prefix(name: str, prefixes):
    """
    Assign a TPU name to the first matching prefix (priority by PREFIXES order).
    If no match, return OTHER_PREFIX.
    """
    for pfx in prefixes:
        if pfx in name:
            return pfx
    return OTHER_PREFIX


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
        "gcloud", "compute", "tpus", "tpu-vm", "ssh",
        name,
        "--zone", zone,
        "--worker=all",
        "--ssh-flag=-n",
        "--command", remote_cmd,
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
        msg = f"[{prefix}] [TIMEOUT] {name}"
        return (prefix, name, zone, "TIMEOUT", msg)
    except Exception as e:
        msg = f"[{prefix}] [ERROR] {name}: {e}"
        return (prefix, name, zone, "ERROR", msg)


# ---------------- Main runner ----------------

def run_audit_all(prefixes):
    logging.info("=== TPU idle/busy audit start ===")

    # Phase 1: list all TPUs in all zones (parallel by zone)
    with Pool(MAX_WORKERS) as pool:
        zone_lists = pool.map(list_tpus_in_zone, ZONES)

    all_tpus = []
    for sub in zone_lists:
        all_tpus.extend(sub)

    if not all_tpus:
        logging.info("No active TPU found.")
        return

    # Assign prefix (including OTHER) after listing
    for t in all_tpus:
        t["prefix"] = assign_prefix(t["name"], prefixes)

    # Phase 2: ssh check all TPUs (parallel)
    with Pool(MAX_WORKERS) as pool:
        results = pool.map(check_single_tpu, all_tpus)

    # Phase 3: summary by prefix (including OTHER if any)
    by_prefix = defaultdict(list)
    for r in results:
        by_prefix[r[0]].append(r)

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
        busy = sum(1 for x in items if x[3] == "BUSY")
        bad = sum(1 for x in items if x[3] in {"ERROR", "TIMEOUT", "SSH_FAIL"})

        total_all += total
        idle_all += idle

        # Divider between groups
        if idx != 0:
            logging.info("-------")

        header = "[OTHER]" if pfx == OTHER_PREFIX else f"[{pfx}]"
        logging.info(f"{header} total {total}, idle {idle}, busy {busy}, bad {bad}")

        # Per-TPU lines
        for _, _, _, _, msg in items:
            logging.info(msg)

    logging.info("-------")
    logging.info(f"[ALL] total {total_all}, idle {idle_all}")


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
