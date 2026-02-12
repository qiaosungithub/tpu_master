#!/usr/bin/env python3
"""
Delete all IDLE TPUs of a given type.

Usage:
  python delete_idle_by_type.py <tpu_type>
  # e.g.  python delete_idle_by_type.py v5p-8
  #       python delete_idle_by_type.py v6e-8

Runs the same logic as `tou` (wrap_master.py) to get current TPU list, parses [IDLE] lines,
filters by type (substring match in VM name), then deletes each with gcloud.
Asks for confirmation before deleting.
"""

import os
import re
import subprocess
import sys
from multiprocessing import Pool


# Normalize shorthand to substring that appears in VM names (e.g. v5-8 -> v5p-8 for common case)
# Strip ANSI escape sequences so regex can match (master.py colors [IDLE] and TPU name).
def _strip_ansi(text: str) -> str:
    return re.sub(r"\033\[[0-9;]*m", "", text)


TYPE_ALIASES = {
    "v5-8": "v5p-8",
    "v6-8": "v6e-8",
    "v5-16": "v5e-16",
    "v6-32": "v6e-32",
    "v6-64": "v6e-64",
    "v5-64": "v5p-64",
    "v6-128": "v6e-128",
    "v5-128": "v5p-128",
}

DELETE_WORKERS = 8  # parallel gcloud delete processes
DELETE_TIMEOUT = 120


def _delete_one(tpu: dict) -> tuple:
    """Delete a single TPU. Returns (status, name, zone, message).
    status in ('ok', 'timeout', 'fail').
    """
    name, zone = tpu["name"], tpu["zone"]
    cmd = [
        "gcloud", "compute", "tpus", "tpu-vm", "delete",
        name,
        "--zone", zone,
        "--quiet",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=DELETE_TIMEOUT)
        return ("ok", name, zone, None)
    except subprocess.TimeoutExpired:
        return ("timeout", name, zone, None)
    except Exception as e:
        return ("fail", name, zone, str(e))


def main():
    if len(sys.argv) < 2:
        print("Usage: python delete_idle_by_type.py <tpu_type>")
        print("  e.g.  delete_idle_by_type.py v5p-8   # deletes all IDLE v5p-8 TPUs")
        print("        delete_idle_by_type.py v6e-8")
        sys.exit(1)

    raw_type = sys.argv[1].strip().lower()
    tpu_type = TYPE_ALIASES.get(raw_type, raw_type)

    # Run wrap_master.py (same as `tou`) so we can capture stdout/stderr reliably.
    # Using subprocess.run(["tou"]) often fails to capture output when tou is a shell alias
    # or when stdout is not a TTY.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    wrap_master = os.path.join(script_dir, "wrap_master.py")
    if not os.path.isfile(wrap_master):
        print(f"Error: wrap_master.py not found at {wrap_master}")
        sys.exit(1)
    print("Running TPU list (wrap_master.py)...")
    try:
        result = subprocess.run(
            [sys.executable, wrap_master],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=script_dir,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        out = result.stdout + result.stderr
        # print(out)
    except subprocess.TimeoutExpired:
        print("Error: TPU list command timed out.")
        sys.exit(1)

    # Parse [IDLE] lines:  ... [IDLE] kmh-tpuvm-v5p-8-spot-xxx (us-central1-a)
    # Strip ANSI color codes from console output so the regex can match.
    pattern = re.compile(r"\[IDLE\]\s+(\S+)\s+\(([^)]+)\)")
    idle_list = []
    for line in out.splitlines():
        clean = _strip_ansi(line)
        m = pattern.search(clean)
        if not m:
            continue
        name, zone = m.group(1), m.group(2)
        # ignore a nopre tpu
        if 'nopre' in name:
            continue
        if tpu_type in name:
            idle_list.append({"name": name, "zone": zone})

    if not idle_list:
        print(f"No IDLE TPUs matching type '{tpu_type}' (input: {raw_type}).")
        sys.exit(0)

    print(f"Found {len(idle_list)} IDLE TPU(s) matching '{tpu_type}':")
    for t in idle_list:
        print(f"  {t['name']} ({t['zone']})")
    print()
    confirm = input("Delete all of them? [y/N]: ").strip().lower()
    if confirm != "y" and confirm != "yes":
        print("Aborted.")
        sys.exit(0)

    print(f"Deleting {len(idle_list)} TPUs in parallel (workers={DELETE_WORKERS})...")
    with Pool(DELETE_WORKERS) as pool:
        results = pool.map(_delete_one, idle_list)
    success = failed = 0
    for status, name, zone, msg in results:
        if status == "ok":
            print(f"  Deleted: {name} ({zone})")
            success += 1
        elif status == "timeout":
            print(f"  Timeout: {name} ({zone})")
            failed += 1
        else:
            print(f"  Failed:  {name} ({zone}) — {msg}")
            failed += 1
    print(f"\nDone: {success} deleted, {failed} failed.")


if __name__ == "__main__":
    main()