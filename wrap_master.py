#!/usr/bin/env python3
"""
Wrapper for master.py: caches the audit records (JSON, single source of truth)
and formats on every call so any --sort mode is always as fresh as the records.

Usage:
  --cache true   : Format from cached records (default; ~10ms)
  --cache false  : Wait for lock, run master.py once, refresh records cache (~80s)
  --sort prefix  : Group output by name prefix (default; e.g. [gzy] [llq])
  --sort type    : Group output by TPU type (e.g. [v5p-64] [v6e-8])

Cache layout:
  .tpu_audit_records.json  — {"ts": <epoch-seconds>, "records": [<TPU dict>, ...]}
  .tpu_audit.lock          — fcntl lock; only one master.py audit at a time

`yizhitou` (=`while true; do tou --cache false; sleep 120; done`) keeps the records
fresh, so both `tou` and `tout` always serve recent data — no per-mode caches.
"""
import argparse
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_SCRIPT = os.path.join(SCRIPT_DIR, "master.py")
RECORDS_CACHE = os.path.join(SCRIPT_DIR, ".tpu_audit_records.json")
LOCK_FILE = os.path.join(SCRIPT_DIR, ".tpu_audit.lock")
VALID_SORTS = ("prefix", "type")

ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_RESET = "\033[0m"
_IDLE_RE = re.compile(r"(\[IDLE\])\s+(\S+)")
_RESERVED_RE = re.compile(r"(\[RESERVED\])\s+(\S+)")


def _log(msg):
    print(f"[wrap] {msg}", file=sys.stderr)


def _parse_args():
    p = argparse.ArgumentParser(description="Wrapper for master.py with shared records cache.")
    p.add_argument("--cache", choices=("true", "false"), default="true",
                   help="true: format from cached records; false: re-run audit and refresh cache.")
    p.add_argument("--sort", choices=VALID_SORTS, default="prefix",
                   help="Group output by name prefix (default) or by TPU type.")
    return p.parse_args()


def _read_records_cache():
    """Return (cached_time, records_list) or None if missing/invalid."""
    if not os.path.exists(RECORDS_CACHE):
        return None
    try:
        with open(RECORDS_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return (float(data["ts"]), list(data["records"]))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _colorize(line):
    """Wrap [IDLE] / [RESERVED] tags + the next word in green/yellow ANSI (matches master.py's IdleOnlyFormatter)."""
    line = _IDLE_RE.sub(
        lambda m: f"{ANSI_GREEN}{m.group(1)}{ANSI_RESET} {ANSI_GREEN}{m.group(2)}{ANSI_RESET}",
        line,
    )
    line = _RESERVED_RE.sub(
        lambda m: f"{ANSI_YELLOW}{m.group(1)}{ANSI_RESET} {ANSI_YELLOW}{m.group(2)}{ANSI_RESET}",
        line,
    )
    return line


def _ts_prefix(epoch):
    """Build the `YYYY-MM-DD HH:MM:SS,mmm [INFO] ` prefix matching master.py's logger format."""
    s = dt.datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    return f"{s} [INFO] "


def run_with_cache_true(sort_mode):
    """Format the cached records per sort_mode and print."""
    entry = _read_records_cache()
    if entry is None:
        _log(f"No cache at {RECORDS_CACHE}. Run `tou --cache false` once (~80s) to populate.")
        sys.exit(1)
    cached_time, records = entry
    age = time.time() - cached_time

    # master.py is in the same dir; importing it lets us reuse format_summary directly.
    sys.path.insert(0, SCRIPT_DIR)
    import master

    prefix = _ts_prefix(cached_time)
    for line in master.format_summary(records, sort_mode, master.PREFIXES):
        sys.stdout.write(prefix + _colorize(line) + "\n")
    sys.stdout.flush()
    _log(f"Using cached records (audit was {age:.0f}s ago, sort={sort_mode})")


def run_with_cache_false(sort_mode):
    """Acquire lock, re-run master.py, refresh records cache, and print this run's output."""
    lock_fd = None
    try:
        if not os.path.exists(LOCK_FILE):
            open(LOCK_FILE, "a").close()
        _log("Acquiring lock...")
        lock_fd = os.open(LOCK_FILE, os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _log("Lock acquired.")

        _log(f"Running master.py --sort {sort_mode} (refreshing records cache)...")
        result = subprocess.run(
            [sys.executable, MASTER_SCRIPT,
             "--sort", sort_mode,
             "--records-out", RECORDS_CACHE],
            capture_output=True, text=True, cwd=SCRIPT_DIR,
        )

        output = result.stdout
        if result.stderr:
            output += result.stderr

        sys.stdout.write(output)
        sys.stdout.flush()
        sys.exit(result.returncode if result.returncode else 0)

    except KeyboardInterrupt:
        _log("Interrupted (Ctrl+C), releasing lock...")
        raise
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                _log("Lock released.")
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass


def main():
    args = _parse_args()
    if args.cache == "true":
        run_with_cache_true(args.sort)
    else:
        run_with_cache_false(args.sort)


if __name__ == "__main__":
    main()
