#!/usr/bin/env python3
"""
Wrapper for master.py: caches output for 5 minutes and uses file locking
to prevent concurrent runs. Second concurrent run waits and then outputs
the first run's result.

Usage:
  --cache true   : Return cached result directly (if any), print cache age
  --cache false  : Wait for lock, run master.py once, update cache
"""
import argparse
import os
import sys
import time
import fcntl
import subprocess

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_SCRIPT = os.path.join(SCRIPT_DIR, "master.py")
CACHE_FILE = os.path.join(SCRIPT_DIR, ".tpu_audit_cache")
LOCK_FILE = os.path.join(SCRIPT_DIR, ".tpu_audit.lock")
CACHE_DURATION = 300  # 5 minutes


def _log(msg: str):
    print(f"[wrap] {msg}", file=sys.stderr)


def _parse_args():
    p = argparse.ArgumentParser(description="Wrapper for master.py with cache option")
    p.add_argument(
        "--cache",
        choices=("true", "false"),
        default="true",
        help="true: return cached result; false: run master.py once",
    )
    return p.parse_args()


def _read_cache():
    """Return (cached_time, output) or None if cache miss/invalid."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        if not content:
            return None
        lines = content.split("\n", 1)
        cached_time = float(lines[0])
        output = lines[1] if len(lines) > 1 else ""
        return (cached_time, output)
    except (ValueError, IndexError, OSError):
        return None


def run_with_cache_true():
    """Return cached result directly, print cache age."""
    entry = _read_cache()
    if entry is None:
        _log("No cache available.")
        sys.exit(1)
    cached_time, output = entry
    age = time.time() - cached_time
    sys.stdout.write(output)
    sys.stdout.flush()
    _log(f"Using cached output (delayed by {age:.0f}s)")


def run_with_cache_false():
    """Wait for lock, run master.py once, update cache."""
    lock_fd = None
    try:
        if not os.path.exists(LOCK_FILE):
            open(LOCK_FILE, "a").close()

        _log("Acquiring lock...")
        lock_fd = os.open(LOCK_FILE, os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _log("Lock acquired.")

        now = time.time()
        _log("Running master.py...")
        result = subprocess.run(
            [sys.executable, MASTER_SCRIPT],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
        )

        output = result.stdout
        if result.stderr:
            output += result.stderr

        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(f"{now}\n{output}")
        except OSError:
            _log("Warning: failed to write cache")

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
        run_with_cache_true()
    else:
        run_with_cache_false()


if __name__ == "__main__":
    main()