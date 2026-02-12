#!/usr/bin/env python3
"""
Wrapper for master.py: caches output for 5 minutes and uses file locking
to prevent concurrent runs. Second concurrent run waits and then outputs
the first run's result.
"""
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


def main():
    lock_fd = None
    try:
        # Ensure lock file exists
        if not os.path.exists(LOCK_FILE):
            open(LOCK_FILE, "a").close()

        _log("Acquiring lock...")
        lock_fd = os.open(LOCK_FILE, os.O_RDWR)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        _log("Lock acquired.")

        now = time.time()

        # Check cache: if run within CACHE_DURATION, use cached output
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                if content:
                    lines = content.split("\n", 1)
                    cached_time = float(lines[0])
                    age = now - cached_time
                    if age < CACHE_DURATION:
                        _log(f"Using cached output (from {age:.0f}s ago)")
                        cached_output = lines[1] if len(lines) > 1 else ""
                        sys.stdout.write(cached_output)
                        sys.stdout.flush()
                        return
            except (ValueError, IndexError, OSError):
                pass

        _log("Cache miss or expired, running master.py...")
        result = subprocess.run(
            [sys.executable, MASTER_SCRIPT],
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
        )

        output = result.stdout
        if result.stderr:
            output += result.stderr

        _log("Saving output to cache...")
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


if __name__ == "__main__":
    main()
