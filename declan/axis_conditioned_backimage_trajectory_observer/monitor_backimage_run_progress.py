"""Write lightweight progress heartbeats for long BackImage observer runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _latest_npz(response_dir: Path) -> tuple[int, str]:
    paths = list(response_dir.rglob("*.npz")) if response_dir.exists() else []
    if not paths:
        return 0, ""
    latest = max(paths, key=lambda path: path.stat().st_mtime)
    return len(paths), str(latest)


def _gpu_snapshot() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - operational fallback
        return f"nvidia-smi-error={type(exc).__name__}:{exc}"
    text = (completed.stdout or completed.stderr).strip()
    return text.replace("\n", ";")


def _write_line(log_path: Path, line: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def monitor(
    *,
    out_dir: Path,
    pid_file: Path,
    log_path: Path,
    interval_s: float,
    expected_files: int | None,
) -> None:
    response_dir = out_dir / "response_tables"
    while True:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        pid = _read_pid(pid_file)
        alive = _pid_alive(pid)
        count, latest = _latest_npz(response_dir)
        expected_text = str(expected_files) if expected_files is not None else "unknown"
        line = (
            f"{now} pid={pid if pid is not None else 'missing'} "
            f"alive={'yes' if alive else 'no'} npz={count}/{expected_text} "
            f"latest={latest or 'none'} gpu={_gpu_snapshot()}"
        )
        _write_line(log_path, line)
        if not alive:
            return
        if expected_files is not None and count >= expected_files:
            return
        time.sleep(interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pid-file", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--expected-files", type=int, default=None)
    args = parser.parse_args()
    monitor(
        out_dir=args.out_dir,
        pid_file=args.pid_file,
        log_path=args.log,
        interval_s=args.interval_s,
        expected_files=args.expected_files,
    )


if __name__ == "__main__":
    main()
