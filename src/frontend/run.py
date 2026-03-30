"""Launch Streamlit UI with safe defaults (localhost unless --host is set)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the batchlense Streamlit dashboard.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address (default: 127.0.0.1 — loopback only). "
            "Use 0.0.0.0 to listen on all interfaces (LAN)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for the web server (default: 8501).",
    )
    args = parser.parse_args()

    frontend_dir = Path(__file__).resolve().parent
    repo_root = frontend_dir.parent.parent
    app_path = frontend_dir / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    raise SystemExit(subprocess.run(cmd, cwd=repo_root).returncode)


if __name__ == "__main__":
    main()
