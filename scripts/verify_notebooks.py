"""Execute source notebooks with the project's Python, preserving clean sources.

No kernel is installed globally. A temporary kernel definition explicitly names
the project interpreter; executed copies and a verification summary are written
under results/notebooks. Missing training results remain unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("01_EDA.ipynb", "02_Model_Analysis.ipynb")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _default_python() -> Path:
    windows = ROOT / ".venv" / "Scripts" / "python.exe"
    posix = ROOT / ".venv" / "bin" / "python"
    return windows if windows.is_file() else posix if posix.is_file() else Path(sys.executable)


def execute_notebooks(python: Path, output_dir: Path, timeout: int, names: list[str]) -> list[dict]:
    """Execute in fresh kernels and verify source notebook bytes stay unchanged."""
    if output_dir.resolve() == (ROOT / "notebooks").resolve():
        raise ValueError("Executed copies must not overwrite source notebooks; use results/notebooks or another output directory.")
    import nbformat
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="brain-mri-notebook-kernel-") as temporary:
        kernel_dir = Path(temporary) / "brain-mri-project"
        kernel_dir.mkdir()
        kernel = {"argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                  "display_name": "Brain MRI project environment", "language": "python"}
        (kernel_dir / "kernel.json").write_text(json.dumps(kernel), encoding="utf-8")
        for name in names:
            source = ROOT / "notebooks" / name
            if not source.is_file():
                raise FileNotFoundError(f"Source notebook not found: {source}")
            original = source.read_bytes()
            notebook = nbformat.read(source, as_version=4)
            nbformat.validate(notebook)
            manager = AsyncKernelManager(kernel_name="brain-mri-project",
                                         kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
            client = NotebookClient(notebook, km=manager, timeout=timeout, allow_errors=False,
                                    record_timing=True, resources={"metadata": {"path": str(ROOT)}})
            started = time.perf_counter()
            record = {"notebook": name, "source_sha256": hashlib.sha256(original).hexdigest(),
                      "python": str(python), "status": "running"}
            print(f"Executing {name} with {python.name}...", flush=True)
            try:
                client.execute(cwd=str(ROOT), cleanup_kc=True)
                record["status"] = "passed"
            except Exception as exc:
                record.update({"status": "failed", "error": str(exc)})
                raise
            finally:
                record["seconds"] = round(time.perf_counter() - started, 3)
                record["executed_at_utc"] = datetime.now(timezone.utc).isoformat()
                if source.read_bytes() != original:
                    record["status"] = "failed"
                    record["error"] = "Source notebook changed during execution."
                # Preserve useful failure output too; source files are never saved here.
                nbformat.write(notebook, output_dir / name)
                records.append(record)
                (output_dir / "verification.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
            if record["status"] != "passed":
                raise RuntimeError(record["error"])
            print(f"Passed {name} ({record['seconds']:.1f}s)", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=None, help="Kernel interpreter; defaults to project .venv when present.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/notebooks"))
    parser.add_argument("--timeout", type=int, default=600, help="Maximum seconds for each cell.")
    parser.add_argument("--notebook", choices=NOTEBOOKS, action="append", help="Execute selected notebook(s); default is both.")
    args = parser.parse_args()
    python = _resolve(args.python) if args.python else _default_python()
    if not python.is_file():
        parser.error(f"Python interpreter not found: {python}")
    if args.timeout < 1:
        parser.error("--timeout must be a positive number of seconds.")
    if Path(sys.executable).resolve() != python.resolve():
        # Run the verifier itself in the same environment as its temporary kernel.
        raise SystemExit(subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]]))
    try:
        execute_notebooks(python, _resolve(args.output_dir), args.timeout, args.notebook or list(NOTEBOOKS))
    except (ImportError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Notebook verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
