"""Run the configured chronological evaluation in isolated output directories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_evaluation import (  # noqa: E402
    DEFAULT_CONFIG,
    _load_config,
    run_evaluation,
)


def _project_path(path: Path) -> Path:
    """Resolve relative CLI paths from the repository rather than the shell."""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main(argv: Sequence[str] | None = None) -> int:
    """Create a fresh timestamped run and print its generated report paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports") / "reproductions",
        help="Root for timestamped reports, relative to the repository by default.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=Path("artifacts") / "reproductions",
        help="Root for checkpoints and work files, relative to the repository.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the reduced one-epoch experiment configured for validation.",
    )
    arguments = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    report_dir = _project_path(arguments.output_root) / timestamp
    work_dir = _project_path(arguments.work_root) / timestamp
    if report_dir.exists() or work_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing reproduction run: {timestamp}"
        )

    config = _load_config(_project_path(arguments.config))
    result = run_evaluation(
        config,
        output_dir=report_dir,
        work_dir=work_dir,
        smoke=arguments.smoke,
    )
    print(
        json.dumps(
            {
                "seed": result["metadata"]["seed"],
                "dataset": result["metadata"]["dataset"],
                "smoke": arguments.smoke,
                "report_directory": str(report_dir),
                "work_directory": str(work_dir),
                "report_paths": result["report_paths"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
