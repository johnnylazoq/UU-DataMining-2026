"""Command-line entry point for the CampusGather preprocessing pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from src.pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the CampusGather data-preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Override the raw dataset directory from the configuration.",
    )
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Only process Bluetooth contacts and gatherings.",
    )
    return parser


def main() -> int:
    """Parse arguments, run the pipeline, and report its outputs."""
    args = build_parser().parse_args()
    summary = run_pipeline(
        config_path=args.config.resolve(),
        project_root=PROJECT_ROOT,
        raw_dir_override=args.raw_dir.resolve() if args.raw_dir else None,
        skip_secondary=args.skip_secondary,
    )

    print(f"Processed contacts: {summary['contacts']}")
    print(f"Extracted gatherings: {summary['gatherings']}")
    print(f"Processed data directory: {summary['processed_dir']}")
    if summary["secondary_datasets"]:
        print("Secondary datasets: exported")
    else:
        print("Secondary datasets: skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())