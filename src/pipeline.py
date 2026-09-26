"""Application-level orchestration for the CampusGather preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config
from .data.io import PREPARED_BLUETOOTH_FILENAME, export_secondary_datasets
from .features.gatherings import extract_gatherings
from .features.preprocessing import clean_bluetooth_data


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """Resolve a configured path relative to a known project directory."""
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def validate_input_files(
    raw_dir: Path,
    processed_dir: Path,
    skip_secondary: bool = False,
) -> None:
    """Raise a clear error when required input files are absent."""
    prepared_bluetooth = processed_dir / PREPARED_BLUETOOTH_FILENAME
    if not prepared_bluetooth.is_file():
        raise FileNotFoundError(
            f"Missing {prepared_bluetooth}.\n"
            "Run notebooks/eda_bt_symmetric.ipynb first; the pipeline reads "
            "its output instead of bt_symmetric.csv."
        )
    if skip_secondary:
        return
    required = ["fb_friends.csv", "genders.csv", "calls.csv", "sms.csv"]
    missing = [raw_dir / name for name in required if not (raw_dir / name).is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Missing input files:\n{formatted}\n"
            f"Place the dataset in {raw_dir} or pass --raw-dir."
        )


def run_pipeline(
    config_path: str | Path,
    project_root: str | Path,
    raw_dir_override: str | Path | None = None,
    skip_secondary: bool = False,
) -> dict[str, Any]:
    """Run preprocessing stages and return a concise execution summary."""
    root = Path(project_root).resolve()
    config = load_config(str(config_path))
    paths = config["paths"]
    raw_dir = (
        Path(raw_dir_override)
        if raw_dir_override is not None
        else resolve_path(paths["raw_data_dir"], root)
    ).resolve()
    processed_dir = resolve_path(paths["processed_data_dir"], root).resolve()
    config["paths"]["raw_data_dir"] = str(raw_dir)
    config["paths"]["processed_data_dir"] = str(processed_dir)

    validate_input_files(raw_dir, processed_dir, skip_secondary)
    contacts = clean_bluetooth_data(processed_dir / PREPARED_BLUETOOTH_FILENAME, config)
    gatherings = extract_gatherings(contacts, config)
    if not skip_secondary:
        export_secondary_datasets(config, raw_dir)

    return {
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "contacts": len(contacts),
        "gatherings": len(gatherings),
        "secondary_datasets": not skip_secondary,
    }
