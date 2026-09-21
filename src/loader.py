"""Backward-compatible imports for the original loader module."""

from .gatherings import extract_gatherings
from .io import (
    export_secondary_datasets,
    load_facebook_data,
    load_gender_data,
    load_telecom_data,
)
from .preprocessing import clean_bluetooth_data

__all__ = [
    "clean_bluetooth_data",
    "extract_gatherings",
    "export_secondary_datasets",
    "load_facebook_data",
    "load_gender_data",
    "load_telecom_data",
]
