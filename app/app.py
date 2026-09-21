"""Streamlit dashboard for exploring CampusGather processed outputs."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@st.cache_data(show_spinner=False)
def load_dataset(filename: str) -> pd.DataFrame:
    """Load one processed Parquet dataset."""
    path = PROCESSED_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run `python main.py` first."
        )
    return pd.read_parquet(path)


def show_missing_data(message: str) -> None:
    """Display a consistent actionable error in the dashboard."""
    st.error(message)
    st.info("Run `python main.py --skip-secondary` to generate the core datasets.")


def render_overview(contacts: pd.DataFrame, gatherings: pd.DataFrame) -> None:
    """Render high-level pipeline metrics."""
    st.subheader("Pipeline overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Contact records", f"{len(contacts):,}")
    metric_columns[1].metric("Participants", f"{contacts['user_a'].nunique():,}")
    metric_columns[2].metric("Gatherings", f"{len(gatherings):,}")
    metric_columns[3].metric(
        "Gathering size",
        f"{gatherings['participant_set'].map(len).mean():.1f}"
        if not gatherings.empty
        else "n/a",
    )


def render_gatherings(gatherings: pd.DataFrame) -> None:
    """Render gathering filters, distribution charts, and sample records."""
    st.subheader("Gathering analysis")
    if gatherings.empty:
        st.warning("No gatherings match the current data.")
        return

    display_data = gatherings.copy()
    display_data["participant_count"] = display_data["participant_set"].map(len)
    minimum_size = int(display_data["participant_count"].min())
    maximum_size = int(display_data["participant_count"].max())
    selected_range = st.slider(
        "Participant count",
        min_value=minimum_size,
        max_value=maximum_size,
        value=(minimum_size, maximum_size),
    )
    filtered = display_data[
        display_data["participant_count"].between(*selected_range)
    ]

    chart_data = (
        filtered["participant_count"]
        .value_counts()
        .sort_index()
        .rename("gatherings")
    )
    st.bar_chart(chart_data)
    st.caption(f"{len(filtered):,} gatherings match the selected size range.")

    table = filtered[
        [
            "gathering_id",
            "slot_id",
            "participant_count",
            "start_time",
            "end_time",
            "mean_rssi",
        ]
    ].head(100)
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_contacts(contacts: pd.DataFrame) -> None:
    """Render contact RSSI and activity summaries."""
    st.subheader("Contact analysis")
    chart_data = contacts["rssi"].round().value_counts().sort_index()
    st.line_chart(chart_data)
    st.caption("RSSI distribution for retained participant-to-participant contacts.")

    slots = (
        contacts.groupby("slot_id")
        .size()
        .rename("contacts")
        .sort_index()
    )
    st.line_chart(slots)


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="CampusGather", page_icon="🎓", layout="wide")
    st.title("CampusGather")
    st.caption("Exploring anonymous temporal interaction patterns")

    with st.sidebar:
        st.header("Dashboard")
        st.write(f"Processed data: `{PROCESSED_DIR.relative_to(PROJECT_ROOT)}`")
        if st.button("Reload datasets"):
            load_dataset.clear()
            st.rerun()

    try:
        contacts = load_dataset("contacts.parquet")
        gatherings = load_dataset("gatherings_v1.parquet")
    except (FileNotFoundError, OSError, ValueError) as error:
        show_missing_data(str(error))
        return

    render_overview(contacts, gatherings)
    tab_gatherings, tab_contacts = st.tabs(["Gatherings", "Contacts"])
    with tab_gatherings:
        render_gatherings(gatherings)
    with tab_contacts:
        render_contacts(contacts)

    st.divider()
    st.caption(
        "All identifiers are anonymous. Results describe aggregate patterns and "
        "should not be used for individual surveillance or decisions."
    )


if __name__ == "__main__":
    main()
