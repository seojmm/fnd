from __future__ import annotations

import argparse
import pickle
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

DEFAULT_PKL_PATHS = [
    Path("/home/seojm/GPM/data/politifact/raw/pol_id_time_mapping.pkl"),
    Path("/home/seojm/GPM/data/gossipcop/raw/gos_id_time_mapping.pkl"),
]


@dataclass
class DatasetTimeStats:
    dataset_name: str
    total_records: int
    total_graphs: int
    graphs_with_valid_time: int
    start_timestamps: list[int]
    end_timestamps: list[int]


def is_root_value(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if value == 0:
        return True
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return len(value) == 0  # type: ignore[arg-type]
    except TypeError:
        return False


def iter_record_values(obj: object) -> tuple[Iterable[object], int]:
    if isinstance(obj, Mapping):
        return obj.values(), len(obj)
    if isinstance(obj, (list, tuple)):
        return obj, len(obj)
    raise TypeError(f"Unsupported pickle object type: {type(obj).__name__}")


def parse_unix_timestamp(value: object) -> int | None:
    if is_root_value(value):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        if parsed <= 0:
            return None
        return int(parsed)
    return None


def compute_time_stats(pkl_path: Path) -> DatasetTimeStats:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    values, total_records = iter_record_values(data)

    total_graphs = 0
    graphs_with_valid_time = 0
    start_timestamps: list[int] = []
    end_timestamps: list[int] = []

    current_graph_timestamps: list[int] = []
    seen_graph = False

    for value in values:
        if is_root_value(value):
            if seen_graph:
                total_graphs += 1
                if current_graph_timestamps:
                    graphs_with_valid_time += 1
                    start_timestamps.append(min(current_graph_timestamps))
                    end_timestamps.append(max(current_graph_timestamps))
            seen_graph = True
            current_graph_timestamps = []
            continue

        if not seen_graph:
            seen_graph = True
        timestamp = parse_unix_timestamp(value)
        if timestamp is not None:
            current_graph_timestamps.append(timestamp)

    if seen_graph:
        total_graphs += 1
        if current_graph_timestamps:
            graphs_with_valid_time += 1
            start_timestamps.append(min(current_graph_timestamps))
            end_timestamps.append(max(current_graph_timestamps))

    dataset_name = pkl_path.parent.parent.name
    return DatasetTimeStats(
        dataset_name=dataset_name,
        total_records=total_records,
        total_graphs=total_graphs,
        graphs_with_valid_time=graphs_with_valid_time,
        start_timestamps=start_timestamps,
        end_timestamps=end_timestamps,
    )


def to_utc_datetimes(timestamps: list[int]) -> list[datetime]:
    return [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in timestamps]


def plot_time_histogram(
    ax: plt.Axes,
    datetimes: list[datetime],
    title: str,
    bins: int,
    color: str,
) -> None:
    ax.set_title(title)
    if not datetimes:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Graph Count")
        return

    date_values = mdates.date2num(datetimes)
    ax.hist(date_values, bins=bins, color=color, alpha=0.85, edgecolor="black", linewidth=0.25)
    ax.set_ylabel("Graph Count")
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="y", alpha=0.2)


def plot_duration_histogram(ax: plt.Axes, durations_hours: np.ndarray, bins: int) -> None:
    ax.set_title("3) Start-End Range Distribution (Duration)")
    ax.set_xlabel("Duration (hours)")
    ax.set_ylabel("Graph Count")
    if durations_hours.size == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.hist(durations_hours, bins=bins, color="#e67700", alpha=0.9, edgecolor="black", linewidth=0.25)
    ax.grid(axis="y", alpha=0.2)
    summary = (
        f"min={durations_hours.min():.2f}h, "
        f"p50={np.median(durations_hours):.2f}h, "
        f"p95={np.percentile(durations_hours, 95):.2f}h, "
        f"max={durations_hours.max():.2f}h"
    )
    ax.text(0.99, 0.95, summary, transform=ax.transAxes, ha="right", va="top", fontsize=9)


def plot_distribution(stats: DatasetTimeStats, output_dir: Path, bins: int) -> tuple[Path, np.ndarray]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stats.dataset_name}_start_end_range_distribution.png"

    start_dates = to_utc_datetimes(stats.start_timestamps)
    end_dates = to_utc_datetimes(stats.end_timestamps)
    durations_hours = np.array(
        [(end_ts - start_ts) / 3600.0 for start_ts, end_ts in zip(stats.start_timestamps, stats.end_timestamps)],
        dtype=float,
    )

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    fig.suptitle(
        f"{stats.dataset_name} propagation graph time distributions (UTC)\n"
        f"start=min(valid timestamp), end=max(valid timestamp)",
        fontsize=13,
    )

    plot_time_histogram(
        axes[0],
        start_dates,
        "1) Start Time Distribution",
        bins,
        "#0b7285",
    )
    plot_time_histogram(
        axes[1],
        end_dates,
        "2) End Time Distribution",
        bins,
        "#2b8a3e",
    )
    plot_duration_histogram(axes[2], durations_hours, bins)

    axes[2].set_xlabel("Duration (hours)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path, durations_hours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize start/end time and start-end range distributions of propagation graphs."
    )
    parser.add_argument(
        "pkl_paths",
        nargs="*",
        type=Path,
        default=DEFAULT_PKL_PATHS,
        help="Pickle file paths. If omitted, the two default mapping files are used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/seojm/GPM/visualizations/time_distributions"),
        help="Directory to save the generated plots.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=50,
        help="Histogram bin count for each distribution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for pkl_path in args.pkl_paths:
        stats = compute_time_stats(pkl_path)
        output_path, durations_hours = plot_distribution(stats, args.output_dir, args.bins)
        skipped_graphs = stats.total_graphs - stats.graphs_with_valid_time
        print(f"[{stats.dataset_name}] records={stats.total_records}, graphs={stats.total_graphs}")
        print(
            f"[{stats.dataset_name}] graphs_with_valid_time={stats.graphs_with_valid_time}, "
            f"skipped_no_time={skipped_graphs}"
        )
        if durations_hours.size > 0:
            print(
                f"[{stats.dataset_name}] duration_hours "
                f"min={durations_hours.min():.2f}, p50={np.median(durations_hours):.2f}, "
                f"p95={np.percentile(durations_hours, 95):.2f}, max={durations_hours.max():.2f}"
            )
        print(f"[{stats.dataset_name}] saved_plot={output_path}")


if __name__ == "__main__":
    main()
