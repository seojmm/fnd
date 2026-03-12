from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPM_ROOT = PROJECT_ROOT / "GPM"
if str(GPM_ROOT) not in sys.path:
    sys.path.append(str(GPM_ROOT))

from data.news_data_loader import FNNDataset


def str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build cached early-detection FNN datasets with temporal cutoff ratio."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["politifact", "gossipcop"],
        choices=["politifact", "gossipcop"],
        help="Dataset names to build.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.2,
        help="Early propagation time ratio in (0, 1].",
    )
    parser.add_argument(
        "--temporal-features",
        type=str_to_bool,
        default=False,
        help="Whether to append timestamp as node feature.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root directory of the datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for dataset_name in args.datasets:
        dataset = FNNDataset(
            root=str(args.data_path),
            name=dataset_name,
            feature="bert",
            temporal_features=args.temporal_features,
            early_time_ratio=args.ratio,
        )
        print(f"[{dataset_name}] processed={dataset.processed_paths[0]}")
        print(f"[{dataset_name}] graphs={len(dataset)}, nodes={dataset._data.x.size(0)}")


if __name__ == "__main__":
    main()
