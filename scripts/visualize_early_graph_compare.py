from __future__ import annotations

import argparse
import pickle
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPM_ROOT = PROJECT_ROOT / "GPM"
if str(GPM_ROOT) not in sys.path:
    sys.path.append(str(GPM_ROOT))

from data.news_data_loader import FNNDataset


def is_zero_or_empty(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if value == 0 or value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return len(value) == 0  # type: ignore[arg-type]
    except TypeError:
        return False


def parse_unix_timestamp(value: object) -> float | None:
    if is_zero_or_empty(value):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = float(stripped)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def make_mapping_getter(time_mapping: object):
    if isinstance(time_mapping, Mapping):
        return lambda idx: time_mapping.get(int(idx), None)
    if isinstance(time_mapping, Sequence):
        return lambda idx: time_mapping[idx] if 0 <= int(idx) < len(time_mapping) else None
    raise TypeError(f"Unsupported time_mapping type: {type(time_mapping).__name__}")


def load_time_mapping(raw_dir: Path) -> object:
    candidates = [
        raw_dir / "pol_id_time_mapping.pkl",
        raw_dir / "gos_id_time_mapping.pkl",
    ]
    for path in candidates:
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Could not find time mapping pkl in {raw_dir}")


def get_graph_global_range(node_graph_id: np.ndarray, graph_idx: int) -> tuple[int, int]:
    starts = np.flatnonzero(np.r_[True, node_graph_id[1:] != node_graph_id[:-1]])
    if graph_idx < 0 or graph_idx >= len(starts):
        raise IndexError(f"graph_idx={graph_idx} out of range [0, {len(starts) - 1}]")
    start = int(starts[graph_idx])
    end = int(starts[graph_idx + 1]) if graph_idx + 1 < len(starts) else int(len(node_graph_id))
    return start, end


def compute_early_mask_and_timestamps(
    raw_dir: Path,
    graph_idx: int,
    early_time_ratio: float,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, float | None]:
    node_graph_id = np.load(raw_dir / "node_graph_id.npy")
    global_start, global_end = get_graph_global_range(node_graph_id, graph_idx)
    graph_global_indices = np.arange(global_start, global_end)

    time_mapping = load_time_mapping(raw_dir)
    get_time_value = make_mapping_getter(time_mapping)

    timestamps = np.full(graph_global_indices.shape[0], np.nan, dtype=np.float64)
    for local_idx, global_idx in enumerate(graph_global_indices):
        ts = parse_unix_timestamp(get_time_value(int(global_idx)))
        if ts is not None:
            timestamps[local_idx] = ts

    keep_mask = np.zeros(graph_global_indices.shape[0], dtype=bool)
    if keep_mask.size > 0:
        keep_mask[0] = True  # keep root

    valid_mask = ~np.isnan(timestamps)
    if not np.any(valid_mask):
        return keep_mask, timestamps, None, None, None

    start_ts = float(np.min(timestamps[valid_mask]))
    end_ts = float(np.max(timestamps[valid_mask]))
    cutoff_ts = start_ts + early_time_ratio * (end_ts - start_ts)
    keep_mask = keep_mask | (valid_mask & (timestamps <= cutoff_ts))

    return keep_mask, timestamps, start_ts, end_ts, cutoff_ts


def unix_to_utc_str(ts: float | None) -> str:
    if ts is None:
        return "N/A"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def build_early_edge_index(edge_index: torch.Tensor, keep_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    edge_keep = keep_mask[src] & keep_mask[dst]

    kept_nodes = np.where(keep_mask)[0]
    old_to_new = np.full(keep_mask.shape[0], -1, dtype=np.int64)
    old_to_new[kept_nodes] = np.arange(kept_nodes.shape[0], dtype=np.int64)

    early_src = old_to_new[src[edge_keep]]
    early_dst = old_to_new[dst[edge_keep]]
    early_edge_index = np.stack([early_src, early_dst], axis=0) if early_src.size > 0 else np.empty((2, 0), dtype=np.int64)
    return kept_nodes, early_edge_index


def draw_comparison_figure(
    full_graph: nx.DiGraph,
    early_graph: nx.DiGraph,
    keep_mask: np.ndarray,
    kept_nodes: np.ndarray,
    early_edge_index: np.ndarray,
    dataset: str,
    graph_idx: int,
    ratio: float,
    start_ts: float | None,
    end_ts: float | None,
    cutoff_ts: float | None,
    output_path: Path,
    seed: int,
) -> None:
    num_nodes = full_graph.number_of_nodes()
    pos_full = (
        nx.spring_layout(full_graph.to_undirected(), seed=seed, k=1.6 / np.sqrt(max(num_nodes, 2)))
        if num_nodes > 1
        else {0: (0.0, 0.0)}
    )
    pos_early = {new_idx: pos_full[int(old_idx)] for new_idx, old_idx in enumerate(kept_nodes.tolist())}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)

    removed_color = "#e03131"
    kept_color = "#0b7285"
    root_color = "#f08c00"

    node_colors = []
    node_sizes = []
    for node in range(num_nodes):
        if node == 0:
            node_colors.append(root_color)
            node_sizes.append(100)
        elif keep_mask[node]:
            node_colors.append(kept_color)
            node_sizes.append(35)
        else:
            node_colors.append(removed_color)
            node_sizes.append(28)

    nx.draw_networkx_nodes(full_graph, pos_full, node_color=node_colors, node_size=node_sizes, ax=axes[0])
    nx.draw_networkx_edges(full_graph, pos_full, width=0.55, alpha=0.26, edge_color="#404040", arrows=False, ax=axes[0])
    axes[0].set_title(
        f"Original Graph\nnodes={num_nodes}, edges={full_graph.number_of_edges()}",
        fontsize=11,
    )
    axes[0].axis("off")

    early_node_colors = [root_color if i == 0 else kept_color for i in range(early_graph.number_of_nodes())]
    early_node_sizes = [110 if i == 0 else 36 for i in range(early_graph.number_of_nodes())]
    nx.draw_networkx_nodes(early_graph, pos_early, node_color=early_node_colors, node_size=early_node_sizes, ax=axes[1])
    nx.draw_networkx_edges(early_graph, pos_early, width=0.75, alpha=0.45, edge_color="#1f1f1f", arrows=False, ax=axes[1])
    axes[1].set_title(
        f"Early {int(round(ratio * 100))}% Graph\nnodes={early_graph.number_of_nodes()}, edges={early_graph.number_of_edges()}",
        fontsize=11,
    )
    axes[1].axis("off")

    kept_ratio = early_graph.number_of_nodes() / max(full_graph.number_of_nodes(), 1)
    subtitle = (
        f"dataset={dataset}, graph_idx={graph_idx}, keep_ratio={kept_ratio:.3f} | "
        f"start={unix_to_utc_str(start_ts)}, end={unix_to_utc_str(end_ts)}, cutoff={unix_to_utc_str(cutoff_ts)}"
    )
    fig.suptitle(subtitle, fontsize=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="politifact", choices=["politifact", "gossipcop"])
    parser.add_argument("--graph_idx", type=int, required=True)
    parser.add_argument("--early_time_ratio", type=float, default=0.2)
    parser.add_argument("--data_root", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--feature", type=str, default="bert")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=PROJECT_ROOT / "visualizations" / "early_detection_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0 < args.early_time_ratio <= 1.0):
        raise ValueError(f"early_time_ratio must be in (0, 1], got {args.early_time_ratio}")

    dataset_full = FNNDataset(
        root=str(args.data_root),
        name=args.dataset,
        feature=args.feature,
        early_time_ratio=1.0,
    )
    graph_full = dataset_full[args.graph_idx]

    raw_dir = args.data_root / args.dataset / "raw"
    keep_mask, _, start_ts, end_ts, cutoff_ts = compute_early_mask_and_timestamps(
        raw_dir=raw_dir,
        graph_idx=args.graph_idx,
        early_time_ratio=args.early_time_ratio,
    )
    if keep_mask.shape[0] != int(graph_full.num_nodes):
        raise RuntimeError(
            f"Node count mismatch for graph_idx={args.graph_idx}: "
            f"mask_nodes={keep_mask.shape[0]} vs graph_nodes={int(graph_full.num_nodes)}"
        )

    full_src = graph_full.edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    full_dst = graph_full.edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    full_graph = nx.DiGraph()
    full_graph.add_nodes_from(range(int(graph_full.num_nodes)))
    full_graph.add_edges_from(zip(full_src.tolist(), full_dst.tolist()))
    full_graph.remove_edges_from(nx.selfloop_edges(full_graph))

    kept_nodes, early_edge_index = build_early_edge_index(graph_full.edge_index, keep_mask)
    early_graph = nx.DiGraph()
    early_graph.add_nodes_from(range(int(kept_nodes.shape[0])))
    early_graph.add_edges_from(zip(early_edge_index[0].tolist(), early_edge_index[1].tolist()))
    early_graph.remove_edges_from(nx.selfloop_edges(early_graph))

    output_path = (
        args.out_dir
        / args.dataset
        / f"graph_{args.graph_idx}_orig_vs_early{int(round(args.early_time_ratio * 100))}.png"
    )
    draw_comparison_figure(
        full_graph=full_graph,
        early_graph=early_graph,
        keep_mask=keep_mask,
        kept_nodes=kept_nodes,
        early_edge_index=early_edge_index,
        dataset=args.dataset,
        graph_idx=args.graph_idx,
        ratio=args.early_time_ratio,
        start_ts=start_ts,
        end_ts=end_ts,
        cutoff_ts=cutoff_ts,
        output_path=output_path,
        seed=args.seed,
    )

    print(f"saved={output_path}")
    print(
        f"original_nodes={full_graph.number_of_nodes()}, original_edges={full_graph.number_of_edges()}, "
        f"early_nodes={early_graph.number_of_nodes()}, early_edges={early_graph.number_of_edges()}"
    )


if __name__ == "__main__":
    main()
