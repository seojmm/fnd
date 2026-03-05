#!/usr/bin/env python3
"""
Visualize one FakeNewsNet graph and render per-pattern overlays.

Example:
  /home/seojm/GPM/.venv/bin/python scripts/visualize_news_patterns.py \
    --dataset politifact \
    --graph_idx 0 \
    --pattern_dir patterns/politifact/128_8_1_0.1 \
    --out_dir visualizations/pattern_overlays
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

nx = None
plt = None


DEFAULT_PROCESSED = {
    "politifact": "pol_data_bert.pt",
    "gossipcop": "gos_data_bert.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a news graph and pattern overlays.")
    parser.add_argument("--dataset", type=str, default="politifact", help="Dataset name (politifact/gossipcop).")
    parser.add_argument("--graph_idx", type=int, required=True, help="Graph index to visualize.")
    parser.add_argument("--pattern_dir", type=str, required=True, help="Directory with ptn.pt/nid.pt/eid.pt.")
    parser.add_argument("--data_root", type=str, default="data", help="Root data directory.")
    parser.add_argument("--processed_path", type=str, default="", help="Optional explicit processed .pt file path.")
    parser.add_argument("--out_dir", type=str, default="visualizations/pattern_overlays", help="Output root dir.")
    parser.add_argument("--layout", choices=["spring", "kamada", "circular"], default="spring")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=200, help="Spring layout iterations.")
    parser.add_argument("--figsize", type=float, default=8.0)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--node_size", type=float, default=16.0)
    parser.add_argument("--max_patterns", type=int, default=-1, help="Limit patterns; -1 means all.")
    parser.add_argument("--directed", action="store_true", help="Render graph as directed.")
    return parser.parse_args()


def ensure_vis_libs():
    global nx, plt
    try:
        import networkx as _nx
        import matplotlib.pyplot as _plt
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Visualization dependencies are missing. Install them with: "
            "/home/seojm/GPM/.venv/bin/pip install matplotlib networkx"
        ) from e

    nx = _nx
    plt = _plt


def infer_processed_path(dataset: str, data_root: Path) -> Path:
    if dataset not in DEFAULT_PROCESSED:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Pass --processed_path directly or use one of {list(DEFAULT_PROCESSED)}."
        )
    return data_root / dataset / "processed" / DEFAULT_PROCESSED[dataset]


def load_processed_graph(processed_path: Path):
    payload = torch.load(processed_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, tuple) or len(payload) < 2:
        raise ValueError(f"Unexpected processed format in {processed_path}")
    data, slices = payload[0], payload[1]
    if "x" not in slices or "edge_index" not in slices:
        raise ValueError("Processed data must include x/edge_index slices.")
    return data, slices


def extract_graph(data, slices, graph_idx: int):
    num_graphs = len(slices["x"]) - 1
    if graph_idx < 0 or graph_idx >= num_graphs:
        raise IndexError(f"graph_idx={graph_idx} is out of range [0, {num_graphs - 1}]")

    node_start = int(slices["x"][graph_idx])
    node_end = int(slices["x"][graph_idx + 1])
    edge_start = int(slices["edge_index"][graph_idx])
    edge_end = int(slices["edge_index"][graph_idx + 1])

    num_nodes = node_end - node_start
    edge_index = data.edge_index[:, edge_start:edge_end].clone()

    if edge_index.numel() > 0 and int(edge_index.max()) >= num_nodes:
        edge_index = edge_index - node_start

    return num_nodes, edge_index


def normalize_edge(u: int, v: int, directed: bool):
    if directed:
        return (u, v)
    return (u, v) if u <= v else (v, u)


def build_graph(num_nodes: int, edge_index: torch.Tensor, directed: bool):
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        for u, v in edge_index.t().tolist():
            graph.add_edge(int(u), int(v))
    return graph


def build_layout(graph: nx.Graph | nx.DiGraph, layout: str, seed: int, iterations: int):
    layout_graph = graph if not isinstance(graph, nx.DiGraph) else graph.to_undirected()
    if layout == "spring":
        return nx.spring_layout(layout_graph, seed=seed, iterations=iterations)
    if layout == "kamada":
        return nx.kamada_kawai_layout(layout_graph)
    return nx.circular_layout(layout_graph)


def extract_pattern_overlay(
    ptn: torch.Tensor, eid: torch.Tensor, graph_idx: int, pattern_idx: int, edge_index: torch.Tensor, num_nodes: int,
    directed: bool
):
    pattern_nodes = [int(v) for v in ptn[pattern_idx, graph_idx].tolist() if 0 <= int(v) < num_nodes]
    pattern_node_set = sorted(set(pattern_nodes))

    pattern_edge_set = set()
    num_edges = edge_index.size(1)

    for edge_id in eid[pattern_idx, graph_idx].tolist():
        edge_id = int(edge_id)
        if 0 <= edge_id < num_edges:
            u = int(edge_index[0, edge_id])
            v = int(edge_index[1, edge_id])
            pattern_edge_set.add(normalize_edge(u, v, directed))

    for u, v in zip(pattern_nodes[:-1], pattern_nodes[1:]):
        pattern_edge_set.add(normalize_edge(u, v, directed))

    return pattern_node_set, sorted(pattern_edge_set)


def draw_graph(
    graph: nx.Graph | nx.DiGraph,
    pos: dict,
    out_path: Path,
    title: str,
    node_size: float,
    dpi: int,
    figsize: float,
    directed: bool,
    highlight_nodes: list[int] | None = None,
    highlight_edges: list[tuple[int, int]] | None = None,
):
    plt.figure(figsize=(figsize, figsize))
    plt.title(title)

    nx.draw_networkx_edges(
        graph, pos, edge_color="#cfcfcf", width=0.7, alpha=0.5, arrows=directed, arrowsize=8 if directed else None
    )
    nx.draw_networkx_nodes(graph, pos, node_color="#9ea4aa", node_size=node_size, alpha=0.8)

    if highlight_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=highlight_edges,
            edge_color="#d7263d",
            width=1.8,
            alpha=0.95,
            arrows=directed,
            arrowsize=10 if directed else None,
        )

    if highlight_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=highlight_nodes,
            node_color="#ff4d6d",
            node_size=node_size * 1.8,
            alpha=0.98,
        )

    plt.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    ensure_vis_libs()

    pattern_dir = Path(args.pattern_dir)
    ptn_path = pattern_dir / "ptn.pt"
    eid_path = pattern_dir / "eid.pt"
    nid_path = pattern_dir / "nid.pt"

    if not ptn_path.exists() or not eid_path.exists() or not nid_path.exists():
        raise FileNotFoundError(f"pattern_dir must include ptn.pt/nid.pt/eid.pt: {pattern_dir}")

    ptn = torch.load(ptn_path, map_location="cpu")
    eid = torch.load(eid_path, map_location="cpu")

    if ptn.ndim != 3 or eid.ndim != 3:
        raise ValueError("Expected ptn/eid to be 3D tensors.")
    if args.graph_idx >= ptn.size(1):
        raise IndexError(f"graph_idx={args.graph_idx} is out of range for patterns (num_graphs={ptn.size(1)}).")

    processed_path = Path(args.processed_path) if args.processed_path else infer_processed_path(
        args.dataset, Path(args.data_root)
    )
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    data, slices = load_processed_graph(processed_path)
    num_nodes, edge_index = extract_graph(data, slices, args.graph_idx)

    graph = build_graph(num_nodes, edge_index, directed=args.directed)
    pos = build_layout(graph, layout=args.layout, seed=args.seed, iterations=args.iterations)

    out_root = Path(args.out_dir) / args.dataset / pattern_dir.name / f"graph_{args.graph_idx:04d}"
    pattern_out = out_root / "patterns"
    out_root.mkdir(parents=True, exist_ok=True)
    pattern_out.mkdir(parents=True, exist_ok=True)

    draw_graph(
        graph=graph,
        pos=pos,
        out_path=out_root / "original_graph.png",
        title=f"{args.dataset} graph {args.graph_idx} (original)",
        node_size=args.node_size,
        dpi=args.dpi,
        figsize=args.figsize,
        directed=args.directed,
    )

    n_patterns = ptn.size(0) if args.max_patterns < 0 else min(args.max_patterns, ptn.size(0))
    print(f"[INFO] graph_idx={args.graph_idx}, nodes={num_nodes}, edges={edge_index.size(1)}, patterns={n_patterns}")

    summary_rows = []
    for pattern_idx in range(n_patterns):
        highlight_nodes, highlight_edges = extract_pattern_overlay(
            ptn=ptn,
            eid=eid,
            graph_idx=args.graph_idx,
            pattern_idx=pattern_idx,
            edge_index=edge_index,
            num_nodes=num_nodes,
            directed=args.directed,
        )

        out_file = pattern_out / f"pattern_{pattern_idx:03d}.png"
        draw_graph(
            graph=graph,
            pos=pos,
            out_path=out_file,
            title=f"{args.dataset} graph {args.graph_idx} pattern {pattern_idx}",
            node_size=args.node_size,
            dpi=args.dpi,
            figsize=args.figsize,
            directed=args.directed,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
        )

        summary_rows.append(
            {
                "pattern_idx": pattern_idx,
                "num_pattern_nodes": len(highlight_nodes),
                "num_pattern_edges": len(highlight_edges),
                "output_png": str(out_file),
            }
        )

        if (pattern_idx + 1) % 20 == 0 or pattern_idx + 1 == n_patterns:
            print(f"[INFO] rendered {pattern_idx + 1}/{n_patterns}")

    summary_csv = out_root / "pattern_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"[DONE] original graph: {out_root / 'original_graph.png'}")
    print(f"[DONE] pattern overlays: {pattern_out}")
    print(f"[DONE] summary: {summary_csv}")


if __name__ == "__main__":
    main()
