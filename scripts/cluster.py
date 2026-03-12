from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from scipy import sparse
from sklearn.cluster import SpectralClustering
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_undirected

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GPM_ROOT = PROJECT_ROOT / "GPM"
if str(GPM_ROOT) not in sys.path:
    sys.path.append(str(GPM_ROOT))

from utils.args import get_args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# num_cluster가 주어지지 않았을 때 그래프의 크기에 따라 적절한 클러스터 수를 추론하는 함수
def infer_num_clusters(num_nodes: int, ratio: float, min_clusters: int, max_clusters: int | None) -> int:
    if num_nodes <= 1:
        return 1
    guessed = max(min_clusters, int(round(num_nodes * ratio)))
    if max_clusters is not None:
        guessed = min(guessed, max_clusters)
    return int(np.clip(guessed, 1, num_nodes))


def _finalize_adjacency(src: np.ndarray, dst: np.ndarray, values: np.ndarray, num_nodes: int) -> sparse.csr_matrix:
    adjacency = sparse.coo_matrix((values, (src, dst)), shape=(num_nodes, num_nodes), dtype=np.float32).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency = adjacency.tolil()
    adjacency.setdiag(0.0)
    adjacency = adjacency.tocsr()
    adjacency.eliminate_zeros()
    return adjacency

# 표준 spectral clustering을 위한 unweighted adjacency matrix룰 생성하는 함수
def build_spectral_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sparse.csr_matrix:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    keep = src != dst
    src = src[keep]
    dst = dst[keep]
    values = np.ones(src.shape[0], dtype=np.float32)
    return _finalize_adjacency(src, dst, values, num_nodes)

# time-aware weighted adjacency matrix를 생성하는 함수. 시간 차이에 대한 지수적 감쇠와 선택적으로 feature cosine similarity를 적용
def build_time_aware_spectral_adjacency(
    edge_index: torch.Tensor,
    timestamps: torch.Tensor,
    node_features: torch.Tensor | None,
    num_nodes: int,
    gamma: float,
) -> sparse.csr_matrix:

    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    keep = src != dst
    src = src[keep]
    dst = dst[keep]

    time_np = timestamps.detach().cpu().numpy().astype(np.float32, copy=True)
    invalid = np.isnan(time_np) | (time_np <= 0)
    valid = time_np[~invalid]
    if valid.size > 0:
        time_np[invalid] = valid.min()
        time_min = time_np.min()
        time_max = time_np.max()
        if time_max > time_min:
            time_np = (time_np - time_min) / (time_max - time_min)
    else:
        time_np[:] = 0.0

    values = np.exp(-gamma * np.abs(time_np[src] - time_np[dst]))
    if node_features is not None:
        feature_np = node_features.detach().cpu().numpy().astype(np.float32, copy=False)
        norms = np.linalg.norm(feature_np, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        feature_np = feature_np / norms
        values = values * np.clip((feature_np[src] * feature_np[dst]).sum(axis=1), 0.0, 1.0)

    return _finalize_adjacency(src, dst, values.astype(np.float32, copy=False), num_nodes)

# 미리 계산된 affinity matrix로 spectral clustering 수행
def run_spectral_clustering(adjacency: sparse.csr_matrix, num_clusters: int, seed: int) -> np.ndarray:
    num_nodes = adjacency.shape[0]
    if num_nodes <= 1 or num_clusters <= 1:
        return np.zeros(num_nodes, dtype=np.int64)

    # Spectral embedding requests roughly k=n_clusters eigenvectors. For very small
    # graphs, k >= N produces noisy warnings and a degenerate partition anyway.
    num_clusters = min(int(num_clusters), max(num_nodes - 1, 1))
    if num_clusters <= 1:
        return np.zeros(num_nodes, dtype=np.int64)

    dense_adj = adjacency.toarray().astype(np.float32, copy=False)
    if dense_adj.sum() == 0:
        return np.zeros(num_nodes, dtype=np.int64)

    labels = SpectralClustering(
        n_clusters=int(np.clip(num_clusters, 1, num_nodes)),
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
    ).fit_predict(dense_adj)
    return labels.astype(np.int64)

# node-level edges를 cluster-level graph로 압축
def compress_graph_by_clusters(
    edge_index: torch.Tensor,
    cluster_labels: np.ndarray,
    drop_self_loops: bool,
) -> tuple[nx.Graph, np.ndarray, np.ndarray]:
    _, remapped = np.unique(cluster_labels.astype(np.int64, copy=False), return_inverse=True)
    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    cluster_src = remapped[src]
    cluster_dst = remapped[dst]

    num_clusters = int(remapped.max()) + 1
    cluster_sizes = np.bincount(remapped, minlength=num_clusters).astype(np.int64)
    weight_matrix = np.zeros((num_clusters, num_clusters), dtype=np.float32)
    np.add.at(weight_matrix, (cluster_src, cluster_dst), 1.0)
    if drop_self_loops:
        np.fill_diagonal(weight_matrix, 0.0)

    compressed = nx.Graph()
    compressed.add_nodes_from(range(num_clusters))
    rows, cols = np.nonzero(weight_matrix > 0)
    for src_cluster, dst_cluster in zip(rows.tolist(), cols.tolist()):
        compressed.add_edge(int(src_cluster), int(dst_cluster), weight=float(weight_matrix[src_cluster, dst_cluster]))

    return compressed, remapped.astype(np.int64), cluster_sizes

# 각 클러스터 내 node features를 평균하여 cluster-level feature로 집계하는 함수
def aggregate_cluster_features(node_features: torch.Tensor, cluster_labels: np.ndarray) -> torch.Tensor:
    num_clusters = int(cluster_labels.max()) + 1 if cluster_labels.size > 0 else 0
    if num_clusters == 0:
        return node_features.new_empty((0, node_features.size(-1)))

    labels = torch.from_numpy(cluster_labels).long()
    cluster_features = []
    for cluster_idx in range(num_clusters):
        members = labels == cluster_idx
        cluster_features.append(node_features[members].mean(dim=0))
    return torch.stack(cluster_features, dim=0)


def compressed_graph_to_edge_index(compressed_graph: nx.Graph, num_nodes: int) -> torch.Tensor:
    if compressed_graph.number_of_edges() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        src, dst = zip(*compressed_graph.edges())
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    return edge_index


def build_palette(num_clusters: int) -> list[tuple[float, float, float, float]]:
    if num_clusters <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(idx) for idx in range(num_clusters)]
    cmap = plt.get_cmap("gist_rainbow")
    return [cmap(idx / max(num_clusters - 1, 1)) for idx in range(num_clusters)]


def draw_cluster_compression(
    edge_index: torch.Tensor,
    remapped_labels: np.ndarray,
    compressed_graph: nx.Graph,
    cluster_sizes: np.ndarray,
    dataset_name: str,
    graph_idx: int,
    method: str,
    output_png: Path,
) -> None:
    num_nodes = remapped_labels.shape[0]
    num_clusters = int(cluster_sizes.shape[0])

    src = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)

    original = nx.Graph()
    original.add_nodes_from(range(num_nodes))
    original.add_edges_from(zip(src.tolist(), dst.tolist()))
    original.remove_edges_from(nx.selfloop_edges(original))

    palette = build_palette(num_clusters)
    node_colors = [palette[cluster_id] for cluster_id in remapped_labels.tolist()]
    pos_original = (
        nx.spring_layout(original, seed=42, k=1.6 / np.sqrt(max(num_nodes, 2)))
        if original.number_of_nodes() > 1
        else {0: (0.0, 0.0)}
    )
    pos_compressed = (
        nx.spring_layout(compressed_graph, seed=42, weight="weight")
        if compressed_graph.number_of_nodes() > 1
        else {0: (0.0, 0.0)}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=180)
    edge_alpha = 0.12 if original.number_of_edges() > 1000 else 0.25
    node_size = 20 if num_nodes > 300 else 35

    nx.draw_networkx_nodes(original, pos_original, node_size=node_size, node_color=node_colors, ax=axes[0])
    nx.draw_networkx_edges(original, pos_original, width=0.4, alpha=edge_alpha, edge_color="#404040", ax=axes[0])
    axes[0].set_title(f"Original ({num_nodes} nodes, {original.number_of_edges()} edges)\ncolored by {num_clusters} clusters", fontsize=11)
    axes[0].axis("off")

    compressed_nodes = sorted(compressed_graph.nodes())
    compressed_colors = [palette[node] for node in compressed_nodes]
    scaled_sizes = 260.0 + 1540.0 * (cluster_sizes.astype(np.float32) / max(cluster_sizes.max(), 1.0))
    compressed_edges = list(compressed_graph.edges(data=True))
    if compressed_edges:
        edge_weights = np.array([data["weight"] for _, _, data in compressed_edges], dtype=np.float32)
        widths = 0.8 + 4.2 * (edge_weights / max(edge_weights.max(), 1.0))
    else:
        widths = []

    nx.draw_networkx_nodes(compressed_graph, pos_compressed, node_size=scaled_sizes.tolist(), node_color=compressed_colors, ax=axes[1])
    nx.draw_networkx_edges(compressed_graph, pos_compressed, width=widths.tolist() if len(widths) else 0.8, alpha=0.65, edge_color="#2f2f2f", ax=axes[1])
    nx.draw_networkx_labels(
        compressed_graph,
        pos_compressed,
        labels={idx: f"C{idx}\n({cluster_sizes[idx]})" for idx in compressed_nodes},
        font_size=8,
        ax=axes[1],
    )
    axes[1].set_title(f"Compressed ({num_clusters} super-nodes)\nedge width: inter-cluster edge count", fontsize=11)
    axes[1].axis("off")

    ratio = num_nodes / max(num_clusters, 1)
    fig.suptitle(f"{dataset_name} graph_id={graph_idx} | method={method} | compression ratio={ratio:.2f}x", fontsize=12)
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def save_compressed_arrays(output_npz: Path, remapped_labels: np.ndarray, compressed_graph: nx.Graph) -> None:
    if compressed_graph.number_of_edges() == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_weight = np.empty((0,), dtype=np.float32)
    else:
        rows, cols, weights = [], [], []
        for src, dst, data in compressed_graph.edges(data=True):
            rows.append(int(src))
            cols.append(int(dst))
            weights.append(float(data["weight"]))
        edge_index = np.array([rows, cols], dtype=np.int64)
        edge_weight = np.array(weights, dtype=np.float32)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, cluster_labels=remapped_labels.astype(np.int64), compressed_edge_index=edge_index, compressed_edge_weight=edge_weight)


def build_clustered_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    timestamps: torch.Tensor | None,
    params: dict,
) -> tuple[Data, torch.Tensor, np.ndarray, nx.Graph, np.ndarray]:
    num_nodes = int(x.size(0))
    num_clusters = (
        params["num_clusters"]
        if params.get("num_clusters") is not None
        else infer_num_clusters(num_nodes, params.get("cluster_ratio", 0.2), params.get("min_clusters", 2), params.get("max_clusters"))
    )

    if params["clustering_method"] == "ts":
        if timestamps is None:
            raise ValueError("Time-aware spectral clustering requires node timestamps.")
        adjacency = build_time_aware_spectral_adjacency(
            edge_index=edge_index,
            timestamps=timestamps,
            node_features=x,
            num_nodes=num_nodes,
            gamma=params.get("time_gamma", 5.0),
        )
    elif params["clustering_method"] == "s":
        adjacency = build_spectral_adjacency(edge_index, num_nodes)
    else:
        raise ValueError("build_clustered_graph only supports --clustering_method ts or s.")

    cluster_labels = run_spectral_clustering(adjacency, num_clusters, seed=42)
    compressed_graph, remapped_labels, cluster_sizes = compress_graph_by_clusters(edge_index, cluster_labels, drop_self_loops=True)
    cluster_x_feat = aggregate_cluster_features(x, remapped_labels)
    compressed_edge_index = compressed_graph_to_edge_index(compressed_graph, cluster_x_feat.size(0))
    graph = Data(x=torch.empty((cluster_x_feat.size(0), 1), dtype=torch.long), edge_index=compressed_edge_index, edge_attr=None)
    return graph, cluster_x_feat, remapped_labels, compressed_graph, cluster_sizes


def main() -> None:
    from data.news_data_loader import load_visualization_graph

    params = get_args()
    params.setdefault("data_path", str(PROJECT_ROOT / "data"))
    params.setdefault("graph_idx", 0)
    if params["clustering_method"] == "none":
        raise ValueError("scripts/cluster.py requires --clustering_method ts or s.")

    seed_everything(42)
    x, edge_index, timestamps = load_visualization_graph(params, "bert")
    _, _, remapped_labels, compressed_graph, cluster_sizes = build_clustered_graph(x, edge_index, timestamps, params)

    out_root = Path("visualizations/cluster_compression") / params["dataset"] / f"graph_{params['graph_idx']}"
    num_clusters = int(cluster_sizes.shape[0])
    png_path = out_root / f"{params['clustering_method']}_k{num_clusters}.png"
    npz_path = out_root / f"{params['clustering_method']}_k{num_clusters}.npz"

    draw_cluster_compression(
        edge_index=edge_index,
        remapped_labels=remapped_labels,
        compressed_graph=compressed_graph,
        cluster_sizes=cluster_sizes,
        dataset_name=params["dataset"],
        graph_idx=params["graph_idx"],
        method=params["clustering_method"],
        output_png=png_path,
    )
    save_compressed_arrays(npz_path, remapped_labels, compressed_graph)


if __name__ == "__main__":
    main()
