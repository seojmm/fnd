import os.path as osp
import pickle
import sys
from functools import cached_property
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.io import read_txt_array
from torch_geometric.utils import add_self_loops, to_undirected
from torch_sparse import coalesce

from utils.utils import idx2mask

# Allow importing scripts/cluster.py regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(SCRIPTS_DIR))
from cluster import build_clustered_graph

NEWS_DATASETS = {'politifact': 'pol', 'gossipcop': 'gos'}


class ToUndirected:
    def __call__(self, data):
        num_nodes = data.x.size(0) if data.x is not None else int(data.edge_index.max()) + 1
        edge_index = to_undirected(data.edge_index, num_nodes=num_nodes)
        data.edge_index, data.edge_attr = coalesce(edge_index, None, num_nodes, num_nodes)
        return data


class DropEdge:
    def __init__(self, tddroprate, budroprate):
        """
        Drop edge operation from BiGCN (Rumor Detection on Social Media with
        Bi-Directional Graph Convolutional Networks).
        """
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __call__(self, data):
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        td_keep = (
            torch.randperm(num_edges)[: int(num_edges * (1 - self.tddroprate))].sort().values
            if self.tddroprate > 0
            else slice(None)
        )
        bu_keep = (
            torch.randperm(num_edges)[: int(num_edges * (1 - self.budroprate))].sort().values
            if self.budroprate > 0
            else slice(None)
        )

        data.edge_index = edge_index[:, td_keep].long()
        data.BU_edge_index = edge_index.flip(0)[:, bu_keep].long()
        data.root = data.x[0].float()
        data.root_index = torch.tensor([0], dtype=torch.long)
        return data


class FNNDataset(InMemoryDataset):
    r"""The graph datasets built upon FakeNewsNet data."""

    def __init__(self, root, name, feature='bert', empty=False, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        self.feature = feature
        super().__init__(root, transform or ToUndirected(), pre_transform, pre_filter)
        if not empty:
            self._data, self.node_graph_id, self.train_idx, self.val_idx, self.test_idx = torch.load(
                self.processed_paths[0], weights_only=False
            )
            self._data, self.slices = self._split_graphs(self._data, self.node_graph_id)

    @staticmethod
    def _split_graphs(data, batch):
        node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
        node_slice = torch.cat([torch.tensor([0]), node_slice])

        row, _ = data.edge_index
        edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
        edge_slice = torch.cat([torch.tensor([0]), edge_slice])

        data.edge_index -= node_slice[batch[row]].unsqueeze(0)
        data.__num_nodes__ = torch.bincount(batch).tolist()

        slices = {'edge_index': edge_slice}
        if data.x is not None:
            slices['x'] = node_slice
        if data.edge_attr is not None:
            slices['edge_attr'] = edge_slice
        if data.y is not None:
            slices['y'] = node_slice if data.y.size(0) == batch.size(0) else torch.arange(0, batch[-1] + 2, dtype=torch.long)

        return data, slices

    def as_indexed_dataset(self):
        x_feat = self._data.x.float()
        self._data.x_feat = x_feat
        self._data.x = torch.arange(x_feat.size(0), dtype=torch.long).unsqueeze(-1)
        self._data.e_feat = self._data.edge_attr
        return self

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw/')

    @property
    def num_classes(self):
        return len(self._data.y.unique())

    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed/')

    @property
    def num_node_attributes(self):
        return 0 if self._data.x is None else self._data.x.size(1)

    @property
    def raw_file_names(self):
        return [f'{name}.npy' for name in ('node_graph_id', 'graph_labels')]

    @property
    def processed_file_names(self):
        suffix = '' if self.pre_filter is None else '_prefiler'
        return f'{self.name[:3]}_data_{self.feature}{suffix}.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):
        store = _NewsGraphStore(self.raw_dir, self.feature)
        self._data, self.node_graph_id = store.data_bundle

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self._data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self._data, self.slices = self.collate(data_list)

        self.train_idx = torch.from_numpy(np.load(Path(self.raw_dir) / 'train_idx.npy')).long()
        self.val_idx = torch.from_numpy(np.load(Path(self.raw_dir) / 'val_idx.npy')).long()
        self.test_idx = torch.from_numpy(np.load(Path(self.raw_dir) / 'test_idx.npy')).long()

        torch.save((self._data, self.node_graph_id, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name}({len(self)})'


class ClusteredNewsDataset:
    def __init__(self, graphs, x_feat, y):
        self.graphs = graphs
        self.y = y
        self._data = SimpleNamespace(x_feat=x_feat, edge_attr=None, e_feat=None)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())
        return self.graphs[int(idx)]

    def __iter__(self):
        return iter(self.graphs)


class _NewsGraphStore:
    def __init__(self, raw_dir, feature='bert'):
        self.raw_dir = Path(raw_dir)
        self.feature = feature

    @cached_property
    def data_bundle(self):
        x = torch.from_numpy(sp.load_npz(self.raw_dir / f'new_{self.feature}_feature.npz').toarray()).float()
        edge_index = read_txt_array(str(self.raw_dir / 'A.txt'), sep=',', dtype=torch.long).t()
        node_graph_id = torch.from_numpy(np.load(self.raw_dir / 'node_graph_id.npy')).long()
        y = torch.from_numpy(np.load(self.raw_dir / 'graph_labels.npy')).long()
        _, y = y.unique(sorted=True, return_inverse=True)

        edge_index, edge_attr = add_self_loops(edge_index, None)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, x.size(0), x.size(0))
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y), node_graph_id

    @property
    def data(self):
        return self.data_bundle[0]

    @property
    def node_graph_id(self):
        return self.data_bundle[1]

    @cached_property
    def base_split(self):
        num_graphs = self.data.y.size(0)
        return {
            name: idx2mask(torch.from_numpy(np.load(self.raw_dir / f'{name}_idx.npy')).long(), num_graphs)
            for name in ('train', 'val', 'test')
        }

    @cached_property
    def time_mapping(self):
        prefix = NEWS_DATASETS.get(self.raw_dir.parent.name)
        if prefix is None:
            raise ValueError(f"Unsupported news dataset: {self.raw_dir.parent.name}")
        with (self.raw_dir / f'{prefix}_id_time_mapping.pkl').open('rb') as handle:
            return pickle.load(handle)

    @cached_property
    def graph_starts(self):
        node_graph_id = self.node_graph_id.cpu().numpy()
        return np.flatnonzero(np.r_[True, node_graph_id[1:] != node_graph_id[:-1]])

    @staticmethod
    def _cache_tag(params, feature):
        method = params['clustering_method']
        tag = f"{feature}_{method}_k{params['num_clusters']}"
        if method == 'ts':
            gamma_tag = str(params.get('time_gamma', 5.0)).replace('.', 'p')
            tag = f"{tag}_g{gamma_tag}"
        return tag

    @staticmethod
    def _build_pe_transform(params):
        node_pe = params.get('node_pe', 'none')
        node_pe_dim = params.get('node_pe_dim', 0)
        if node_pe == 'rw' and node_pe_dim > 0:
            return T.AddRandomWalkPE(node_pe_dim, 'pe')
        if node_pe == 'lap' and node_pe_dim > 0:
            return T.AddLaplacianEigenvectorPE(node_pe_dim, 'pe')
        return None

    @staticmethod
    def _timestamp(value):
        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            return 0.0

    def load_splits(self, split_count):
        return [{key: value.clone() for key, value in self.base_split.items()} for _ in range(split_count)]

    def extract_graph(self, graph_idx, with_timestamps=False):
        start = int(self.graph_starts[graph_idx])
        end = int(self.graph_starts[graph_idx + 1]) if graph_idx + 1 < len(self.graph_starts) else self.node_graph_id.numel()
        edge_mask = (self.node_graph_id[self.data.edge_index[0]] == graph_idx) & (
            self.node_graph_id[self.data.edge_index[1]] == graph_idx
        )
        timestamps = None
        if with_timestamps:
            lookup = self.time_mapping.get if isinstance(self.time_mapping, dict) else self.time_mapping.__getitem__
            timestamps = torch.tensor(
                [self._timestamp(lookup(node_id)) for node_id in range(start, end)],
                dtype=torch.float32,
            )
        return self.data.x[start:end], self.data.edge_index[:, edge_mask] - start, timestamps

    def load_clustered_dataset(self, params, split_count):
        method = params['clustering_method']
        cache_path = self.raw_dir.parent / 'processed' / (
            f"{params['dataset'][:3]}_clustered_{self._cache_tag(params, self.feature)}.pt"
        )
        if cache_path.exists():
            payload = torch.load(cache_path, weights_only=False, map_location='cpu')
            dataset = ClusteredNewsDataset(payload['graphs'], payload['x_feat'], payload['y'])
            splits = [{key: value.clone() for key, value in split.items()} for split in payload['splits']]
            return dataset, splits

        labels = self.data.y.clone().long()
        transform = self._build_pe_transform(params)
        graphs, feature_blocks, global_offset = [], [], 0

        for graph_idx, label in enumerate(labels):
            x, edge_index, timestamps = self.extract_graph(graph_idx, with_timestamps=method == 'ts')
            graph, cluster_x_feat, _, _, _ = build_clustered_graph(x, edge_index, timestamps, params)
            graph.x = torch.arange(global_offset, global_offset + cluster_x_feat.size(0), dtype=torch.long).unsqueeze(-1)
            graph.y = label.view(())
            if transform is not None:
                graph = transform(graph)
            graphs.append(graph)
            feature_blocks.append(cluster_x_feat)
            global_offset += cluster_x_feat.size(0)

        x_feat = (
            torch.cat(feature_blocks, dim=0)
            if feature_blocks
            else torch.empty((0, self.data.x.size(-1)), dtype=self.data.x.dtype)
        )
        splits = self.load_splits(split_count)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'graphs': graphs, 'x_feat': x_feat, 'y': labels, 'splits': splits}, cache_path)
        return ClusteredNewsDataset(graphs, x_feat, labels), self.load_splits(split_count)


def load_fnn_dataset(params, feature='bert'):
    split_count = params['cv_fold'] if params.get('cv_fold') is not None else params['split_repeat']
    store = _NewsGraphStore(Path(params['data_path']) / params['dataset'] / 'raw', feature)
    method = params.get('clustering_method', 'none')
    if method == 'none':
        dataset = FNNDataset(root=params['data_path'], name=params['dataset'], feature=feature).as_indexed_dataset()
        return dataset, store.load_splits(split_count)
    if method in {'ts', 's'}:
        return store.load_clustered_dataset(params, split_count)
    raise ValueError(f"Unsupported clustering_method: {method}")


def load_visualization_graph(params, feature='bert'):
    store = _NewsGraphStore(Path(params['data_path']) / params['dataset'] / 'raw', feature)
    return store.extract_graph(params['graph_idx'], with_timestamps=params.get('clustering_method') == 'ts')
