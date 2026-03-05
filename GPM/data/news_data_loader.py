import os.path as osp

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import coalesce
from torch_geometric.io import read_txt_array

import random
import numpy as np
import scipy.sparse as sp

"""
    Functions to help load the graph data
"""

def read_file(folder, name, dtype=None):
    path = osp.join(folder, '{}.txt'.format(name))
    return read_txt_array(path, sep=',', dtype=dtype)


def split(data, batch):
    """
    PyG util code to create graph batches
    """

    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)
    data.__num_nodes__ = torch.bincount(batch).tolist()

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)

    return data, slices


def read_graph_data(folder, feature, temporal_features=False, time_mapping=None):
    """
    PyG util code to create PyG data instance from raw graph data.
    Now includes temporal features integration.
    """
    node_attributes = sp.load_npz(folder + f'new_{feature}_feature.npz')
    edge_index = read_file(folder, 'A', torch.long).t()
    node_graph_id = np.load(folder + 'node_graph_id.npy')
    graph_labels = np.load(folder + 'graph_labels.npy')

    x = torch.from_numpy(node_attributes.todense()).to(torch.float)
    
    # Temporal features 반영
    if temporal_features and time_mapping is not None:
        # time_mapping이 node_index를 key로, timestamp를 value로 갖는다고 가정
        # 실제 pkl 구조에 맞게 매핑 로직은 약간의 수정이 필요할 수 있습니다.
        num_nodes = x.size(0)
        time_features = torch.zeros((num_nodes, 1), dtype=torch.float)
        
        for i in range(num_nodes):
            # pkl 파일의 key 형식에 따라 time_mapping[i] 부분 수정 필요
            if i in time_mapping:
                time_features[i, 0] = float(time_mapping[i])
            else:
                time_features[i, 0] = 0.0 # 누락된 시간 정보 처리
                
        # 기존 피처(x)에 시간 피처(time_features)를 concat
        x = torch.cat([x, time_features], dim=-1)

    node_graph_id = torch.from_numpy(node_graph_id).to(torch.long)
    y = torch.from_numpy(graph_labels).to(torch.long)
    _, y = y.unique(sorted=True, return_inverse=True)

    num_nodes = edge_index.max().item() + 1 if x is None else x.size(0)
    
    # Edge Feature로 Delta T (속도 개념)를 추가하고 싶다면 여기서 계산 가능
    edge_attr = None 
    
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data, slices = split(data, node_graph_id)

    return data, slices


class ToUndirected:
    def __init__(self):
        """
        PyG util code to transform the graph to the undirected graph
        """
        pass

    def __call__(self, data):
        edge_attr = None
        edge_index = to_undirected(data.edge_index, data.x.size(0))
        num_nodes = edge_index.max().item() + 1 if data.x is None else data.x.size(0)
        # edge_index, edge_attr = add_self_loops(edge_index, edge_attr)
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data


class DropEdge:
    def __init__(self, tddroprate, budroprate):
        """
        Drop edge operation from BiGCN (Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks)
        1) Generate TD and BU edge indices
        2) Drop out edges
        Code from https://github.com/TianBian95/BiGCN/blob/master/Process/dataset.py
        """
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __call__(self, data):
        edge_index = data.edge_index

        if self.tddroprate > 0:
            row = list(edge_index[0])
            col = list(edge_index[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edge_index

        burow = list(edge_index[1])
        bucol = list(edge_index[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow, bucol]

        data.edge_index = torch.LongTensor(new_edgeindex)
        data.BU_edge_index = torch.LongTensor(bunew_edgeindex)
        data.root = torch.FloatTensor(data.x[0])
        data.root_index = torch.LongTensor([0])

        return data


class FNNDataset(InMemoryDataset):
    r"""
        The Graph datasets built upon FakeNewsNet data

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The `name
            <https://chrsmrrs.github.io/datasets/docs/datasets/>`_ of the
            dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, root, name, feature='bert', empty=False, transform=None, pre_transform=None, pre_filter=None, temporal_features=False):
        self.name = name
        self.root = root
        self.feature = feature
        self.temporal_features = temporal_features
        super(FNNDataset, self).__init__(root, transform, pre_transform, pre_filter)
        if not empty:
            self.data, self.slices, self.train_idx, self.val_idx, self.test_idx = torch.load(self.processed_paths[0], weights_only=False)
            
    @property
    def raw_dir(self):
        name = 'raw/'
        return osp.join(self.root, self.name, name)

    @property
    def processed_dir(self):
        name = 'processed/'
        return osp.join(self.root, self.name, name)

    @property
    def num_node_attributes(self):
        if self.data.x is None:
            return 0
        return self.data.x.size(1)

    @property
    def raw_file_names(self):
        names = ['node_graph_id', 'graph_labels']
        return ['{}.npy'.format(name) for name in names]

    @property
    def processed_file_names(self):
        # 시간 피처 유무에 따라 캐싱 파일 이름을 다르게 저장하여 충돌 방지
        temp_suffix = "_time" if self.temporal_features else ""
        if self.pre_filter is None:
            return f'{self.name[:3]}_data_{self.feature}{temp_suffix}.pt'
        else:
            return f'{self.name[:3]}_data_{self.feature}_prefiler{temp_suffix}.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')

    def process(self):
        time_mapping = None
  
        if self.temporal_features:
            # pkl 파일 로드 (raw 폴더에 있다고 가정)
            time_file_path = osp.join(self.raw_dir, 'pol_id_time_mapping (1).pkl')
            if osp.exists(time_file_path):
                with open(time_file_path, 'rb') as f:
                    time_mapping = pickle.load(f)
            else:
                print(f"Warning: Temporal mapping file not found at {time_file_path}")

        self.data, self.slices = read_graph_data(self.raw_dir, self.feature, self.temporal_features, time_mapping)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        # The fixed data split for benchmarking evaluation
        # train-val-test split is 70%-10%-20%
        self.train_idx = torch.from_numpy(np.load(self.raw_dir + 'train_idx.npy')).to(torch.long)
        self.val_idx = torch.from_numpy(np.load(self.raw_dir + 'val_idx.npy')).to(torch.long)
        self.test_idx = torch.from_numpy(np.load(self.raw_dir + 'test_idx.npy')).to(torch.long)

        torch.save((self.data, self.slices, self.train_idx, self.val_idx, self.test_idx), self.processed_paths[0])

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))