import os.path as osp
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, CoraFull, Amazon, Coauthor, WikiCS, Flickr, Yelp, Reddit2, WebKB, \
    WikipediaNetwork, HeterophilousGraphDataset, Actor, LRGBDataset, GNNBenchmarkDataset, TUDataset, DeezerEurope, \
    Twitch
from data.news_data_loader import FNNDataset
from .dataset.attributed_graph_dataset import AttributedGraphDataset
from .dataset.heterophily_graph_dataset import load_pokec_mat
from .dataset.transfer_learning_citation_dataset import CitationNetworkDataset
from .dataset.zinc_dataset import ZINC

import torch_geometric.transforms as T
from torch_geometric.utils import degree

from utils.utils import idx2mask, mask2idx

mol_graphs = ['esol', 'freesolv', 'lipo', 'bace', 'bbbp', 'clintox', 'hiv', 'tox21', 'toxcast', 'muv', 'pcba', 'sider',
              'zinc', 'zinc_full']


def get_split(graph, setting):
    if setting == 'da':
        train_split = 0
        val_split = 0.2
    elif setting == 'low':
        train_split = 0.1
        val_split = 0.1
    elif setting == 'median':
        train_split = 0.5
        val_split = 0.25
    elif setting == 'high':
        train_split = 0.6
        val_split = 0.2
    elif setting == 'very_high':
        train_split = 0.8
        val_split = 0.1
    elif setting == 'pretrain':
        train_split = 1.0
        val_split = 0.0
    else:
        raise ValueError("Split setting error!")

    num_nodes = graph.num_nodes
    idx = torch.randperm(num_nodes)

    train_idx = idx[:int(num_nodes * train_split)]
    val_idx = idx[int(num_nodes * train_split):int(num_nodes * (train_split + val_split))]
    test_idx = idx[int(num_nodes * (train_split + val_split)):]

    train_mask = idx2mask(train_idx, num_nodes)
    val_mask = idx2mask(val_idx, num_nodes)
    test_mask = idx2mask(test_idx, num_nodes)

    split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    return split


def get_graph_split(dataset):
    train_split = 0.8
    val_split = 0.1

    num_graphs = len(dataset)
    idx = torch.randperm(num_graphs)

    train_idx = idx[:int(num_graphs * train_split)]
    val_idx = idx[int(num_graphs * train_split):int(num_graphs * (train_split + val_split))]
    test_idx = idx[int(num_graphs * (train_split + val_split)):]

    train_mask = idx2mask(train_idx, num_graphs)
    val_mask = idx2mask(val_idx, num_graphs)
    test_mask = idx2mask(test_idx, num_graphs)

    split = {'train': train_mask, 'val': val_mask, 'test': test_mask}

    return split


def load_graph_task(params):
    name = params['dataset']
    data_path = params['data_path']
    split_setting = params['split']

    assert split_setting == 'public'

    if params['node_pe'] == 'rw':
        transform = T.Compose([T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'lap':
        transform = T.Compose([T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'none':
        transform = None
    else:
        raise ValueError("Node positional encoding error!")

    assert split_setting == 'public'
    name_map = {'collab': 'COLLAB', 'imdb-b': 'IMDB-BINARY', 'imdb-m': 'IMDB-MULTI', 'reddit-b': 'REDDIT-BINARY',
                'reddit-m5k': 'REDDIT-MULTI-5K', 'reddit-m12k': 'REDDIT-MULTI-12K'}
    name = name_map[name]
    MAX_DEG = 400

    # Pre-transformation version to prevent OOM
    if params['node_pe'] == 'rw':
        pre_transform = T.Compose([T.Constant(1), T.AddRandomWalkPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'lap':
        pre_transform = T.Compose([T.Constant(1), T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')])
    elif params['node_pe'] == 'none':
        pre_transform = T.Constant(1)
    else:
        raise ValueError("Node positional encoding error!")

    dataset = TUDataset(root=data_path, name=name, use_node_attr=True, use_edge_attr=True, pre_transform=pre_transform, force_reload=True)

    degree_list = []
    for g in dataset:
        degrees = degree(g.edge_index[0], g.num_nodes, dtype=torch.long)
        degrees[degrees > MAX_DEG] = MAX_DEG
        degree_list.append(degrees)
        
    degrees = torch.cat(degree_list, dim=0)
    degrees[degrees > MAX_DEG] = MAX_DEG

    dataset.data.x = degrees.unsqueeze(1).long()
    num_feat = dataset.x.max().item() + 1
    dataset._data.x_feat = F.one_hot(torch.arange(num_feat), num_classes=num_feat).float()

    splits = [get_graph_split(dataset)] * params['split_repeat']

    return dataset, splits

