import copy
import os.path as osp
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from utils.eval import *
from utils.utils import get_device_from_model, check_path
from .encoder import PatternEncoder
from .vq import VectorQuantize

from torch_geometric.nn import Node2Vec


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()

        self.input_dim = params['input_dim'] + params['edge_dim'] + params['node_pe_dim']
        self.hidden_dim = params['hidden_dim']
        self.output_dim = params['output_dim']
        self.num_layers = params['num_layers']
        self.pattern_encoder = PatternEncoder(params)

        self.vq = VectorQuantize(
            dim=self.hidden_dim,
            codebook_size=params["codebook_size"],
            codebook_dim=self.hidden_dim,
            heads=params['heads'],
            separate_codebook_per_head=True,
            use_cosine_sim=True,
            kmeans_init=True,
            ema_update=True,
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=params["heads"],
            dim_feedforward=self.hidden_dim * 4,
            dropout=params["dropout"],
            norm_first=params["norm_first"]
        )
        self.encoder = nn.ModuleList([copy.deepcopy(self.encoder_layer) for _ in range(self.num_layers)])
        self.norm = nn.LayerNorm(self.hidden_dim)

        self.head = nn.Linear(self.hidden_dim, self.output_dim)

        if params['use_attn_fusion']:
            attn_dim = self.hidden_dim if not params['use_cls_token'] else 2 * self.hidden_dim
            self.attn_layer = nn.Linear(attn_dim, 1)

    def linear_probe(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True

    # Important Pattern Identifier: 추출된 여러 패턴들 간의 중요도를 식별하기 위해 Transformer 사용
    def transformer_encode(self, x):
        for layer in self.encoder:
            last_x = x
            x = layer(self.norm(x))
            x = last_x + x
            
        return x

    # Instance Embedding & Prediction: 최종적으로 그래프 인스턴스(노드, 링크, 그래프)의 임베딩을 구하고 예측 수행
    # 변환된 패턴 임베딩들의 평균을 구하거나, [CLS] 토큰을 사용하여 최종 표현을 얻음
    def get_instance_emb(self, pattern_emb, params):
        if params['use_cls_token']:
            if params['use_attn_fusion']:
                target = pattern_emb[0, :, :]  # [n, d]
                source = pattern_emb[1:, :, :]  # [h-1, n, d]
                attn = self.attn_layer(
                    torch.cat([target.unsqueeze(0).repeat(source.size(0), 1, 1), source], dim=-1)).squeeze(-1)
                attn = F.softmax(attn, dim=0)
                instance_emb = torch.sum(attn.unsqueeze(-1) * source, dim=0) + target
            else:
                instance_emb = pattern_emb[0].squeeze(0)
        else:
            if params['use_attn_fusion']:
                source = pattern_emb
                attn = self.attn_layer(source).squeeze(-1)
                attn = F.softmax(attn, dim=0)
                instance_emb = torch.sum(attn.unsqueeze(-1) * source, dim=0)
            else:
                instance_emb = pattern_emb.mean(dim=0)
                
        return instance_emb

    def forward(self, graph, items, params, **kwargs):
        mode = kwargs['mode']
        dynamic_patterns = kwargs.get('dynamic_patterns', None)
        # [수정] dynamic_nids 추가 수신
        dynamic_nids = kwargs.get('dynamic_nids', None)
        return self.encode_graph(graph, items, params, mode, dynamic_patterns=dynamic_patterns, dynamic_nids=dynamic_nids)


    def encode_graph(self, graph, graphs, params, mode, dynamic_patterns=None, dynamic_nids=None):
        device = get_device_from_model(self)
        feat = graph._data.x_feat.to(device)

        if dynamic_patterns is not None:
            # [수정] patterns와 nids를 각각 분리하여 처리
            patterns = dynamic_patterns.unsqueeze(0).to(device) # [1, num_graphs, k]
            nids = dynamic_nids.unsqueeze(0).to(device)         # [1, num_graphs, k]
        else:
            # [기존 로직 보존] Inference나 Agent 미사용 시 기존 사전 추출 패턴 사용
            num_patterns = params['num_patterns']
            pattern_set = params['pattern_set']
            if pattern_set.get('train') is not None:
                pattern_set = pattern_set[mode] if params['split'] != 'pretrain' else pattern_set['full']

            all_patterns = pattern_set['pattern']
            all_nid = pattern_set['nid']
            selected_patterns = all_patterns[:, graphs, :]
            selected_nid = all_nid[:, graphs, :]
            h, num_graphs, k = selected_nid.shape

            if mode == 'train':
                idx = torch.randint(0, h, (num_graphs, num_patterns))
                patterns = torch.stack([selected_patterns[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
                nids = torch.stack([selected_nid[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
            else:
                patterns = selected_patterns.to(device)
                nids = selected_nid.to(device)
        # ---------------------------------------------------------

        if graph[0].get('pe') is not None:
            node_pe_list = [graph[g].pe for g in graphs]
            max_nodes = max(pe.size(0) for pe in node_pe_list)
            dim = node_pe_list[0].size(1)
            node_pe = torch.zeros((len(node_pe_list), max_nodes, dim))
            for i, pe in enumerate(node_pe_list):
                node_pe[i, :pe.size(0), :] = pe
            node_pe = node_pe.to(device)
        else:
            node_pe = None

        if graph._data.edge_attr is not None:
            e_feat = graph._data.e_feat.to(device)
            all_eid = pattern_set['eid']
            selected_eid = all_eid[:, graphs, :]
            if mode == 'train':
                eids = torch.stack([selected_eid[idx[i], i, :] for i in range(num_graphs)], dim=1).to(device)
            else:
                eids = selected_eid.to(device)
        else:
            e_feat = None
            eids = None

        pattern_feat = self.pattern_encoder.encode_graph(nids, feat, patterns, eids, e_feat, node_pe, params)

        if params['use_vq']:
            pattern_feat, _, commit_loss, _ = self.vq(pattern_feat)
        else:
            commit_loss = 0

        if params['use_cls_token']:
            cls_token = torch.ones(1, pattern_feat.size(1), pattern_feat.size(2), device=device)
            pattern_feat = torch.cat([cls_token, pattern_feat], dim=0)
        pattern_emb = self.transformer_encode(pattern_feat)
        instance_emb = self.get_instance_emb(pattern_emb, params)

        pred = self.head(instance_emb)

        return pred, instance_emb, pattern_emb, commit_loss
