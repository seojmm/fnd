import yaml
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.linkproppred import *
from ogb.nodeproppred import *

from model.random_walk import get_patterns_for_graph

from utils.eval import evaluate, evaluate_cls_all
from utils.utils import get_device_from_model, seed_everything, check_path, get_num_params, to_millions, mask2idx


def multitask_cross_entropy(y_pred, y):
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    y[y == 0] = -1
    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred.double(), (exist_y + 1) / 2)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def multitask_regression(y_pred, y, metric='rmse'):
    if metric == 'rmse':
        criterion = nn.MSELoss(reduction="none")
    elif metric == 'mae':
        criterion = nn.L1Loss(reduction="none")

    is_valid = y ** 2 > 0
    loss = 0.0

    for idx in range(y.shape[1]):
        exist_y = y[is_valid[:, idx], idx]
        exist_pred = y_pred[is_valid[:, idx], idx]
        task_loss = criterion(exist_pred, exist_y)
        loss += torch.sum(task_loss)

    return loss / torch.sum(is_valid)


def preprocess_graph(datasets, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']  # default: 128
    pattern_size = params['pattern_size']  # default: 8
    p = params['p']
    q = params['q']

    pattern_dir = osp.join(params['pattern_path'], params['dataset'])
    pattern_dict = {}

    if isinstance(datasets, dict):
        for key, subset in datasets.items():
            cur_dir = osp.join(pattern_dir, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}", key)
            check_path(cur_dir)

            pattern_path = osp.join(cur_dir, f"ptn.pt")
            nid_path = osp.join(cur_dir, f"nid.pt")
            eid_path = osp.join(cur_dir, f"eid.pt")

            if osp.exists(pattern_path) and osp.exists(nid_path) and osp.exists(eid_path):
                patterns = torch.load(pattern_path)
                nids = torch.load(nid_path)
                eids = torch.load(eid_path)
            else:
                patterns, nids, eids = get_patterns_for_graph(subset, params)
                torch.save(patterns, pattern_path)
                torch.save(nids, nid_path)
                torch.save(eids, eid_path)

            pattern_dict[key] = {'pattern': patterns, 'nid': nids, 'eid': eids}
    else:
        cur_dir = osp.join(pattern_dir, f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
        check_path(cur_dir)

        pattern_path = osp.join(cur_dir, f"ptn.pt")
        nid_path = osp.join(cur_dir, f"nid.pt")
        eid_path = osp.join(cur_dir, f"eid.pt")

        if osp.exists(pattern_path) and osp.exists(nid_path) and osp.exists(eid_path):
            patterns = torch.load(pattern_path, weights_only=True)
            nids = torch.load(nid_path, weights_only=True)
            eids = torch.load(eid_path, weights_only=True)

        else:
            patterns, nids, eids = get_patterns_for_graph(datasets, params)
            torch.save(patterns, pattern_path)
            torch.save(nids, nid_path)
            torch.save(eids, eid_path)

        pattern_dict = {'pattern': patterns, 'nid': nids, 'eid': eids}

    return pattern_dict


def train_graph(dataset, model, optimizer, agent=None, agent_optimizer=None, split=None, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    if agent is not None:
        agent.train()
        
    device = get_device_from_model(model)
    bs = params['batch_size']

    total_loss, total_val_loss, total_test_loss = 0, 0, 0

    if isinstance(split, int):
        dataset = dataset['train'] if params['split'] != 'pretrain' else dataset['full']
        num_graphs = len(dataset)
        graphs = torch.arange(num_graphs)
    else:
        dataset = dataset
        graphs = mask2idx(split['train'])
        num_graphs = len(graphs)

    y = dataset.y

    num_batches = (num_graphs + bs - 1) // bs
    train_perm = torch.randperm(num_graphs)

    for i in range(num_batches):
        cur_graphs = graphs[train_perm[i * bs: (i + 1) * bs]]
        cur_y = y[cur_graphs].to(device)

        dynamic_patterns = None
        action_log_probs = None
        
        # ---------------------------------------------------------
        # Step 1: Agent Pattern Sampling
        # ---------------------------------------------------------
        if agent is not None:
            x_feat = dataset._data.x_feat.to(device)
            edge_index = dataset._data.edge_index.to(device)
            # 시간 정보가 x_feat의 마지막 차원에 있다고 가정 (데이터 구조에 따라 인덱스 조정 필요)
            node_time = x_feat[:, -1].to(device) 
            
            dynamic_patterns, action_log_probs = agent(x_feat, edge_index, node_time)

        # ---------------------------------------------------------
        # Step 2: Task GNN Forward (Original)
        # ---------------------------------------------------------
        pred, instance_emb, pattern_emb, commit_loss = model(dataset, cur_graphs, params, dynamic_patterns=dynamic_patterns, mode='train')
        
        # 인스턴스별 Loss를 계산하기 위해 reduction='none' 사용
        if y.ndim == 1:
            loss_orig_per_instance = F.cross_entropy(pred, cur_y, label_smoothing=params['label_smoothing'], reduction='none')
        else:
            loss_orig_per_instance = multitask_cross_entropy(pred, cur_y) # (이 함수도 reduction='none' 형태로 수정 필요할 수 있음)
            
        loss_orig = loss_orig_per_instance.mean()
        loss = loss_orig + commit_loss

        # ---------------------------------------------------------
        # Step 3: Causal Feature Masking & Reward Calculation
        # ---------------------------------------------------------
        if agent is not None:
            # 현재 배치의 패턴에 포함된 노드들 추출
            batch_patterns = dynamic_patterns[cur_graphs] # [bs, pattern_size + 1]
            mask_nodes = torch.unique(batch_patterns)
            
            # 원본 피처 백업 및 마스킹 (0으로 치환)
            orig_x_feat = dataset._data.x_feat.clone()
            dataset._data.x_feat[mask_nodes] = 0.0
            
            # 마스킹된 상태로 Task GNN 재실행 (기울기 추적 불필요)
            with torch.no_grad():
                pred_cf, _, _, _ = model(dataset, cur_graphs, params, dynamic_patterns=dynamic_patterns, mode='train')
                if y.ndim == 1:
                    loss_cf_per_instance = F.cross_entropy(pred_cf, cur_y, label_smoothing=params['label_smoothing'], reduction='none')
                else:
                    loss_cf_per_instance = multitask_cross_entropy(pred_cf, cur_y)
                    
            # 원본 피처 복구
            dataset._data.x_feat = orig_x_feat
            
            # Reward 계산: 패턴 제거 시 Loss가 증가하면 좋은 패턴(+)
            reward = (loss_cf_per_instance - loss_orig_per_instance).detach() # [bs]
            
            # Agent Loss 계산: Policy Gradient (REINFORCE)
            # 현재 배치(cur_graphs)에 해당하는 에이전트의 로그 확률 총합 계산
            batch_log_probs = action_log_probs[cur_graphs].sum(dim=1) # [bs]
            agent_loss = -torch.mean(reward * batch_log_probs)
            
            loss = loss + agent_loss # Task Loss + Agent Loss 병합

        # ---------------------------------------------------------
        # Step 4: Backpropagation & Optimization
        # ---------------------------------------------------------
        total_loss += loss.item()

        optimizer.zero_grad()
        if agent is not None:
            agent_optimizer.zero_grad()
            
        loss.backward()
        
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
            if agent is not None:
                torch.nn.utils.clip_grad_norm_(agent.parameters(), params['grad_clip'])
                
        optimizer.step()
        if agent is not None:
            agent_optimizer.step()

        if scheduler is not None:
            scheduler.step()

    total_loss /= num_batches
    return {'train': total_loss, 'val': 0, 'test': 0} # eval은 eval_graph에서 처리됨


def eval_graph(graph, model, split=None, params=None):
    model.eval()
    bs = params['batch_size']

    results = {}
    results['metric'] = params['metric']
    results['train'] = 0

    with torch.no_grad():
        for key in ['val', 'test']:
            if isinstance(split, int):
                dataset = graph[key]
                num_graphs = len(dataset)
                graphs = torch.arange(num_graphs)
            else:
                dataset = graph
                graphs = mask2idx(split[key])
                num_graphs = len(graphs)

            y = dataset.y[graphs]

            num_batches = (num_graphs + bs - 1) // bs

            pred_list = []
            for i in range(num_batches):
                cur_graphs = graphs[i * bs: (i + 1) * bs]
                pred, _, _, _ = model(dataset, cur_graphs, params, mode=key)
                pred_list.append(pred.detach())
            pred = torch.cat(pred_list, dim=0)

            value = evaluate(pred, y, params=params)
            results[key] = value
            cls_all = evaluate_cls_all(pred, y)
            if cls_all is not None:
                results[f'{key}_metrics'] = cls_all
        

    return results
