import os.path as osp
import torch
import torch.nn.functional as F

from utils.eval import evaluate_cls_all
from utils.utils import get_device_from_model, check_path, mask2idx
from model.random_walk import get_patterns_for_graph


def get_pattern_dir(params):
    method = params['clustering_method']
    if method == 'none':
        return osp.join(params['pattern_path'], params['dataset'], 'original')
    cluster_dir = f"cluster_{method}_k{params['num_clusters']}"
    if method == 'ts':
        gamma_tag = str(params.get('time_gamma', 5.0)).replace('.', 'p')
        cluster_dir = f"{cluster_dir}_g{gamma_tag}"
    return osp.join(params['pattern_path'], params['dataset'], cluster_dir)


def preprocess_graph(dataset, params):
    pre_sample_pattern_num = params['pre_sample_pattern_num']
    pattern_size = params['pattern_size']
    p = params['p']
    q = params['q']

    cur_dir = osp.join(get_pattern_dir(params), f"{pre_sample_pattern_num}_{pattern_size}_{p}_{q}")
    check_path(cur_dir)

    pattern_path = osp.join(cur_dir, "ptn.pt")
    nid_path = osp.join(cur_dir, "nid.pt")
    eid_path = osp.join(cur_dir, "eid.pt")

    if osp.exists(pattern_path):
        return {
            'pattern': torch.load(pattern_path, weights_only=False),
            'nid': torch.load(nid_path, weights_only=False),
            'eid': torch.load(eid_path, weights_only=False),
        }

    patterns, nids, eids = get_patterns_for_graph(dataset, params)
    torch.save(patterns, pattern_path)
    torch.save(nids, nid_path)
    torch.save(eids, eid_path)
    return {'pattern': patterns, 'nid': nids, 'eid': eids}


def train_graph(dataset, model, optimizer, split=None, scheduler=None, params=None):
    if params['inference']:
        return {'train': 0, 'val': 0, 'test': 0}

    model.train()
    device = get_device_from_model(model)
    bs = params['batch_size']

    graphs = mask2idx(split['train'])
    num_graphs = len(graphs)

    y = dataset.y

    num_batches = (num_graphs + bs - 1) // bs
    train_perm = torch.randperm(num_graphs)
    total_loss = 0

    for i in range(num_batches):
        cur_graphs = graphs[train_perm[i * bs: (i + 1) * bs]]
        cur_y = y[cur_graphs].to(device)

        pred, _, _, commit_loss = model(dataset, cur_graphs, params, mode='train')
        loss = F.cross_entropy(pred, cur_y) + commit_loss
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        if params['grad_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params['grad_clip'])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    total_loss /= num_batches
    return {'train': total_loss, 'val': 0, 'test': 0}


def eval_graph(graph, model, split=None, params=None):
    model.eval()
    bs = params['batch_size']
    results = {'metric': 'acc'}

    with torch.no_grad():
        for key in ['train', 'val', 'test']:
            graphs = mask2idx(split[key])
            y = graph.y[graphs]
            num_graphs = len(graphs)
            num_batches = (num_graphs + bs - 1) // bs
            pred_list = []
            for i in range(num_batches):
                cur_graphs = graphs[i * bs: (i + 1) * bs]
                pred, _, _, _ = model(graph, cur_graphs, params, mode=key)
                pred_list.append(pred.detach())
            pred = torch.cat(pred_list, dim=0)
            metrics = evaluate_cls_all(pred, y)
            results[key] = metrics['acc']
            results[f'{key}_metrics'] = metrics

    return results
