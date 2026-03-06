import time

import yaml
import os.path as osp
import gc
import numpy as np
import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.transforms as T

from data.news_data_loader import FNNDataset
from model.model import Model
from model.agent import AgentNet
from task.graph import preprocess_graph, train_graph, eval_graph

from utils.sys import set_memory_limit
from utils.args import get_args
from utils.early_stop import EarlyStopping
from utils.scheduler import get_scheduler
from utils.logger import Logger
from utils.utils import seed_everything, check_path, get_num_params, to_millions, idx2mask

import wandb


def load_news_graph_task(params):
    if params['node_pe'] == 'rw':
        pre_transform = T.AddRandomWalkPE(params['node_pe_dim'], 'pe')
    elif params['node_pe'] == 'lap':
        pre_transform = T.AddLaplacianEigenvectorPE(params['node_pe_dim'], 'pe')
    elif params['node_pe'] == 'none':
        pre_transform = None
    else:
        raise ValueError("Node positional encoding error!")

    dataset = FNNDataset(root=params['data_path'], name=params['dataset'], feature='bert', pre_transform=pre_transform)

    # Keep a global feature table for node id lookup used by the model/pattern sampler.
    dataset._data.x_feat = dataset._data.x.float()
    num_nodes = dataset._data.x.size(0)
    dataset._data.x = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1)

    split = {
        'train': idx2mask(dataset.train_idx, len(dataset)),
        'val': idx2mask(dataset.val_idx, len(dataset)),
        'test': idx2mask(dataset.test_idx, len(dataset)),
    }
    splits = [split] * params['split_repeat']

    return dataset, splits


def run(params):
    seed_everything(42)  # Make sure the split is the same for each run

    # if params['node_pe'] is not 'none', then the value of 'node_pe_dim' cannot be 0
    assert not (params['node_pe'] != 'none' and params['node_pe_dim'] == 0)
    if params['node_pe'] == 'none':
        params['node_pe_dim'] = 0

    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device("cpu")
    params['device'] = device
    print("Use Device:", device)

    graph, splits = load_news_graph_task(params)
        
    if isinstance(graph, Data):
        params['input_dim'] = graph.x.size(1)
        params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    elif isinstance(graph, InMemoryDataset):
        params['input_dim'] = graph._data.x_feat.size(1)
        params['edge_dim'] = graph._data.e_feat.size(1) if graph._data.edge_attr is not None else 0
    elif isinstance(graph, dict):
        params['input_dim'] = graph['train']._data.x_feat.size(1)
        params['edge_dim'] = graph['train']._data.e_feat.size(1) if graph['train']._data.edge_attr is not None else 0
        if params['dataset'] in mol_graphs:
            params['input_dim'] = 16
            params['edge_dim'] = 16

    if params.get('num_tasks') is not None:
        params['output_dim'] = params['num_tasks']
    else:
        params['output_dim'] = graph.y.max().item() + 1
        
    preprocess = preprocess_graph
    train = train_graph
    eval = eval_graph

    # ---------------------------------------------------------
    # Agent를 사용하지 않을 때만 정적 패턴 전처리 수행
    # ---------------------------------------------------------
    pattern_set = None
    if not params.get('use_agent', False):
        start_time = time.time()
        pattern_set = preprocess(graph, params)
        end_time = time.time()
        print(f"Preprocessing time: {end_time - start_time:.2f}s")
        params['pattern_set'] = pattern_set
    else:
        print("Using AgentNet for Dynamic Pattern Sampling. Skipping static preprocess.")
        params['pattern_set'] = {} # 더미 딕셔너리 할당
    # ---------------------------------------------------------

    training_time = []
    inference_time = []

    logger = Logger()
    if splits is None:
        splits = range(params['split_repeat'])

    for idx, split in enumerate(splits):
        seed_everything(idx)

        model = Model(params=params).to(device)
        
        # ---------------------------------------------------------
        # AgentNet 초기화
        # ---------------------------------------------------------
        agent = None
        agent_optimizer = None
        if params.get('use_agent', False):
            agent = AgentNet(
                num_features=params['input_dim'],
                hidden_units=params['hidden_dim'],
                pattern_size=params['pattern_size'],
                gumbel_temp=params.get('gumbel_temp', 1.0)
            ).to(device)
            agent_optimizer = torch.optim.AdamW(
                agent.parameters(), lr=params.get('agent_lr', 0.001), 
                weight_decay=params.get('agent_weight_decay', 1e-5)
            )
        # ---------------------------------------------------------
        
        if params['pretrain_data'] != 'none':
            pretrain_path = osp.join(params['save_path'], params['pretrain_data'])
            model.load_state_dict(torch.load(osp.join(pretrain_path, f"epoch_{params['pretrain_epoch']}.pt")))
        if params['linear_probe']:
            model.linear_probe()

        num_params = to_millions(get_num_params(model))
        stopper = EarlyStopping(patience=params["early_stop"])

        if idx == 0:
            print(f'The number of parameters: {num_params}M')

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
            betas=(params['opt_beta1'], params['opt_beta2']), eps=params['opt_eps']
        )
        scheduler = get_scheduler(optimizer, params)

        for epoch in range(1, params['epochs'] + 1):
            start_time = time.time()
            
            # agent와 agent_optimizer 인자 추가 전달
            loss = train(graph, model, optimizer, agent=agent, agent_optimizer=agent_optimizer, split=split, scheduler=scheduler, params=params)
            
            end_time = time.time()
            training_time.append(end_time - start_time)

            if epoch % params['eval_every'] == 0 and params['split'] != 'pretrain':
                start_time = time.time()
                result = eval(graph, model, agent=agent, split=split, params=params)
                end_time = time.time()
                inference_time.append(end_time - start_time)

                is_stop = stopper(result)
                logger.log(idx, epoch, loss, result)
                if is_stop:
                    print("Early Stopping at Epoch:", epoch)
                    break

                wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                    "training dynamics/train_value": result['train'],
                    "training dynamics/val_value": result['val'],
                    "training dynamics/test_value": result['test'],
                })
            else:
                wandb.log({
                    "training dynamics/train_loss": loss['train'],
                    "training dynamics/val_loss": loss['val'],
                    "training dynamics/test_loss": loss['test'],
                })

            if params['save_every'] != 0 and epoch % params['save_every'] == 0:
                save_path = osp.join(params['save_path'], params['dataset'])
                check_path(save_path)
                torch.save(model.state_dict(), osp.join(save_path, f"epoch_{epoch}.pt"))
                print('Model saved at epoch', epoch)

        single_best = logger.get_single_best(idx)
        wandb.log({
            "best values/train": single_best["train"],
            "best values/val": single_best["val"],
            "best values/test": single_best["test"],
        })

        # After training
        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    best = logger.get_best()
    final_log = {
        "final result/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
        "final result/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
        "final result/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
        "final result/train_mean": best['train']['mean'],
        "final result/val_mean": best['val']['mean'],
        "final result/test_mean": best['test']['mean'],
        "final result/train_std": best['train']['std'],
        "final result/val_std": best['val']['std'],
        "final result/test_std": best['test']['std'],
    }
    extra_best = logger.get_best_extra_metrics()
    for metric in ['acc', 'precision', 'recall', 'f1']:
        stat = extra_best.get('test', {}).get(metric)
        if stat is None:
            continue
        final_log[f"final result/test_{metric}"] = "{:.2f} ± {:.2f}".format(stat['mean'], stat['std'])
        final_log[f"final result/test_{metric}_mean"] = stat['mean']
        final_log[f"final result/test_{metric}_std"] = stat['std']
    wandb.log(final_log)
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})
    wandb.log({
        "time/training_mean": np.mean(training_time),
        "time/training_std": np.std(training_time),
        "time/training": "{:.2f} ± {:.2f}".format(np.mean(training_time), np.std(training_time)),
        "time/inference_mean": np.mean(inference_time),
        "time/inference_std": np.std(inference_time),
        "time/inference": "{:.2f} ± {:.2f}".format(np.mean(inference_time), np.std(inference_time))
    })

    wandb.finish()

    # Clear everything
    del graph, logger
    torch.cuda.empty_cache()
    gc.collect()


def main():
    set_memory_limit()  # 90% by default
    params = get_args()

    params['data_path'] = osp.join(osp.dirname(__file__), '..', 'data')
    params['pattern_path'] = osp.join(osp.dirname(__file__), '..', 'patterns')
    params['save_path'] = osp.join(osp.dirname(__file__), '..', 'model')

    data_config = osp.join(osp.dirname(__file__), '..', 'config', 'data.yaml')
    with open(data_config, 'r') as f:
        data_config = yaml.safe_load(f)
        
    params['task'] = data_config[params['dataset']]['task']
    params['metric'] = data_config[params['dataset']]['metric']
    params['num_tasks'] = data_config[params['dataset']].get('num_tasks', None)

    if params["use_params"]:
        with open(osp.join(osp.dirname(__file__), '..', 'config', 'main.yaml'), 'r') as f:
            default_params = yaml.safe_load(f)
            params.update(default_params[params['task']][params['dataset']])

    if params['inference']:
        params['epochs'] = 1
        params['eval_every'] = 1
    if params['no_node_pe']:
        params['node_pe'] = 'none'
    if params['no_ap']:
        params['pe_encoder'] = 'none'

    wandb.init(
        project="GPM",
        config=params,
        mode="disabled" if params["debug"] else "online"
    )
    params = dict(wandb.config)
    print(params)

    run(params)


if __name__ == "__main__":
    main()
