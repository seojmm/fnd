import argparse
import gc
import os.path as osp
import sys
from math import prod
from time import time

import numpy as np
import torch
import wandb
import yaml

PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data.news_data_loader import load_fnn_dataset
from model.model import Model
from task.graph import eval_graph, preprocess_graph, train_graph
from utils.args import get_args
from utils.early_stop import EarlyStopping
from utils.logger import Logger
from utils.scheduler import get_scheduler
from utils.sys import set_memory_limit
from utils.utils import check_path, get_num_params, seed_everything, to_millions

PAPER_METRICS = ('acc', 'precision', 'recall', 'f1')


def flatten_metrics(prefix, result):
    log = {}
    for split in ('train', 'val', 'test'):
        for name in PAPER_METRICS:
            log[f'{prefix}/{split}_{name}'] = result[f'{split}_metrics'][name]
    return log


def flatten_metric_summary(prefix, summary):
    log = {}
    for split in ('train', 'val', 'test'):
        for name in PAPER_METRICS:
            mean = summary[split][name]['mean']
            std = summary[split][name]['std']
            log[f'{prefix}/{split}_{name}'] = f'{mean:.2f} ± {std:.2f}'
            log[f'{prefix}/{split}_{name}_mean'] = mean
            log[f'{prefix}/{split}_{name}_std'] = std
    return log


def format_multiscale(value):
    if isinstance(value, str):
        try:
            return [int(x) for x in value.split(',') if x]
        except ValueError:
            return value
    return value


def print_grid_search_metadata(parser, trials, strategy, nb_trials=None):
    tunable_items = []
    for key, opt_arg in parser.opt_args.items():
        if opt_arg.tunable:
            tunable_items.append((key.lstrip('-').replace('-', '_'), opt_arg.opt_values))

    total_grid_space = prod(len(values) for _, values in tunable_items) if tunable_items else 0
    print("\n[Grid Search Metadata]")
    print(f"  strategy: {strategy}")
    print(f"  tunable params: {len(tunable_items)}")
    print(f"  total grid space: {total_grid_space}")
    if strategy == 'random_search':
        requested = nb_trials if nb_trials is not None else 0
        print(f"  requested random trials: {requested}")
        print(f"  unique generated trials: {len(trials)}")
        if requested != len(trials):
            print("  note: unique generated trials can be smaller than requested (bounded by grid space).")
    else:
        print(f"  generated trials: {len(trials)}")

    for name, values in tunable_items:
        print(f"  - {name}: {len(values)} values (ex: {values[:3]})")


def prepare_params(raw_params):
    params = dict(raw_params)

    if params['use_params']:
        with open(osp.join(PROJECT_ROOT, 'config', 'main.yaml'), 'r') as handle:
            params.update(yaml.safe_load(handle).get(params['dataset'], {}))

    params['data_path'] = osp.join(PROJECT_ROOT, 'data')
    params['pattern_path'] = osp.join(PROJECT_ROOT, 'patterns')
    params['save_path'] = osp.join(PROJECT_ROOT, 'model')

    if params['inference']:
        params['epochs'] = 1
        params['eval_every'] = 1
    if params['no_node_pe']:
        params['node_pe'] = 'none'
    if params['no_ap']:
        params['pe_encoder'] = 'none'

    params['multiscale'] = format_multiscale(params['multiscale'])
    return params


def train_model(params):
    if params['node_pe'] == 'none':
        params['node_pe_dim'] = 0

    params['device'] = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device('cpu')
    device = params['device']
    print("Use Device:", device)

    graph, splits = load_fnn_dataset(params)
    params['input_dim'] = graph._data.x_feat.size(1)
    params['edge_dim'] = 0 if graph._data.edge_attr is None else graph._data.e_feat.size(1)
    params['output_dim'] = 2

    preprocess_start = time()
    params['pattern_set'] = preprocess_graph(graph, params)
    print(f"Preprocessing time: {time() - preprocess_start:.2f}s")

    training_times, inference_times = [], []
    logger = Logger()

    for run_idx, split in enumerate(splits):
        seed_everything(run_idx)

        model = Model(params=params).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            betas=(params['opt_beta1'], params['opt_beta2']),
            eps=params['opt_eps'],
        )
        scheduler = get_scheduler(optimizer, params)
        stopper = EarlyStopping(patience=params['early_stop'])

        if run_idx == 0:
            print(f"The number of parameters: {to_millions(get_num_params(model))}M")

        for epoch in range(1, params['epochs'] + 1):
            train_start = time()
            loss = train_graph(graph, model, optimizer, split=split, scheduler=scheduler, params=params)
            training_times.append(time() - train_start)

            if epoch % params['eval_every'] == 0 and params['split'] != 'pretrain':
                eval_start = time()
                result = eval_graph(graph, model, split=split, params=params)
                inference_times.append(time() - eval_start)

                logger.log(run_idx, epoch, loss, result)
                if stopper(result):
                    print("Early Stopping at Epoch:", epoch)
                    break

                wandb.log({
                    'training dynamics/train_loss': loss['train'],
                    'training dynamics/val_loss': loss['val'],
                    'training dynamics/test_loss': loss['test'],
                    'training dynamics/train_value': result['train'],
                    'training dynamics/val_value': result['val'],
                    'training dynamics/test_value': result['test'],
                    **flatten_metrics('training dynamics', result),
                })
            else:
                wandb.log({
                    'training dynamics/train_loss': loss['train'],
                    'training dynamics/val_loss': loss['val'],
                    'training dynamics/test_loss': loss['test'],
                })

            if params['save_every'] and epoch % params['save_every'] == 0:
                save_path = osp.join(params['save_path'], params['dataset'])
                check_path(save_path)
                torch.save(model.state_dict(), osp.join(save_path, f'epoch_{epoch}.pt'))
                print('Model saved at epoch', epoch)

        best = logger.get_single_best(run_idx)
        wandb.log({
            'best values/train': best['train'],
            'best values/val': best['val'],
            'best values/test': best['test'],
            **flatten_metrics('best metrics', best),
        })

        del model, optimizer, scheduler
        torch.cuda.empty_cache()
        gc.collect()

    best = logger.get_best()
    wandb.log({
        'final result/train': f"{best['train']['mean']:.2f} ± {best['train']['std']:.2f}",
        'final result/val': f"{best['val']['mean']:.2f} ± {best['val']['std']:.2f}",
        'final result/test': f"{best['test']['mean']:.2f} ± {best['test']['std']:.2f}",
        'final result/train_mean': best['train']['mean'],
        'final result/val_mean': best['val']['mean'],
        'final result/test_mean': best['test']['mean'],
        'final result/train_std': best['train']['std'],
        'final result/val_std': best['val']['std'],
        'final result/test_std': best['test']['std'],
        **flatten_metric_summary('final metrics', logger.get_best_extra_metrics()),
    })
    wandb.log({'meta/run': logger.get_run_raw(), 'meta/best': logger.get_best_raw()})
    wandb.log({
        'time/training_mean': np.mean(training_times),
        'time/training_std': np.std(training_times),
        'time/training': f"{np.mean(training_times):.2f} ± {np.std(training_times):.2f}",
        'time/inference_mean': np.mean(inference_times),
        'time/inference_std': np.std(inference_times),
        'time/inference': f"{np.mean(inference_times):.2f} ± {np.std(inference_times):.2f}",
    })

    del graph, logger
    torch.cuda.empty_cache()
    gc.collect()
    return best['test']['mean']


def run_experiment(params, trial_idx=None, num_trials=None):
    wandb.init(
        project='GPM-based FND',
        config=params,
        mode='disabled' if params['debug'] else 'online',
    )
    try:
        params = dict(wandb.config)
        print(f"\nDataset: {params['dataset']}")
        print(f"CV Fold: {params['cv_fold']}, Split Repeat: {params['split_repeat']}")
        if trial_idx is not None and num_trials is not None:
            print(f"Trial {trial_idx}/{num_trials}")
        return train_model(params)
    finally:
        wandb.finish()


def run_grid_search(parser, args):
    strategy = 'random_search' if args.grid_sampler in ('random', 'random_search') else 'grid_search'
    parser.strategy = strategy
    if strategy == 'random_search' and args.nb_trials is None:
        parser.error('--grid_sampler random_search requires --nb_trials')

    trials = args.generate_trials(args.nb_trials) if strategy == 'random_search' else args.generate_trials()
    print_grid_search_metadata(parser, trials, strategy, nb_trials=args.nb_trials)

    results = []
    for trial_idx, trial in enumerate(trials, start=1):
        params = prepare_params(vars(trial))
        score = run_experiment(params, trial_idx=trial_idx, num_trials=len(trials))
        results.append({'trial': trial_idx, 'params': params.copy(), 'score': score})
        print(f"[Trial {trial_idx}/{len(trials)}] Score: {score:.4f}\n")

    best_trial = max(results, key=lambda item: item['score'])
    print(f"\n{'=' * 80}")
    print(f"BEST TRIAL: {best_trial['trial']}/{len(trials)}")
    print(f"BEST SCORE: {best_trial['score']:.4f}")
    print("BEST PARAMS:")
    for key, value in best_trial['params'].items():
        print(f"  {key}: {value}")
    print(f"{'=' * 80}\n")


def build_parser():
    parser = get_args(None, hyper=True)
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    parser.add_argument(
        '--grid_sampler',
        type=str,
        default='grid_search',
        choices=['grid_search', 'random_search', 'random'],
        help='grid search strategy',
    )
    parser.add_argument(
        '--nb_trials',
        type=int,
        default=None,
        help='number of trials for random_search',
    )
    return parser


def main():
    set_memory_limit()
    seed_everything(42)

    parser = build_parser()
    args = parser.parse_args()

    if args.grid_search:
        run_grid_search(parser, args)
        return

    run_experiment(prepare_params(vars(args)))


if __name__ == '__main__':
    main()
