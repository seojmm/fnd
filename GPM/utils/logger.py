import numpy as np
import torch

metric2order = {'loss': 'min', 'acc': 'max', 'f1': 'max', 'precision': 'max', 'recall': 'max', 'auc': 'max',
                'ap': 'max', 'mcc': 'max', 'hits@20': 'max', 'hits@50': 'max', 'hits@100': 'max', 'ndcg': 'max', 'map': 'max', 'mrr': 'max', 'rmse': 'min',
                'mae': 'min'}


class Logger:
    def __init__(self):
        self.data = {}
        self.best = {}

    def check_result(self, result):
        if 'metric' not in result:
            raise ValueError('Result must contain metric key')
        if result['metric'] not in metric2order:
            raise ValueError('Metric not supported')
        if result['train'] is None:
            result['train'] = 0
        if result['val'] is None:
            result['val'] = 0

        return result

    def log(self, run, epoch, loss, result):
        result = self.check_result(result)

        train_value = result['train']
        val_value = result['val']
        test_value = result['test']

        if run not in self.data:
            self.data[run] = {'train': [], 'val': [], 'test': []}

        self.data[run]['loss_train'] = loss
        self.data[run]['train'].append(train_value)
        self.data[run]['val'].append(val_value)
        self.data[run]['test'].append(test_value)
        self.data[run]['epoch'] = epoch

        if run not in self.best:
            self.best[run] = {'train': None, 'val': None, 'test': None, 'val_metrics': None, 'test_metrics': None}

        if metric2order[result['metric']] == 'max':
            if self.best[run]['val'] is None or val_value >= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['val_metrics'] = result.get('val_metrics')
                self.best[run]['test_metrics'] = result.get('test_metrics')
                self.best[run]['epoch'] = epoch
        else:
            if self.best[run]['val'] is None or val_value <= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['val_metrics'] = result.get('val_metrics')
                self.best[run]['test_metrics'] = result.get('test_metrics')
                self.best[run]['epoch'] = epoch

    def get_run_raw(self):
        return self.data

    def get_best_raw(self):
        return self.best

    def get_single_run(self, run_idx):
        return self.data[run_idx]

    def get_single_best(self, run_idx):
        return self.best[run_idx]

    def get_run(self):
        train = np.mean([np.mean(self.data[run_idx]['train']) for run_idx in self.data])
        val = np.mean([np.mean(self.data[run_idx]['val']) for run_idx in self.data])
        test = np.mean([np.mean(self.data[run_idx]['test']) for run_idx in self.data])
        return {'train': train, 'val': val, 'test': test}

    def get_best(self):
        train = [self.best[run_idx]['train'] for run_idx in self.best]
        val = [self.best[run_idx]['val'] for run_idx in self.best]
        test = [self.best[run_idx]['test'] for run_idx in self.best]

        return {'train': {'mean': np.mean(train), 'std': np.std(train)},
                'val': {'mean': np.mean(val), 'std': np.std(val)},
                'test': {'mean': np.mean(test), 'std': np.std(test)}}

    def get_best_extra_metrics(self):
        metric_names = ['acc', 'precision', 'recall', 'f1']
        output = {'val': {}, 'test': {}}

        for split in ['val', 'test']:
            key = f'{split}_metrics'
            for metric in metric_names:
                values = []
                for run_idx in self.best:
                    metrics = self.best[run_idx].get(key)
                    if metrics is not None and metric in metrics and metrics[metric] is not None:
                        values.append(metrics[metric])
                if values:
                    output[split][metric] = {'mean': np.mean(values), 'std': np.std(values)}

        return output
