import numpy as np


METRICS = ('acc', 'precision', 'recall', 'f1')


class Logger:
    def __init__(self):
        self.history = {}
        self.best = {}

    def log(self, run, epoch, loss, result):
        self.history.setdefault(run, []).append({'epoch': epoch, 'loss': loss, 'result': result})

        if run not in self.best or result['val'] >= self.best[run]['val']:
            self.best[run] = {
                'epoch': epoch,
                'train': result['train'],
                'val': result['val'],
                'test': result['test'],
                'train_metrics': result['train_metrics'],
                'val_metrics': result['val_metrics'],
                'test_metrics': result['test_metrics'],
            }

    def get_run_raw(self):
        return self.history

    def get_best_raw(self):
        return self.best

    def get_single_best(self, run):
        return self.best[run]

    def get_best(self):
        return {
            split: {
                'mean': float(np.mean([self.best[run][split] for run in self.best])),
                'std': float(np.std([self.best[run][split] for run in self.best])),
            }
            for split in ('train', 'val', 'test')
        }

    def get_best_extra_metrics(self):
        summary = {split: {} for split in ('train', 'val', 'test')}
        for split in ('train', 'val', 'test'):
            key = f'{split}_metrics'
            for name in METRICS:
                values = [self.best[run][key][name] for run in self.best]
                summary[split][name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
        return summary
