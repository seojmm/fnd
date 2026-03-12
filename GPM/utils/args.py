from argparse import ArgumentParser
from test_tube import HyperOptArgumentParser
from typing import Optional


def get_args(parent_parser: Optional[ArgumentParser] = None, hyper: bool = False) -> ArgumentParser:
    if parent_parser is not None:
        parser = HyperOptArgumentParser(
            parents=[parent_parser], 
            add_help=False, 
            conflict_handler='resolve'
        ) if hyper else ArgumentParser(
            parents=[parent_parser], 
            add_help=False, 
            conflict_handler='resolve'
        )
    else:
        parser = HyperOptArgumentParser(
            add_help=False, 
            conflict_handler='resolve'
        ) if hyper else ArgumentParser(
            add_help=False, 
            conflict_handler='resolve'
        )
    
    parser.add_argument('--grid_search', action='store_true')
    parser.add_argument('--cv_fold', type=int, default=5)  # for cross-validation

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--use_params", action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    
    # Clustering
    parser.add_argument(
        '--clustering_method',
        type=str,
        default='none',
        choices=['none', 'ts', 's'],
        help='none: original news graph, ts: time-aware spectral clustering, s: standard spectral clustering',
    )
    parser.add_argument("--num_clusters", type=int, default=32)
    parser.add_argument("--cluster_ratio", type=float, default=0.2)
    parser.add_argument("--min_clusters", type=int, default=1)
    parser.add_argument("--max_clusters", type=int, default=None)
    parser.add_argument("--time_gamma", type=float, default=5.0, help='Used only when --clustering_method ts')

    # Dataset
    parser.add_argument('--dataset', '--data', type=str, default="politifact")
    parser.add_argument('--early_time_ratio', type=float, default=1.0, help='Use only early propagation window by ratio (e.g., 0.2 for 20%%).')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--split', type=str, default='public')
    parser.add_argument('--split_repeat', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=0)

    # If you want to use the pretrain model, you need to set the pretrain_data and pretrain_epoch
    parser.add_argument('--pretrain_data', type=str, default='none')
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--linear_probe', action='store_true', help='Only train the linear classifier')
    parser.add_argument('--reset_head', action='store_true', help='Reset the head of the model')
    parser.add_argument('--inference', action='store_true', help='Only inference the model')

    # Patterns and Augmentation
    parser.add_argument('--pre_sample_pattern_num', type=int, default=256)
    parser.add_argument('--pre_sample_batch_size', type=int, default=4096)
    parser.add_argument('--num_patterns', type=int, default=16)
    parser.add_argument('--pattern_size', type=int, default=4)
    if hyper:
        parser.add_argument('--multiscale', type=str, default='1,2,3,4')
    else:
        parser.add_argument('--multiscale', type=int, nargs='+', default=[1, 2, 3, 4])
    parser.add_argument('--p', type=float, default=4)
    parser.add_argument('--q', type=float, default=1)

    # Encoder
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--num_layers', '--layers', type=int, default=2)
    parser.add_argument('--norm_first', action='store_true')

    # PE
    parser.add_argument('--node_pe', type=str, default='none', choices=['rw', 'lap', 'none'])
    parser.add_argument('--node_pe_dim', type=int, default=8)
    parser.add_argument('--pe_encoder', type=str, default='mean', choices=['mean', 'gru', 'none'])
    parser.add_argument('--pe_weight', type=float, default=0.1)

    # Pattern Encoder
    parser.add_argument('--pattern_encoder', type=str, default='transformer', choices=['mean', 'gru', 'transformer'])
    parser.add_argument('--pattern_encoder_layers', type=int, default=2)
    parser.add_argument('--pattern_encoder_heads', type=int, default=2)

    # VQ and Others
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--codebook_size', type=int, default=8196)
    parser.add_argument('--use_cls_token', action='store_true')
    parser.add_argument('--use_attn_fusion', action='store_true')

    # Optimization
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--weight_decay', '--decay', type=float, default=5e-4)
    parser.add_argument('--early_stop', type=int, default=40)
    parser.add_argument('--grad_clip', type=float, default=2.0)

    parser.add_argument('--scheduler', type=str, default='cosine', choices=['warmup', 'cosine', 'none'])
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--eta_min', type=float, default=1e-6)

    parser.add_argument('--opt_beta1', type=float, default=0.9)
    parser.add_argument('--opt_beta2', type=float, default=0.999)
    parser.add_argument('--opt_eps', type=float, default=1e-8)

    # Ablation
    # If set "use_params", the model will use the default parameters, which cannot be changed by the arguments
    # If you want to change the parameters, we can set the ablation parameters here
    parser.add_argument('--no_node_pe', action='store_true')
    parser.add_argument('--no_ap', action='store_true')

    return parser
