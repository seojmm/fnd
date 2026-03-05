import torch
from torch import nn
from torch_geometric.utils import add_self_loops, coalesce
from torch_scatter import scatter_max
from math import sqrt, log
from typing import Optional, Tuple
from argparse import ArgumentParser

import torch_sparse.tensor

try:
    from .agent_utils import gumbel_softmax, spmm, scatter
except ImportError:
    from agent_utils import gumbel_softmax, spmm, scatter


def add_model_args(parent_parser: Optional[ArgumentParser] = None) -> ArgumentParser:
    if parent_parser is not None:
        parser = ArgumentParser(parents=[parent_parser], add_help=False, conflict_handler="resolve")
    else:
        parser = ArgumentParser(add_help=False, conflict_handler="resolve")

    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pattern_size", type=int, default=8)
    parser.add_argument("--gumbel_temp", type=float, default=1.0)
    parser.add_argument("--reduce", type=str, default="sum", help="Options: ['sum', 'mean', 'max', 'log', 'sqrt']")
    parser.add_argument("--self_loops", action="store_true", default=False)
    parser.add_argument("--num_pos_attention_heads", type=int, default=1)
    parser.add_argument("--post_ln", action="store_true", default=False)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--no_time_cond", action="store_true", default=False)
    parser.add_argument("--mlp_width_mult", type=int, default=1)
    parser.add_argument("--activation_function", type=str, default="leaky_relu")
    parser.add_argument("--negative_slope", type=float, default=0.01)
    parser.add_argument("--input_mlp", action="store_true", default=False)
    parser.add_argument("--num_edge_features", type=int, default=0)
    parser.add_argument("--edge_negative_slope", type=float, default=0.2)
    return parser


class TimeEmbedding(nn.Module):
    # Kept from original AgentNet implementation.
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.LeakyReLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        return self.timembedding(t)


class AgentNet(nn.Module):
    """
    Dynamic Graph Pattern Sampler.
    This module keeps AgentNet's RL traversal core and removes graph-level classification heads.
    """

    def __init__(
        self,
        num_features: int,
        hidden_units: int = 64,
        dropout: float = 0.1,
        pattern_size: int = 8,
        reduce: str = "sum",
        self_loops: bool = True,
        num_pos_attention_heads: int = 1,
        post_ln: bool = False,
        attn_dropout: float = 0.0,
        no_time_cond: bool = False,
        mlp_width_mult: int = 1,
        activation_function: str = "leaky_relu",
        negative_slope: float = 0.01,
        input_mlp: bool = False,
        gumbel_temp: float = 1.0,
        num_edge_features: int = 0,
        edge_negative_slope: float = 0.2,
    ):
        super().__init__()

        self.dim = hidden_units
        self.dropout = dropout
        self.pattern_size = pattern_size
        self.reduce = reduce
        self.self_loops = self_loops
        self.num_pos_attention_heads = num_pos_attention_heads
        self.post_ln = post_ln
        self.attn_dropout = attn_dropout
        self.time_cond = not no_time_cond
        self.activation_function = activation_function
        self.negative_slope = negative_slope
        self.input_mlp = input_mlp
        self.gumbel_temp = gumbel_temp
        self.num_edge_features = num_edge_features
        self.edge_negative_slope = edge_negative_slope

        activation = self._make_activation()

        if self.time_cond:
            self.time_emb = TimeEmbedding(self.pattern_size + 2, self.dim, self.dim * mlp_width_mult)

        if self.input_mlp:
            self.input_proj = nn.Sequential(
                nn.Linear(num_features, self.dim * 2),
                activation,
                nn.Linear(self.dim * 2, self.dim),
            )
        else:
            self.input_proj = nn.Sequential(nn.Linear(num_features, self.dim))

        if self.num_edge_features > 0:
            self.edge_input_proj = nn.Sequential(
                nn.Linear(num_edge_features, self.dim * 2),
                activation,
                nn.Linear(self.dim * 2, self.dim),
            )
            edge_dim = self.dim
        else:
            self.edge_input_proj = nn.Sequential(nn.Identity())
            edge_dim = 0

        self.agent_emb = nn.Embedding(1, self.dim)

        if self.post_ln:
            self.agent_ln = nn.LayerNorm(self.dim)
            self.node_ln = nn.LayerNorm(self.dim)
            self.conv_ln = nn.LayerNorm(self.dim)
        else:
            self.agent_ln = nn.Identity()
            self.node_ln = nn.Identity()
            self.conv_ln = nn.Identity()

        self.agent_node_ln = nn.Sequential(
            (nn.Identity() if self.post_ln else nn.LayerNorm(self.dim + edge_dim)),
            nn.Linear(self.dim + edge_dim, self.dim),
            (nn.LeakyReLU(self.edge_negative_slope) if self.edge_negative_slope > 0 else nn.ReLU()),
        )
        self.message_val = nn.Sequential(
            (nn.Identity() if self.post_ln else nn.LayerNorm(self.dim + edge_dim)),
            nn.Linear(self.dim + edge_dim, self.dim),
            (nn.LeakyReLU(self.edge_negative_slope) if self.edge_negative_slope > 0 else nn.ReLU()),
        )
        self.node_mlp = nn.Sequential(
            nn.Identity() if self.post_ln else nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2 * mlp_width_mult),
            activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2 * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout),
        )
        self.conv_mlp = nn.Sequential(
            nn.Identity() if self.post_ln else nn.LayerNorm(self.dim * 2),
            nn.Linear(self.dim * 2, self.dim * 2 * mlp_width_mult),
            activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2 * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout),
        )
        self.agent_mlp = nn.Sequential(
            nn.Identity() if self.post_ln else nn.LayerNorm(self.dim * 2 + edge_dim),
            nn.Linear(self.dim * 2 + edge_dim, self.dim * 2 * mlp_width_mult),
            activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.dim * 2 * mlp_width_mult, self.dim),
            nn.Dropout(self.dropout),
        )

        if self.time_cond:
            self.node_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim * 2))
            self.agent_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim * 2 + edge_dim))
            self.conv_mlp_time = nn.Sequential(activation, nn.Linear(self.dim * mlp_width_mult, self.dim * 2))

        # Q/K attention for neighbour action selection.
        self.query = nn.Sequential(
            (nn.Identity() if self.post_ln else nn.LayerNorm(self.dim)),
            nn.Linear(self.dim, self.dim * self.num_pos_attention_heads),
        )
        self.key = nn.Sequential(
            (nn.Identity() if self.post_ln else nn.LayerNorm(self.dim * 2 + edge_dim)),
            nn.Linear(self.dim * 2 + edge_dim, self.dim * self.num_pos_attention_heads),
            nn.Identity(),
        )
        self.attn_lin = nn.Sequential(nn.Linear(self.num_pos_attention_heads, 1))

        # Delta-time (propagation velocity) projection injected into agent-state updates.
        self.velocity_proj = nn.Sequential(
            nn.Linear(1, self.dim),
            activation,
            nn.Linear(self.dim, self.dim),
        )

        self.reset_parameters()

    def _make_activation(self):
        if self.activation_function == "gelu":
            return nn.GELU()
        if self.activation_function == "relu":
            return nn.ReLU()
        return nn.LeakyReLU(negative_slope=self.negative_slope)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.activation_function == "gelu":
                    nn.init.xavier_uniform_(module.weight)
                elif self.activation_function == "relu":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    nn.init.kaiming_uniform_(module.weight, a=self.negative_slope, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, nn.Embedding):
                module.reset_parameters()

    def _build_agent_neighbour(self, adj, current_pos: torch.Tensor):
        coo = torch_sparse.tensor.__getitem__(adj, current_pos).coo()
        agent_neighbour = torch.stack(coo[:2])
        edge_val = coo[2]
        return agent_neighbour, edge_val

    def sample_patterns(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
        hard: bool = True,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          sampled_patterns: [num_nodes, pattern_size + 1]
          action_log_probs: [num_nodes, pattern_size]
        """
        device = x.device
        num_nodes = x.size(0)
        tau = self.gumbel_temp if tau is None else tau

        node_emb = self.input_proj(x)

        if edge_feat is not None and self.num_edge_features > 0:
            edge_emb = self.edge_input_proj(edge_feat)
        else:
            edge_emb = None

        edge_index_sl = edge_index.clone()
        edge_emb_sl = None if edge_emb is None else edge_emb.clone()
        if self.self_loops:
            edge_index_sl, edge_emb_sl = add_self_loops(edge_index_sl, edge_attr=edge_emb_sl, num_nodes=num_nodes)
        if edge_emb_sl is not None:
            edge_index_sl, edge_emb_sl = coalesce(edge_index_sl, edge_emb_sl, num_nodes)
        else:
            edge_index_sl, _ = coalesce(edge_index_sl, None, num_nodes)

        adj = torch_sparse.tensor.SparseTensor(
            row=edge_index_sl[0],
            col=edge_index_sl[1],
            value=edge_emb_sl,
            sparse_sizes=(num_nodes, num_nodes),
            is_sorted=False,
        )

        agent_ids = torch.zeros(num_nodes, dtype=torch.long, device=device)
        agent_emb = self.agent_emb(agent_ids)
        current_pos = torch.arange(num_nodes, device=device)

        sampled_patterns = torch.zeros(num_nodes, self.pattern_size + 1, dtype=torch.long, device=device)
        sampled_patterns[:, 0] = current_pos
        action_log_probs = torch.zeros(num_nodes, self.pattern_size, dtype=node_emb.dtype, device=device)

        if node_time.dtype != node_emb.dtype:
            node_time = node_time.to(node_emb.dtype)

        ones_edge_value = torch.ones(edge_index_sl.size(1), dtype=node_emb.dtype, device=device)

        for step in range(1, self.pattern_size + 1):
            time_emb = None
            if self.time_cond:
                time_emb = self.time_emb(torch.tensor([step], device=device, dtype=torch.long))

            agent_neighbour, agent_neighbour_edge_emb = self._build_agent_neighbour(adj, current_pos)
            if agent_neighbour[0].numel() == 0:
                raise ValueError("No neighbour candidates found for current agent positions.")

            # Compute neighbour action scores.
            q = self.query(agent_emb).reshape(agent_emb.size(0), self.num_pos_attention_heads, -1)
            if agent_neighbour_edge_emb is not None:
                key_input = torch.cat(
                    [
                        node_emb[agent_neighbour[1]],
                        agent_neighbour_edge_emb,
                        node_emb[current_pos][agent_neighbour[0]],
                    ],
                    dim=-1,
                )
            else:
                key_input = torch.cat(
                    [
                        node_emb[agent_neighbour[1]],
                        node_emb[current_pos][agent_neighbour[0]],
                    ],
                    dim=-1,
                )
            k = self.key(key_input).reshape(agent_neighbour.size(1), self.num_pos_attention_heads, -1)
            attn_score = (q[agent_neighbour[0]] * k).sum(dim=-1).view(-1) / sqrt(q.size(-1))
            if self.num_pos_attention_heads > 1:
                attn_score = self.attn_lin(attn_score.view(agent_neighbour.size(1), self.num_pos_attention_heads)).view(-1)

            # Temporal masking: disallow jumps to nodes in the past.
            # Mask condition: time(u) < time(v_t)  --> invalid
            current_t = node_time[current_pos][agent_neighbour[0]]
            neigh_t = node_time[agent_neighbour[1]]
            temporal_valid = neigh_t >= current_t
            attn_score = attn_score.masked_fill(~temporal_valid, -1e9)

            attn_hard, attn_soft = gumbel_softmax(
                attn_score,
                agent_neighbour[0],
                num_nodes=num_nodes,
                hard=hard,
                tau=tau,
                return_soft=True,
            )
            selected_indices = scatter_max(attn_hard, agent_neighbour[0], dim=0, dim_size=num_nodes)[1]
            next_pos = agent_neighbour[1][selected_indices]
            sampled_patterns[:, step] = next_pos

            # Pattern extraction log-prob: log pi(a_t | s_t) for REINFORCE.
            selected_probs = attn_soft[selected_indices].clamp_min(1e-12)
            action_log_probs[:, step - 1] = torch.log(selected_probs)

            # Propagation velocity: delta_t between previous and current node.
            delta_t = (node_time[next_pos] - node_time[current_pos]).unsqueeze(-1)
            velocity_emb = self.velocity_proj(delta_t)

            agent_node_attention_value = attn_hard[selected_indices]

            if agent_neighbour_edge_emb is not None:
                edge_taken_emb = agent_neighbour_edge_emb[selected_indices]
                agent_cat = torch.cat(
                    [agent_emb, edge_taken_emb * agent_node_attention_value.unsqueeze(-1)],
                    dim=-1,
                )
            else:
                edge_taken_emb = None
                agent_cat = torch.cat([agent_emb], dim=-1)

            agent_ln = self.agent_node_ln(agent_cat)
            del agent_cat

            node_agent = torch.stack([next_pos, torch.arange(num_nodes, device=device)])
            active_nodes = torch.unique(next_pos)

            node_aggr = spmm(
                node_agent,
                agent_node_attention_value,
                num_nodes,
                num_nodes,
                agent_ln,
                reduce=self.reduce,
            )
            node_update = torch.cat([node_emb[active_nodes], node_aggr[active_nodes]], dim=-1)
            if self.time_cond:
                node_update = node_update + self.node_mlp_time(time_emb)
            node_emb[active_nodes] = self.node_ln(node_emb[active_nodes] + self.node_mlp(node_update))

            if edge_emb_sl is not None:
                message_input = torch.cat([node_emb[edge_index_sl[1]], edge_emb_sl], dim=-1)
                message_val = self.message_val(message_input)
                conv_aggr = scatter(message_val, edge_index_sl[0], dim=0, dim_size=num_nodes, reduce=self.reduce)
            else:
                conv_aggr = spmm(
                    edge_index_sl,
                    ones_edge_value,
                    num_nodes,
                    num_nodes,
                    node_emb,
                    reduce=self.reduce,
                )
            conv_update = torch.cat([node_emb[active_nodes], conv_aggr[active_nodes]], dim=-1)
            if self.time_cond:
                conv_update = conv_update + self.conv_mlp_time(time_emb)
            node_emb[active_nodes] = self.conv_ln(node_emb[active_nodes] + self.conv_mlp(conv_update))

            if edge_taken_emb is not None:
                agent_state_input = torch.cat(
                    [
                        agent_emb,
                        node_emb[next_pos] * agent_node_attention_value.unsqueeze(-1),
                        edge_taken_emb * agent_node_attention_value.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            else:
                agent_state_input = torch.cat(
                    [
                        agent_emb,
                        node_emb[next_pos] * agent_node_attention_value.unsqueeze(-1),
                    ],
                    dim=-1,
                )
            if self.time_cond:
                agent_state_input = agent_state_input + self.agent_mlp_time(time_emb)
            agent_emb = self.agent_ln(agent_emb + self.agent_mlp(agent_state_input) + velocity_emb)

            current_pos = next_pos

        return sampled_patterns, action_log_probs

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_time: torch.Tensor,
        edge_feat: Optional[torch.Tensor] = None,
        hard: bool = True,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sample_patterns(x, edge_index, node_time, edge_feat=edge_feat, hard=hard, tau=tau)
