from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .preprocessing import PropagationGraph

HierarchicalGraph = PropagationGraph


@dataclass
class AgentRollout:
    visited_nodes: torch.Tensor
    traversed_edges: torch.Tensor
    action_log_probs: torch.Tensor
    entropies: torch.Tensor
    elapsed_times: torch.Tensor
    causal_evidence: torch.Tensor
    prefix_lengths: torch.Tensor
    halted: bool
    halt_step: int
    level: str


@dataclass
class DualAgentOutput:
    cluster_rollouts: List[AgentRollout]
    node_rollouts: List[AgentRollout]
    guided_masks: List[torch.Tensor]
    rl_loss: torch.Tensor
    reward_aux_loss: torch.Tensor
    reward_trace: torch.Tensor
    cluster_reward_trace: torch.Tensor


@dataclass
class AgentStepOutput:
    action_hard: torch.Tensor
    next_node: Optional[int]
    next_edge: Optional[int]
    action_log_prob: torch.Tensor
    entropy: torch.Tensor
    halted: bool


class CausalPatternScorer(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, lambda_spurious: float) -> None:
        super().__init__()
        self.lambda_spurious = lambda_spurious
        self.causal_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spurious_encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.causal_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.joint_decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def _label_entropy(self, labels: torch.Tensor) -> torch.Tensor:
        counts = torch.bincount(labels, minlength=2).float()
        probs = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * (probs + 1e-8).log()).sum()
        return entropy

    def _estimate_information(
        self,
        decoder: nn.Module,
        representations: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if representations.size(0) == 0:
            zero = representations.new_zeros(0)
            return zero, representations.new_tensor(0.0)

        logits = decoder(representations)
        ce = F.cross_entropy(logits, labels, reduction="none")
        entropy = self._label_entropy(labels).to(representations.device)
        mutual_information = entropy - ce
        aux_loss = ce.mean()
        return mutual_information, aux_loss

    def forward(
        self,
        causal_summary: torch.Tensor,
        spurious_summary: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # causal_summary: [batch_size, max_steps, feature_dim]
        # spurious_summary: [batch_size, max_steps, feature_dim]
        # labels: [batch_size]
        # valid_mask: [batch_size, max_steps]
        reward = causal_summary.new_zeros(valid_mask.size(0), valid_mask.size(1))
        aux_losses: List[torch.Tensor] = []

        for step in range(valid_mask.size(1)):
            active = valid_mask[:, step]
            if active.sum() < 2:
                continue

            causal_rep = self.causal_encoder(causal_summary[active, step])
            spurious_rep = self.spurious_encoder(spurious_summary[active, step])
            step_labels = labels[active]

            mi_c, aux_c = self._estimate_information(self.causal_decoder, causal_rep, step_labels)
            joint_rep = torch.cat([causal_rep, spurious_rep], dim=-1)
            mi_joint, aux_joint = self._estimate_information(self.joint_decoder, joint_rep, step_labels)
            conditional_mi = (mi_joint - mi_c).clamp_min(0.0)
            reward[active, step] = mi_c - self.lambda_spurious * conditional_mi
            aux_losses.extend([aux_c, aux_joint])

        if aux_losses:
            aux_loss = torch.stack(aux_losses).mean()
        else:
            aux_loss = causal_summary.new_tensor(0.0)
        return reward, aux_loss


class TraversalAgent(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        max_steps: int,
        gumbel_tau: float = 1.0,
        halt_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.gumbel_tau = gumbel_tau
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_update = nn.GRUCell(hidden_dim + 2, hidden_dim)
        self.query = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        self.candidate_key = nn.Linear(hidden_dim * 3 + 2, hidden_dim)
        self.halt_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.evidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.halt_bias = halt_bias

    def _step_state(
        self,
        hidden: torch.Tensor,
        node_repr: torch.Tensor,
        elapsed_time: torch.Tensor,
        evidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_input = torch.cat([node_repr, elapsed_time.view(1), evidence.view(1)], dim=0).unsqueeze(0)
        hidden = self.state_update(state_input, hidden.unsqueeze(0)).squeeze(0)
        evidence = torch.sigmoid(self.evidence_head(hidden)).view(())
        return hidden, evidence

    def step(
        self,
        current: int,
        hidden: torch.Tensor,
        current_repr: torch.Tensor,
        evidence: torch.Tensor,
        elapsed: torch.Tensor,
        node_features: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        neighbors: Sequence[torch.Tensor],
        neighbor_edges: Sequence[torch.Tensor],
        allowed_nodes: Optional[torch.Tensor] = None,
    ) -> AgentStepOutput:
        candidate_nodes = neighbors[current]
        candidate_edges = neighbor_edges[current]
        if allowed_nodes is not None and candidate_nodes.numel() > 0:
            mask = allowed_nodes[candidate_nodes]
            candidate_nodes = candidate_nodes[mask]
            candidate_edges = candidate_edges[mask]

        if candidate_nodes.numel() == 0:
            zero = hidden.new_tensor(0.0)
            return AgentStepOutput(
                action_hard=hidden.new_zeros(0),
                next_node=None,
                next_edge=None,
                action_log_prob=zero,
                entropy=zero,
                halted=True,
            )

        query = self.query(
            torch.cat(
                [hidden, current_repr, elapsed.view(1), evidence.view(1)],
                dim=0,
            )
        )
        candidate_repr = self.node_encoder(node_features[candidate_nodes])
        edge_repr = self.edge_encoder(
            torch.cat(
                [edge_attr[candidate_edges], edge_time[candidate_edges].unsqueeze(-1)],
                dim=-1,
            )
        )
        key_input = torch.cat(
            [
                candidate_repr + edge_repr,
                hidden.unsqueeze(0).expand(candidate_nodes.size(0), -1),
                current_repr.unsqueeze(0).expand(candidate_nodes.size(0), -1),
                edge_time[candidate_edges].unsqueeze(-1),
                evidence.expand(candidate_nodes.size(0)).unsqueeze(-1),
            ],
            dim=-1,
        )
        move_logits = (query.unsqueeze(0) * self.candidate_key(key_input)).sum(dim=-1)
        halt_logit = self.halt_head(torch.cat([hidden, elapsed.view(1), evidence.view(1)], dim=0)) + self.halt_bias
        logits = torch.cat([move_logits, halt_logit.view(1)], dim=0)

        action_hard = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True, dim=0)
        log_probs = F.log_softmax(logits, dim=0)
        probs = F.softmax(logits, dim=0)
        action_log_prob = torch.sum(action_hard * log_probs)
        entropy = -(probs * log_probs).sum()

        action_index = int(action_hard.argmax(dim=0).item())
        if action_index == candidate_nodes.size(0):
            return AgentStepOutput(
                action_hard=action_hard,
                next_node=None,
                next_edge=None,
                action_log_prob=action_log_prob,
                entropy=entropy,
                halted=True,
            )

        return AgentStepOutput(
            action_hard=action_hard,
            next_node=int(candidate_nodes[action_index].item()),
            next_edge=int(candidate_edges[action_index].item()),
            action_log_prob=action_log_prob,
            entropy=entropy,
            halted=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        neighbors: Sequence[torch.Tensor],
        neighbor_edges: Sequence[torch.Tensor],
        node_time: torch.Tensor,
        start_node: int,
        allowed_nodes: Optional[torch.Tensor] = None,
        level: str = "node",
    ) -> AgentRollout:
        return self.rollout(
            node_features=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_time,
            neighbors=neighbors,
            neighbor_edges=neighbor_edges,
            node_time=node_time,
            start_node=start_node,
            allowed_nodes=allowed_nodes,
            level=level,
        )

    def rollout(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        neighbors: Sequence[torch.Tensor],
        neighbor_edges: Sequence[torch.Tensor],
        node_time: torch.Tensor,
        start_node: int,
        allowed_nodes: Optional[torch.Tensor] = None,
        level: str = "node",
    ) -> AgentRollout:
        device = node_features.device
        current = int(start_node)
        hidden = node_features.new_zeros(self.hidden_dim)
        evidence = node_features.new_tensor(0.0)
        root_time = node_time[start_node]

        visited = [current]
        traversed_edges: List[int] = []
        log_probs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        elapsed_times = [node_features.new_tensor(0.0)]
        evidence_trace = []
        prefix_lengths: List[int] = []
        halted = False
        halt_step = self.max_steps

        for step in range(self.max_steps):
            current_repr = self.node_encoder(node_features[current])
            elapsed = (node_time[current] - root_time).clamp_min(0.0)
            hidden, evidence = self._step_state(hidden, current_repr, elapsed, evidence)
            evidence_trace.append(evidence)

            step_output = self.step(
                current=current,
                hidden=hidden,
                current_repr=current_repr,
                evidence=evidence,
                elapsed=elapsed,
                node_features=node_features,
                edge_attr=edge_attr,
                edge_time=edge_time,
                neighbors=neighbors,
                neighbor_edges=neighbor_edges,
                allowed_nodes=allowed_nodes,
            )

            if step_output.action_hard.numel() == 0:
                halted = True
                halt_step = step
                break

            log_probs.append(step_output.action_log_prob)
            entropies.append(step_output.entropy)

            if step_output.halted:
                halted = True
                halt_step = step
                prefix_lengths.append(len(visited))
                break

            traversed_edges.append(int(step_output.next_edge))
            current = int(step_output.next_node)
            visited.append(current)
            elapsed_times.append((node_time[current] - root_time).clamp_min(0.0))
            prefix_lengths.append(len(visited))

        if not evidence_trace:
            evidence_trace.append(node_features.new_tensor(0.0))
        if len(prefix_lengths) < len(log_probs):
            prefix_lengths.extend([len(visited)] * (len(log_probs) - len(prefix_lengths)))

        return AgentRollout(
            visited_nodes=torch.tensor(visited, dtype=torch.long, device=device),
            traversed_edges=torch.tensor(traversed_edges, dtype=torch.long, device=device)
            if traversed_edges
            else torch.empty(0, dtype=torch.long, device=device),
            action_log_probs=torch.stack(log_probs) if log_probs else node_features.new_zeros(0),
            entropies=torch.stack(entropies) if entropies else node_features.new_zeros(0),
            elapsed_times=torch.stack(elapsed_times),
            causal_evidence=torch.stack(evidence_trace),
            prefix_lengths=torch.tensor(prefix_lengths, dtype=torch.long, device=device)
            if prefix_lengths
            else torch.empty(0, dtype=torch.long, device=device),
            halted=halted,
            halt_step=halt_step,
            level=level,
        )


class ClusterAgent(TraversalAgent):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, max_steps: int, gumbel_tau: float) -> None:
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, max_steps=max_steps, gumbel_tau=gumbel_tau, halt_bias=0.1)


class NodeAgent(TraversalAgent):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, max_steps: int, gumbel_tau: float) -> None:
        super().__init__(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, max_steps=max_steps, gumbel_tau=gumbel_tau, halt_bias=-0.1)


class DualAgentEnvironment(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        max_cluster_steps: int,
        max_node_steps: int,
        gumbel_tau: float = 1.0,
        lambda_spurious: float = 0.5,
        entropy_coef: float = 0.01,
        local_hops: int = 1,
    ) -> None:
        super().__init__()
        self.cluster_agent = ClusterAgent(node_dim=node_dim, edge_dim=edge_dim + 1, hidden_dim=hidden_dim, max_steps=max_cluster_steps, gumbel_tau=gumbel_tau)
        self.node_agent = NodeAgent(node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, max_steps=max_node_steps, gumbel_tau=gumbel_tau)
        self.cluster_reward_scorer = CausalPatternScorer(feature_dim=node_dim, hidden_dim=hidden_dim, lambda_spurious=lambda_spurious)
        self.node_reward_scorer = CausalPatternScorer(feature_dim=node_dim, hidden_dim=hidden_dim, lambda_spurious=lambda_spurious)
        self.entropy_coef = entropy_coef
        self.local_hops = local_hops

    def _build_guided_mask(self, graph: HierarchicalGraph, cluster_path: torch.Tensor) -> torch.Tensor:
        allowed = torch.zeros(graph.micro_x.size(0), dtype=torch.bool, device=graph.micro_x.device)
        if cluster_path.numel() == 0:
            allowed[graph.root] = True
            return allowed

        cluster_set = torch.zeros(graph.macro_x.size(0), dtype=torch.bool, device=graph.micro_x.device)
        cluster_set[cluster_path.unique()] = True
        allowed = cluster_set[graph.cluster_assignment]

        for _ in range(self.local_hops):
            expanded = allowed.clone()
            for src in range(graph.micro_x.size(0)):
                if allowed[src]:
                    nbrs = graph.micro_neighbors[src]
                    expanded[nbrs] = True
                    incoming = graph.micro_edge_index[1] == src
                    if incoming.any():
                        expanded[graph.micro_edge_index[0, incoming]] = True
            allowed = expanded
        allowed[graph.root] = True
        return allowed

    def _prefix_summary_from_nodes(
        self,
        graph: HierarchicalGraph,
        rollout: AgentRollout,
        guided_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_steps = rollout.action_log_probs.numel()
        causal_summary = graph.micro_x.new_zeros((num_steps, graph.micro_x.size(-1)))
        spurious_summary = graph.micro_x.new_zeros((num_steps, graph.micro_x.size(-1)))
        if num_steps == 0:
            return causal_summary, spurious_summary

        trajectory_nodes = rollout.visited_nodes
        for step in range(num_steps):
            prefix_length = int(rollout.prefix_lengths[step].item()) if rollout.prefix_lengths.numel() > 0 else trajectory_nodes.numel()
            prefix_nodes = trajectory_nodes[:prefix_length]
            prefix_feat = graph.micro_x[prefix_nodes]
            if prefix_nodes.numel() > 1:
                edge_dt = graph.micro_node_time[prefix_nodes[1:]] - graph.micro_node_time[prefix_nodes[:-1]]
                temporal_ok = torch.cat(
                    [torch.ones(1, dtype=torch.bool, device=prefix_nodes.device), edge_dt <= graph.time_threshold],
                    dim=0,
                )
            else:
                temporal_ok = torch.ones(prefix_nodes.numel(), dtype=torch.bool, device=prefix_nodes.device)

            if guided_mask is None:
                guided_ok = torch.ones_like(temporal_ok)
            else:
                guided_ok = guided_mask[prefix_nodes]

            causal_mask = guided_ok & temporal_ok
            spurious_mask = ~causal_mask
            if causal_mask.any():
                causal_summary[step] = prefix_feat[causal_mask].mean(dim=0)
            if spurious_mask.any():
                spurious_summary[step] = prefix_feat[spurious_mask].mean(dim=0)
        return causal_summary, spurious_summary

    def _prefix_summary_from_clusters(self, graph: HierarchicalGraph, rollout: AgentRollout) -> tuple[torch.Tensor, torch.Tensor]:
        num_steps = rollout.action_log_probs.numel()
        causal_summary = graph.macro_x.new_zeros((num_steps, graph.macro_x.size(-1)))
        spurious_summary = graph.macro_x.new_zeros((num_steps, graph.macro_x.size(-1)))
        if num_steps == 0:
            return causal_summary, spurious_summary

        trajectory_clusters = rollout.visited_nodes
        for step in range(num_steps):
            prefix_length = int(rollout.prefix_lengths[step].item()) if rollout.prefix_lengths.numel() > 0 else trajectory_clusters.numel()
            prefix_nodes = trajectory_clusters[:prefix_length]
            prefix_feat = graph.macro_x[prefix_nodes]
            revisited = torch.zeros(prefix_nodes.numel(), dtype=torch.bool, device=prefix_nodes.device)
            for idx in range(prefix_nodes.numel()):
                revisited[idx] = (prefix_nodes[:idx] == prefix_nodes[idx]).any()
            if prefix_nodes.numel() > 1:
                delta_t = graph.macro_node_time[prefix_nodes[1:]] - graph.macro_node_time[prefix_nodes[:-1]]
                temporal_ok = torch.cat(
                    [torch.ones(1, dtype=torch.bool, device=prefix_nodes.device), delta_t >= 0],
                    dim=0,
                )
            else:
                temporal_ok = torch.ones(prefix_nodes.numel(), dtype=torch.bool, device=prefix_nodes.device)
            causal_mask = temporal_ok & (~revisited)
            spurious_mask = ~causal_mask
            if causal_mask.any():
                causal_summary[step] = prefix_feat[causal_mask].mean(dim=0)
            if spurious_mask.any():
                spurious_summary[step] = prefix_feat[spurious_mask].mean(dim=0)
        return causal_summary, spurious_summary

    def _stack_prefix_summaries(
        self,
        summaries: List[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_steps = max((pair[0].size(0) for pair in summaries), default=0)
        if max_steps == 0:
            dummy = torch.zeros((len(summaries), 0, 1), device=device)
            mask = torch.zeros((len(summaries), 0), dtype=torch.bool, device=device)
            return dummy, dummy, mask

        feature_dim = summaries[0][0].size(-1)
        causal = torch.zeros((len(summaries), max_steps, feature_dim), device=device)
        spurious = torch.zeros((len(summaries), max_steps, feature_dim), device=device)
        mask = torch.zeros((len(summaries), max_steps), dtype=torch.bool, device=device)
        for idx, (causal_summary, spurious_summary) in enumerate(summaries):
            steps = causal_summary.size(0)
            if steps == 0:
                continue
            causal[idx, :steps] = causal_summary
            spurious[idx, :steps] = spurious_summary
            mask[idx, :steps] = True
        return causal, spurious, mask

    def _policy_gradient_loss(
        self,
        rollouts: List[AgentRollout],
        reward_trace: torch.Tensor,
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for graph_idx, rollout in enumerate(rollouts):
            num_actions = rollout.action_log_probs.numel()
            if num_actions == 0:
                continue
            rewards = reward_trace[graph_idx, :num_actions].detach()
            advantages = rewards - rewards.mean() if num_actions > 1 else rewards
            entropy_bonus = rollout.entropies.mean() if rollout.entropies.numel() > 0 else reward_trace.new_tensor(0.0)
            losses.append(-(advantages * rollout.action_log_probs).mean() - self.entropy_coef * entropy_bonus)
        if losses:
            return torch.stack(losses).mean()
        return reward_trace.new_tensor(0.0)

    def forward(
        self,
        graphs: Sequence[HierarchicalGraph],
        labels: torch.Tensor,
    ) -> DualAgentOutput:
        # labels: [batch_size]
        cluster_rollouts: List[AgentRollout] = []
        node_rollouts: List[AgentRollout] = []
        guided_masks: List[torch.Tensor] = []

        for graph in graphs:
            cluster_edge_time = graph.macro_edge_weight if graph.macro_edge_weight.numel() > 0 else graph.macro_x.new_zeros(0)
            cluster_rollout = self.cluster_agent(
                node_features=graph.macro_x,
                edge_index=graph.macro_edge_index,
                edge_attr=graph.macro_edge_attr,
                edge_time=cluster_edge_time,
                neighbors=graph.macro_neighbors,
                neighbor_edges=graph.macro_neighbor_edges,
                node_time=graph.macro_node_time,
                start_node=graph.root_cluster,
                allowed_nodes=None,
                level="cluster",
            )
            guided_mask = self._build_guided_mask(graph, cluster_rollout.visited_nodes)
            node_rollout = self.node_agent(
                node_features=graph.micro_x,
                edge_index=graph.micro_edge_index,
                edge_attr=graph.micro_edge_attr,
                edge_time=graph.micro_edge_time,
                neighbors=graph.micro_neighbors,
                neighbor_edges=graph.micro_neighbor_edges,
                node_time=graph.micro_node_time,
                start_node=graph.root,
                allowed_nodes=guided_mask,
                level="node",
            )

            cluster_rollouts.append(cluster_rollout)
            node_rollouts.append(node_rollout)
            guided_masks.append(guided_mask)

        device = labels.device
        cluster_prefix = [self._prefix_summary_from_clusters(graph, rollout) for graph, rollout in zip(graphs, cluster_rollouts)]
        node_prefix = [
            self._prefix_summary_from_nodes(graph, rollout, guided_mask)
            for graph, rollout, guided_mask in zip(graphs, node_rollouts, guided_masks)
        ]
        cluster_causal, cluster_spurious, cluster_mask = self._stack_prefix_summaries(cluster_prefix, device)
        node_causal, node_spurious, node_mask = self._stack_prefix_summaries(node_prefix, device)
        # cluster_causal/node_causal: [batch_size, max_steps, node_dim]
        # cluster_mask/node_mask: [batch_size, max_steps]

        cluster_reward_trace, cluster_aux_loss = self.cluster_reward_scorer(cluster_causal, cluster_spurious, labels, cluster_mask)
        node_reward_trace, node_aux_loss = self.node_reward_scorer(node_causal, node_spurious, labels, node_mask)
        # reward_trace: [batch_size, max_steps]

        cluster_rl_loss = self._policy_gradient_loss(cluster_rollouts, cluster_reward_trace)
        node_rl_loss = self._policy_gradient_loss(node_rollouts, node_reward_trace)
        reward_aux_loss = 0.5 * (cluster_aux_loss + node_aux_loss)
        rl_loss = 0.5 * (cluster_rl_loss + node_rl_loss)

        return DualAgentOutput(
            cluster_rollouts=cluster_rollouts,
            node_rollouts=node_rollouts,
            guided_masks=guided_masks,
            rl_loss=rl_loss,
            reward_aux_loss=reward_aux_loss,
            reward_trace=node_reward_trace,
            cluster_reward_trace=cluster_reward_trace,
        )
