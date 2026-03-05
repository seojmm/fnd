import torch
from torch import Tensor, LongTensor
from typing import Optional

from torch_scatter import scatter_add, scatter_max, scatter_mean
from torch_scatter.composite import scatter_softmax
from torch_scatter.utils import broadcast
from torch_geometric.utils.num_nodes import maybe_num_nodes


def spmm(index: Tensor, value: Tensor, m: int, n: int, matrix: Tensor, reduce: str = "sum") -> Tensor:
    """Sparse matrix (COO) x dense matrix multiplication with grouped reduction."""
    assert n == matrix.size(-2)

    row, col = index[0], index[1]
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    if reduce == "sum":
        out = scatter_add(out, row, dim=-2, dim_size=m)
    elif reduce in {"mean", "log", "sqrt"}:
        out = scatter_add(out, row, dim=-2, dim_size=m)

        dim_size = out.size(-2)
        index_dim = -2
        if index_dim < 0:
            index_dim = index_dim + out.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        count = scatter_add(value.detach(), index[0], index_dim, None, dim_size)
        count[count < 1] = 1
        if reduce == "log":
            count = count / torch.log2(count + 1)
        elif reduce == "sqrt":
            count = torch.sqrt(count)
        count = broadcast(count, out, -2)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode="floor")
    else:
        out = scatter_max(out, row, dim=-2, dim_size=m)[0]

    return out


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    if reduce in {"sum", "add"}:
        return scatter_add(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    if reduce in {"log", "sqrt"}:
        out = scatter_add(src, index, dim, out, dim_size)
        count = scatter_add(torch.ones_like(index, dtype=src.dtype, device=src.device), index, dim, None, dim_size)
        count[count < 1] = 1
        if reduce == "sqrt":
            count = torch.sqrt(count)
        else:
            count = count / torch.log2(count + 1)
        count = broadcast(count, out, -2)
        if out.is_floating_point():
            out.true_divide_(count)
        else:
            out.div_(count, rounding_mode="floor")
        return out
    return scatter_max(src, index, dim, out, dim_size)[0]


def grouped_softmax(src: Tensor, index: LongTensor, num_nodes: int = 0) -> Tensor:
    num_nodes = maybe_num_nodes(index, num_nodes)
    return scatter_softmax(src, index, dim=0, dim_size=num_nodes)


def gumbel_softmax(
    src: Tensor,
    index: LongTensor,
    num_nodes: int = 0,
    hard: bool = True,
    tau: float = 1.0,
    return_soft: bool = False,
):
    """
    Sparse grouped Gumbel-Softmax.
    Returns straight-through one-hot choices when hard=True.
    """
    num_nodes = maybe_num_nodes(index, num_nodes)

    gumbels = -torch.empty_like(src, memory_format=torch.legacy_contiguous_format).exponential_().log()
    logits = (src + gumbels) / tau
    y_soft = scatter_softmax(logits, index, dim=0, dim_size=num_nodes)

    if hard:
        max_index = scatter_max(y_soft, index, dim=0, dim_size=num_nodes)[1]
        y_hard = torch.zeros_like(y_soft, memory_format=torch.legacy_contiguous_format).scatter_(0, max_index, 1.0)
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    if return_soft:
        return y, y_soft
    return y
