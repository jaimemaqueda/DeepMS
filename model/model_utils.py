import torch
from mslib.macro import EOS_IDX


def _make_seq_first(*args):
    # N, S, ... -> S, N, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)


def _make_batch_first(*args):
    # S, N, ... -> N, S, ...
    if len(args) == 1:
        arg, = args
        return arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None
    return (*(arg.permute(1, 0, *range(2, arg.dim())) if arg is not None else None for arg in args),)


def _get_key_padding_mask(operations, seq_dim=0):
    """
    Args:
        operations: Shape [S, ...]
    """
    with torch.no_grad():
        key_padding_mask = (operations == EOS_IDX).cumsum(dim=seq_dim) > 0

        if seq_dim == 0:
            return key_padding_mask.transpose(0, 1)
        return key_padding_mask


def _get_padding_mask(operations, seq_dim=0, extended=False):
    with torch.no_grad():
        threshold = 3 if extended else 0 # Extend padding_mask by 3 positions to include EOS in the loss
        padding_mask = (operations == EOS_IDX).cumsum(dim=seq_dim) <= threshold
        padding_mask = padding_mask.float()

        if seq_dim == 0:
            return padding_mask.unsqueeze(-1)
        return padding_mask


def _get_visibility_mask(operations, seq_dim=0):
    """
    Args:
        operations: Shape [S, ...]
    """
    S = operations.size(seq_dim)
    with torch.no_grad():
        visibility_mask = (operations == EOS_IDX).sum(dim=seq_dim) < S - 1

        if seq_dim == 0:
            return visibility_mask.unsqueeze(-1)
        return visibility_mask
