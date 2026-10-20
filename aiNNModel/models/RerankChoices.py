'''The registry of the second phase's networks: name -> (module, its
hyper-parameter dataclass). `cli/train.py` builds from it and
`common/Checkpoints.build_rerank_from_checkpoint` rebuilds from it, so the
generic checkpoint layer never imports a training package (the reason
`PrototypeChoices` lives here too).

    model = build_rerank('tilegt', {'in_dim': 3072, 'layers': 4})

`tilegt_ctx` has two dataclasses (trunk and context); its hyper-parameters are
`{'trunk': {...}, 'context': {...}}`.
'''
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict

import torch.nn as nn

from RerankHeads import (CandidateContext, ContextCfg, CorrNet, CorrNetCfg, EmbedCfg,
                         StackCfg, StackNet, TileGT, TileGTCfg, TileGTContext, WinEmbedder)


def _tuple_shapes(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A JSON round trip turns the shapes into lists of lists."""
    out: Dict[str, Any] = dict(cfg)
    if 'shapes' in out:
        out['shapes'] = tuple(tuple(int(v) for v in s) for s in out['shapes'])
    return out


def _tilegt_ctx(cfg: Dict[str, Any]) -> nn.Module:
    return TileGTContext(TileGTCfg(**cfg.get('trunk', {})), ContextCfg(**cfg.get('context', {})))


RERANK_CHOICES: Dict[str, Callable[[Dict[str, Any]], nn.Module]] = {
    'corrnet': lambda cfg: CorrNet(CorrNetCfg(**cfg)),
    'tilegt': lambda cfg: TileGT(TileGTCfg(**cfg)),
    'tilegt_ctx': _tilegt_ctx,
    'winembed': lambda cfg: WinEmbedder(EmbedCfg(**_tuple_shapes(cfg))),
    'stack': lambda cfg: StackNet(StackCfg(**cfg)),
}


def build_rerank(arch: str, cfg: Dict[str, Any]) -> nn.Module:
    """The network `arch` with hyper-parameters `cfg` (a dict, as a checkpoint
    holds it)."""
    try:
        return RERANK_CHOICES[arch](dict(cfg))
    except KeyError:
        raise KeyError(f'no network {arch!r}; the networks are {sorted(RERANK_CHOICES)}') from None


def hparams_of(model: nn.Module) -> Dict[str, Any]:
    """The hyper-parameters of a network `build_rerank` made, as a dict."""
    if isinstance(model, TileGTContext):
        return {'trunk': asdict(model.trunk.cfg), 'context': asdict(model.context.cfg)}
    cfg = getattr(model, 'cfg', None)
    if cfg is None:
        raise TypeError(f'{type(model).__name__} has no cfg to record')
    return asdict(cfg)


__all__ = ['RERANK_CHOICES', 'build_rerank', 'hparams_of', 'CandidateContext']
