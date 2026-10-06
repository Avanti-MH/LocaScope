'''Patches -> `[N, L, D]`, the shape `common.Head.Head` reads. Two routes:
`encode_raw` for a FROZEN encoder (inference only, chunked, no graph),
`trunk_raw` for a FINE-TUNED trunk (a graph attached, not chunked).
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass
class LayerTokens:
    """What `encode_raw` returns when a head asked for encoder layers: the last
    block's tokens (`last`, exactly what `encode_raw` returns otherwise) and the
    requested blocks' layer tokens by absolute 0-based index. `Head` reads
    `last` for the CLS and for a single-layer reduction, `layers` for a mix."""
    last: torch.Tensor                     # [N, T, D]
    layers: Dict[int, torch.Tensor]        # block -> [N, T, D]


def encode_raw(encoder, patches: torch.Tensor, batch_size: int,
               device, layers: Sequence[int] = ()):
    '''FROZEN route. `patches` uint8 `[N, tile, tile, 3]` -> the encoder's
    UN-REDUCED exit, `[N, L, D]`: `tokens()` for a ViT, `spatial()` reshaped
    for a CNN, ON `device` -- the tokens are produced there and stay there.
    Chunked, because `TileEncoder._run` is `@torch.no_grad()` -- there is no
    graph to keep alive, so chunking bounds the forward's activations (unlike
    `trunk_raw`, see there); the result is the batch the caller already chose.

    `layers` (absolute 0-based blocks, the union every head in the call needs)
    returns a `LayerTokens` instead, from ONE forward per chunk; empty -- the
    default, and every caller before it existed -- returns the tensor as
    before.'''
    kind = encoder.model_spec.kind
    keep = lambda t: t                                        # noqa: E731
    # The uint8 batch goes to the encoder as it is (one transform, on the
    # card) and what comes back never leaves it. `keep` is a no-op since the
    # encoder's exits stopped moving results to the host (2026-10-06); it used
    # to stop `_run`'s `.cpu()`, which `.to(device)` then undid.
    if layers:
        if kind != 'tokens':
            raise TypeError(f'encoder layers need a token model; this one is '
                            f'{kind!r}')
        last = encoder.depth - 1
        want = sorted(set(int(i) for i in layers) | {last})
        stacked = torch.cat([encoder.layer_tokens(patches[s:s + batch_size], want,
                                                  reduce=keep)
                             for s in range(0, len(patches), batch_size)]).to(device)
        by_block = {b: stacked[:, k] for k, b in enumerate(want)}
        return LayerTokens(last=by_block[last], layers=by_block)
    out = []
    for start in range(0, len(patches), batch_size):
        batch = patches[start:start + batch_size]
        out.append(encoder.tokens(batch, reduce=keep) if kind == 'tokens'
                   else encoder.spatial(batch, reduce=keep).flatten(2).transpose(1, 2))
    return torch.cat(out).to(device)


def check_layer_tokens(encoder, patches: torch.Tensor, n: int = 4) -> tuple:
    """`((cos to tokens(), cos to the decoy), ok)` on `n` patches: the last
    block of `layer_tokens` against `tokens()`, which it claims to equal, and
    the block before the last against `tokens()`, which it must not.

    Two paths through the same trunk (`forward_intermediates` against the
    plain forward) that only agree if the prefix order and the final norm are
    what `layer_tokens` says. A wrong one does not raise: a mix_ head trains on
    whatever it is handed and reports a number. The decoy is what makes
    agreement mean something -- a neighbouring block is close too, so the gate
    is a margin over it, not a threshold."""
    images = [p.numpy() for p in patches[:n]]
    last = encoder.depth - 1
    plain = encoder.tokens(images)
    both = encoder.layer_tokens(images, [last - 1, last])
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(  # noqa: E731
        a.flatten(0, 1), b.flatten(0, 1), dim=-1).min())
    same, decoy = cos(both[:, 1], plain), cos(both[:, 0], plain)
    return (same, decoy), same > 0.999 and decoy < same - 0.01


def normalise_patches(encoder, patches: torch.Tensor, device) -> torch.Tensor:
    '''uint8 `[N, tile, tile, 3]` -> normalised float `[N, 3, tile, tile]` on
    `device`, using the encoder's OWN mean/std so the trunk sees the
    distribution its weights were trained on.

    The Resize and CenterCrop halves of `cfg.transform` are deliberately NOT
    applied. `TransformConfig.build()` is `Resize -> CenterCrop -> ToTensor ->
    Normalize` and only the last two belong here: the patches are already
    the trunk's input size, and a `crop_pct` below 1 would throw away the
    outer ring of every one of them -- free for retrieval, where one vector
    comes out, and not free here, where the discarded ring is training
    signal. (`spatial()`/`features()` apply the whole chain, which is exactly
    why `TileEncoder.features_with_grad` exists.)
    '''
    t = encoder.cfg.transform
    if t.preprocess != 'none':
        raise ValueError(
            f"transform.preprocess={t.preprocess!r} is not applied here -- this "
            f"function does Normalize only. Add the branch before using a "
            f"'grey' encoder rather than silently feeding it RGB")
    x = patches.to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(t.mean, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(t.std, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def trunk_raw(encoder, patches: torch.Tensor, device) -> torch.Tensor:
    '''FINE-TUNED route. `[N, L, C]`, the shape `common.Head.Head` reads, with
    a graph attached.

    `features_with_grad(exit_name='spatial')` hands back `[N, C, H, W]`;
    flattening it to `[N, H*W, C]` is the same layout `TileEncoder.pool`'s own
    rank-4 branch produces before it calls `pooling_kinds`, and it is the
    layout `common.Head.pooled_view`/`grid_view` read. With `num_prefix=0`
    those two then mean GAP and the full cell grid -- so the fine-tuned route
    reduces with the SAME code as the frozen route, and the two are
    comparable because nothing about the head's input differs except which
    trunk produced it.

    `num_prefix=0` IS RIGHT FOR A ViT TRUNK TOO, not just for a CNN: the
    spatial exit has already dropped the prefix (`_vit_spatial_forward` asks
    `forward_intermediates`, which strips `num_prefix_tokens` itself), so
    there is nothing left to slice off. The encoder's own
    `model_spec.num_prefix` would be 1 or 9 here and passing it would eat that
    many real patch cells -- it describes the TOKEN exit, which this is not.

    NOT chunked, unlike `encode_raw`. Chunk-and-concat keeps the graph, so it
    would save no memory during training -- the activations of every chunk
    stay alive until `backward()`. The caller's own batch size is the memory
    knob for this route.
    '''
    maps = encoder.features_with_grad(normalise_patches(encoder, patches, device),
                                      exit_name='spatial')
    return maps.flatten(2).transpose(1, 2)
