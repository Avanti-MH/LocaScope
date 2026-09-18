'''Patches -> `[N, L, D]`, the shape `common.Head.Head` reads. Two routes:
`encode_raw` for a FROZEN encoder (inference only, chunked, no graph),
`trunk_raw` for a FINE-TUNED trunk (a graph attached, not chunked).
'''
from __future__ import annotations

import torch


def encode_raw(encoder, patches: torch.Tensor, batch_size: int,
               device) -> torch.Tensor:
    '''FROZEN route. `patches` uint8 `[N, tile, tile, 3]` -> the encoder's
    UN-REDUCED exit, `[N, L, D]`: `tokens()` for a ViT, `spatial()` reshaped
    for a CNN. Chunked, because `TileEncoder._run` is `@torch.no_grad()` and
    returns to CPU -- there is no graph to keep alive, so chunking genuinely
    bounds memory here (unlike `trunk_raw`, see there).'''
    kind = encoder.model_spec.kind
    images = [p.numpy() for p in patches]
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        out.append(encoder.tokens(batch) if kind == 'tokens'
                   else encoder.spatial(batch).flatten(2).transpose(1, 2))
    return torch.cat(out).to(device)


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
