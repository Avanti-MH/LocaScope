"""The name of an arm of the stage 2 bench, and what it means.

    uni2:BagOfTiles::cls_avg+gap::gigapath:CSLS::grid2x2+csls@100
    └──── retriever ───────────┘ └──── reranker ─────────────────┘

An arm is a RETRIEVER and then any number of RERANKERS, joined by `::`. Each one is

    [encoder:]Method::pooling[+operation][@K]

    encoder     which encoder reads the tiles; left out, the retriever's own (a reranker may name
                another: it encodes what it needs itself)
    Method      what finds (a retriever) or reorders (a reranker) the windows
    pooling     how a tile's tokens are reduced to one vector (`cls`, `cls_avg`, `rings3`, ...);
                `raw` for a method that does not take one (the patch tokens as they are)
    +operation  a choice INSIDE the method: how the tiles of one field of view are put together
                (`gap` the window's mean vector, `quad` its four quadrants'), which way a score is
                computed, which of its outputs the list is ordered by. A method with no such choice
                has no `+`
    @K          keep the first K windows of the list this segment leaves. A retriever without it
                hands over its whole production list (`--k`); a reranker without it keeps all it
                was given. A list shorter than K is kept whole

The text of an arm is canonical: the retriever's encoder is always written, a reranker's only when it
is not the retriever's. `parse_arm(text, default_encoder).text()` is that text, so one arm has one name.

This file knows the NAMES and what each takes -- not how a method works. `RetrieverArms.py` (bench)
builds the retrievers and rerankers a spec names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: the pooling of a method that takes none: the patch tokens, as they are
RAW: str = 'raw'

SEP: str = '::'


@dataclass(frozen=True)
class MethodInfo:
    """What a method takes. `ops` are the operations after `+` (empty: it takes none), `raw` that its
    pooling must be `raw`. `retriever`: it may be the first part of an arm (it searches the slide's whole grid
    of windows, or is handed all of them); `reranker`: it may follow another part (it scores the list it is
    handed). A method may be both (`Retriever.takes_wsi`, `Retriever.takes_candidates` say how)."""
    ops: Tuple[str, ...] = ()
    raw: bool = False
    retriever: bool = True
    reranker: bool = False


#: Every method an arm may name. The operations of a retriever are the same choice the stage makes
#: (`SlidingWindow` is `SlidingWinSimRot`'s window score; `min` is the bench's alone); a reranker's are
#: either an algorithm (`CSLS`: `csls` | `rank`) or the output column the list is ordered by.
METHODS: Dict[str, MethodInfo] = {
    # first phase: they make the candidate list
    'SlidingWindow': MethodInfo(ops=('mean', 'geomean', 'min')),
    'BagOfTiles':    MethodInfo(ops=('gap', 'quad')),
    'TileVote':      MethodInfo(ops=('exact', 'hnsw', 'ivf')),
    'ClusterBow':    MethodInfo(),
    'WinEmbed':      MethodInfo(),
    # second phase: they reorder it
    'Dual':      MethodInfo(ops=('dual', 'cos_main'), retriever=False, reranker=True),
    'CSLS':      MethodInfo(ops=('csls', 'rank'), retriever=False, reranker=True),
    'Corr':      MethodInfo(ops=('diag', 'contrast', 'mnn', 'sinkhorn'), retriever=False, reranker=True),
    'CandGraph': MethodInfo(ops=('nbr_support', 'prominence', 'rot_margin'), raw=True, retriever=False, reranker=True),
    'Shift':     MethodInfo(retriever=False, reranker=True),
    'TokenMatch': MethodInfo(raw=True, retriever=False, reranker=True),
    'TileGT':    MethodInfo(ops=('tilegt', 'tilegt_rot'), retriever=False, reranker=True),
    'CorrNet':   MethodInfo(retriever=False, reranker=True),
    'XEnc':      MethodInfo(retriever=False, reranker=True),
}

#: The names of the proposal (`proposal.md` §4, §5.2, §6.4, §11), spelt here so an arm's name needs no torch;
#: `test_neural` holds them equal to `aiNNModel/models/GloRet.py`'s.
CONTEXT_EN: Tuple[str, ...] = ('SelfAttn', 'GNN', 'SptSelfAttn', 'Identity', 'CLSTrans')
GLOBAL_EN: Tuple[str, ...] = ('AttnPool', 'MaskedMean', 'CNN', 'GraphPool', 'Align', 'Identity')
SEARCHES: Tuple[str, ...] = ('exact', 'hnsw', 'ivf')
NO_MATCH: Tuple[str, ...] = ('dustbin', 'sinkhorn', 'sigmoid')


def learned_combo(context: str, global_en: str, arch: int) -> bool:
    """Whether `GloRet` (`arch` 1) or `LogLoRet` (2) has a method with this ContextEn and GlobalEn: `CLSTrans` goes
    with `Identity` and only it, `GraphPool` with `GNN`, `Align` is Arch 1's alone (GloRet.combo_problem)."""
    if (context == 'CLSTrans') != (global_en == 'Identity'):
        return False
    if global_en == 'GraphPool' and context != 'GNN':
        return False
    return not (global_en == 'Align' and arch != 1)


def _learned_methods() -> Dict[str, MethodInfo]:
    """`GloRet-{ContextEn}-{GlobalEn}` and `LogLoRet-{ContextEn}-{GlobalEn}`. Both can be the first part of an arm
    (GloRet searches the slide's windows; LogLoRet is handed all of them) and can follow another (they score the
    list they are given). GloRet's operation is how its window vectors are searched, none for `Align`, which has
    none; LogLoRet's is the no-match option it was trained with."""
    out: Dict[str, MethodInfo] = {}
    for context in CONTEXT_EN:
        for global_en in GLOBAL_EN:
            if learned_combo(context, global_en, 1):
                out[f'GloRet-{context}-{global_en}'] = MethodInfo(
                    ops=() if global_en == 'Align' else SEARCHES, retriever=True, reranker=True)
            if learned_combo(context, global_en, 2):
                out[f'LogLoRet-{context}-{global_en}'] = MethodInfo(ops=NO_MATCH, retriever=True, reranker=True)
    return out


METHODS.update(_learned_methods())


class ArmSyntaxError(ValueError):
    """The text is not an arm; the message says where."""


@dataclass(frozen=True)
class Segment:
    method: str
    pooling: str
    op: str = ''
    encoder: str = ''
    cut: int = 0                       # @K; 0: none

    def text(self, show_encoder: bool = True) -> str:
        head: str = f'{self.encoder}:' if self.encoder and show_encoder else ''
        spec: str = self.pooling + (f'+{self.op}' if self.op else '') + (f'@{self.cut}' if self.cut else '')
        return f'{head}{self.method}{SEP}{spec}'


@dataclass(frozen=True)
class ArmSpec:
    """A retriever and the rerankers after it, in order."""
    segments: Tuple[Segment, ...]

    @property
    def retriever(self) -> Segment:
        return self.segments[0]

    @property
    def rerankers(self) -> Tuple[Segment, ...]:
        return self.segments[1:]

    def prefix(self, n: int) -> 'ArmSpec':
        """The arm made of the first `n` segments: what the list was after them."""
        return ArmSpec(self.segments[:n])

    def text(self) -> str:
        """The canonical text: the retriever's encoder always, a reranker's when it differs."""
        enc: str = self.segments[0].encoder
        parts: List[str] = [self.segments[0].text()]
        for seg in self.segments[1:]:
            parts.append(seg.text(show_encoder=bool(seg.encoder) and seg.encoder != enc))
        return SEP.join(parts)


def _segment(method_token: str, spec_token: str, default_encoder: str, first: bool) -> Segment:
    encoder, colon, method = method_token.rpartition(':')
    if not colon:
        encoder = ''
    if method not in METHODS:
        raise ArmSyntaxError(f'{method!r} is not a method; they are {", ".join(METHODS)}')
    info: MethodInfo = METHODS[method]
    if first and not info.retriever:
        raise ArmSyntaxError(f'{method} reorders a list and cannot be an arm\'s first part; '
                             f'a retriever is {", ".join(m for m, i in METHODS.items() if i.retriever)}')
    if not first and not info.reranker:
        raise ArmSyntaxError(f'{method} finds windows and cannot follow another part of an arm; a part that can '
                             f'is {", ".join(m for m, i in METHODS.items() if i.reranker)}')
    rest: str = spec_token
    cut: int = 0
    if '@' in rest:
        rest, _, cut_text = rest.rpartition('@')
        if not cut_text.isdigit() or int(cut_text) < 1:
            raise ArmSyntaxError(f'{spec_token!r}: @K needs a whole number from 1, got {cut_text!r}')
        cut = int(cut_text)
    pooling, plus, op = rest.partition('+')
    if not pooling:
        raise ArmSyntaxError(f'{method}{SEP}{spec_token}: a pooling is needed (`raw` for none)')
    if info.raw and pooling != RAW:
        raise ArmSyntaxError(f'{method} takes no pooling: write `{method}{SEP}{RAW}`, not {pooling!r}')
    if plus and not info.ops:
        raise ArmSyntaxError(f'{method} has no operation to choose; write `{method}{SEP}{pooling}`, not +{op}')
    if plus and op not in info.ops:
        raise ArmSyntaxError(f'{method} has no operation {op!r}; it has {", ".join(info.ops)}')
    if info.ops and not plus:
        raise ArmSyntaxError(f'{method} needs one of its operations: {method}{SEP}{pooling}+{"|".join(info.ops)}')
    return Segment(method=method, pooling=pooling, op=op, encoder=encoder or (default_encoder if first else ''),
                   cut=cut)


def parse_arm(text: str, default_encoder: str = '') -> ArmSpec:
    """`ArmSpec` of `text`. `default_encoder` is the retriever's when the text names none; a reranker with
    no encoder of its own is given the retriever's."""
    tokens: List[str] = [t.strip() for t in text.strip().split(SEP)]
    if len(tokens) < 2 or len(tokens) % 2:
        raise ArmSyntaxError(f'{text!r}: an arm is `[encoder:]Method::pooling[+op]` repeated, joined by `::` '
                             f'(got {len(tokens)} piece(s))')
    segments: List[Segment] = []
    for i in range(0, len(tokens), 2):
        segments.append(_segment(tokens[i], tokens[i + 1], default_encoder, first=i == 0))
    head: Segment = segments[0]
    if not head.encoder:
        raise ArmSyntaxError(f'{text!r}: the retriever has no encoder; write `uni2:{head.method}{SEP}...`')
    fixed: List[Segment] = [head] + [s if s.encoder else Segment(s.method, s.pooling, s.op, head.encoder, s.cut)
                                     for s in segments[1:]]
    return ArmSpec(tuple(fixed))


def is_arm_text(text: str) -> bool:
    """True when `text` is in this grammar (has a `::`), False for the bench's older bare tokens."""
    return SEP in text


def expand(retrievers: List[ArmSpec], rerankers: List[Segment]) -> List[ArmSpec]:
    """Every retriever followed by every reranker: the k x h arms. A reranker with no encoder takes the
    retriever's."""
    out: List[ArmSpec] = []
    for r in retrievers:
        for seg in rerankers:
            fixed: Segment = seg if seg.encoder else Segment(seg.method, seg.pooling, seg.op, r.retriever.encoder, seg.cut)
            out.append(ArmSpec(r.segments + (fixed,)))
    return out


# ── an arm as the config of the chain it names ────────────────────────────────

#: A pooling written with this suffix searches the main lattice only (`cls_avg-M`).
MAIN_ONLY: str = '-M'


#: The `<data>` segment of the weights `training/RerankHead` and `training/GloRet` write at their default training
#: data (`--train-datasets @pure-train`; `Checkpoints.data_tag`).
RERANK_DATA: str = 'pure-train'


def weights_name(encoder: str, pooling: str, arch: str, data: str = RERANK_DATA) -> str:
    """`RerankHead/weights/<data>--<encoder>--<pooling>--<arch>--best.pt`, where training writes them
    (`Checkpoints.weights_name`, spelt here so an arm's config does not import torch)."""
    return 'RerankHead/weights/' + '--'.join((data, encoder, pooling, arch, 'best')) + '.pt'


def learned_weights(encoder: str, pooling: str, method: str, op: str = '', data: str = RERANK_DATA) -> str:
    """`GloRet/weights/<data>--<encoder>--<pooling>--<method>[--<op>]--best.pt`. The operation is in the name when
    it was trained in (LogLoRet's no-match option); GloRet's search is not."""
    tail: Tuple[str, ...] = (method, op, 'best') if op and method.startswith('LogLoRet-') else (method, 'best')
    return 'GloRet/weights/' + '--'.join((data, encoder, pooling, *tail)) + '.pt'


def _encoder(seg: Segment, pooling: str) -> Any:
    from TileEncoderFunc import encoder_config                     # noqa: PLC0415
    return encoder_config(seg.encoder, pooling=pooling).with_model(dtype='fp16')


def _offered(cfg: Any) -> Optional[Tuple[Any, ...]]:
    """The key of the level features a phase of this config offers downstream, None for one that reads none."""
    from stage2_retrieval.Context import feature_key               # noqa: PLC0415
    enc: Optional[Any] = getattr(cfg, 'encoder', None)
    if enc is None or not hasattr(cfg, 'overlap'):
        return None
    return feature_key(enc, int(cfg.tile_size), bool(cfg.overlap))


def segment_config(seg: Segment, *, first: bool, topk_in: int, raw_topk: int, min_sep_tiles: float,
                   offered: Sequence[Tuple[Any, ...]]) -> Any:
    """The config of the retriever `seg` names. A first phase lists `raw_topk` windows separated by
    `min_sep_tiles`; a reranker reads `topk_in` of the list before it, and reuses the level features of a
    phase before it (`offered`) when they are the very features it would make (the same encoder, pooling,
    tile and lattices). The other settings are each method's own defaults, the ones the bench has run with."""
    from stage2_retrieval import RerankerEncode as RE, RerankerNeural as RN, RerankerScorers as RS   # noqa: PLC0415
    pooling: str = seg.pooling
    overlap: bool = True
    if pooling.endswith(MAIN_ONLY):
        pooling, overlap = pooling[:-len(MAIN_ONLY)], False
    if pooling == RAW and seg.method not in ('CandGraph', 'TokenMatch'):
        raise NotImplementedError(f'{seg.text()}: the stage has no `raw` reading for {seg.method} yet')
    if pooling == 'attnpool':
        raise NotImplementedError(f'{seg.text()}: TilePool attnpool reads the patch tokens of every tile, which the '
                                  f'level features the stage keeps are not (they are pooled by the encoder)')
    if seg.method.startswith(('GloRet-', 'LogLoRet-')):
        weights: str = learned_weights(seg.encoder, pooling, seg.method, seg.op)
        if seg.method.startswith('GloRet-') and first:
            from stage2_retrieval.RetrieverGloRet import RetrieverGloRetConfig   # noqa: PLC0415
            return RetrieverGloRetConfig(encoder=_encoder(seg, pooling), weights=weights, method=seg.method,
                                         tile_size=256, overlap=overlap, raw_topk=raw_topk,
                                         min_sep_tiles=min_sep_tiles, search=seg.op or 'exact', band=16)
        from stage2_retrieval.Context import feature_key               # noqa: PLC0415
        enc_l: Any = _encoder(seg, pooling)
        reuse_l: Tuple[str, ...] = ('level_features',) if feature_key(enc_l, 256, overlap) in offered else ()
        from stage2_retrieval import RetrieverGloRet as RG             # noqa: PLC0415
        read: int = raw_topk if first else topk_in
        if seg.method.startswith('GloRet-'):
            return RG.RerankerGloRetConfig(encoder=enc_l, weights=weights, method=seg.method, tile_size=256,
                                           overlap=overlap, topk_in=read, reuse=reuse_l, batch=256)
        return RG.RerankerLogLoRetConfig(encoder=enc_l, weights=weights, method=seg.method, no_match=seg.op,
                                         tile_size=256, overlap=overlap, topk_in=read, reuse=reuse_l, batch=64)
    if first:
        enc: Any = _encoder(seg, pooling)
        common: Dict[str, Any] = dict(encoder=enc, tile_size=256, overlap=overlap, raw_topk=raw_topk,
                                      min_sep_tiles=min_sep_tiles)
        if seg.method == 'SlidingWindow':
            from stage2_retrieval.RetrieverSlidingWinSimRot import RetrieverSlidingWinSimRotConfig   # noqa: PLC0415
            return RetrieverSlidingWinSimRotConfig(**common, score=seg.op)
        if seg.method == 'BagOfTiles':
            from stage2_retrieval.RetrieverBagOfTiles import RetrieverBagOfTilesConfig   # noqa: PLC0415
            return RetrieverBagOfTilesConfig(**common, mode=seg.op, channels=256)
        if seg.method == 'TileVote':
            from stage2_retrieval.RetrieverTileVote import RetrieverTileVoteConfig   # noqa: PLC0415
            return RetrieverTileVoteConfig(**common, backend=seg.op, m=64, tau=0.05, smooth=1, hnsw_m=32,
                                           ef_search=128, nlist=1024, nprobe=16, chunk=8192)
        if seg.method == 'ClusterBow':
            from stage2_retrieval.RetrieverClusterBow import RetrieverClusterBowConfig   # noqa: PLC0415
            return RetrieverClusterBowConfig(**common, k1=32, k2=64, iters=10, fit_tiles=200000, buckets=4096,
                                             bigram_weight=0.5, seed=0, channels=256)
        if seg.method == 'WinEmbed':
            from stage2_retrieval.RetrieverWinEmbed import RetrieverWinEmbedConfig   # noqa: PLC0415
            return RetrieverWinEmbedConfig(**common, weights=weights_name(seg.encoder, pooling, 'winembed'),
                                           band=256)
        raise ValueError(f'{seg.method} is not a first-phase retriever')
    if seg.method == 'CandGraph':
        return RS.RerankerCandGraphConfig(tile_size=256, topk_in=topk_in, on='in', radius=2.0, order_by=seg.op)
    if seg.method == 'TokenMatch':
        return RE.RerankerTokenMatchConfig(encoder=_encoder(seg, ''), tile_size=256, topk_in=topk_in)
    enc = _encoder(seg, pooling)
    from stage2_retrieval.Context import feature_key               # noqa: PLC0415
    reuse: Tuple[str, ...] = ('level_features',) if feature_key(enc, 256, overlap) in offered else ()
    shared: Dict[str, Any] = dict(encoder=enc, tile_size=256, overlap=overlap, topk_in=topk_in)
    if seg.method == 'Dual':
        return RS.RerankerDualConfig(**shared, order_by=seg.op, reuse=reuse)
    if seg.method == 'CSLS':
        return RS.RerankerCslsConfig(**shared, mode=seg.op, n_ref=20000, k=10, seed=0, chunk=8192, reuse=reuse)
    if seg.method == 'Corr':
        return RS.RerankerCorrConfig(**shared, tau=0.05, iters=20, order_by=seg.op, reuse=reuse)
    if seg.method == 'Shift':
        return RE.RerankerShiftConfig(**shared, stride_px=64, radius=1, reuse=reuse)
    if seg.method == 'XEnc':
        return RE.RerankerXEncConfig(encoder=enc, tile_size=256, topk_in=topk_in)
    if seg.method == 'TileGT':
        return RN.RerankerTileGTConfig(**shared, weights=weights_name(seg.encoder, pooling, 'tilegt'),
                                       shift_loc=False, order_by=seg.op, batch=256, reuse=reuse)
    if seg.method == 'CorrNet':
        return RN.RerankerCorrNetConfig(**shared, weights=weights_name(seg.encoder, pooling, 'corrnet'),
                                        batch=512, reuse=reuse)
    raise ValueError(f'{seg.method} is not a reranker')


def phase_configs(spec: ArmSpec, *, raw_topk: int = 1000, min_sep_tiles: float = 1.0) -> List[Any]:
    """The config of every phase of the arm, in order. A segment's `@K` is what its consumer reads: the next
    phase's `topk_in` (the last segment's `@K` is the report's or stage 3's to use, `arm_topk`)."""
    out: List[Any] = []
    offered: List[Tuple[Any, ...]] = []
    length: int = raw_topk
    for i, seg in enumerate(spec.segments):
        topk_in: int = length
        if i and spec.segments[i - 1].cut:
            topk_in = min(length, spec.segments[i - 1].cut)
        cfg: Any = segment_config(seg, first=i == 0, topk_in=topk_in, raw_topk=raw_topk,
                                  min_sep_tiles=min_sep_tiles, offered=offered)
        out.append(cfg)
        key: Optional[Tuple[Any, ...]] = _offered(cfg)
        if key is not None and key not in offered:
            offered.append(key)
        length = int(getattr(cfg, 'raw_topk', getattr(cfg, 'topk_in', length)))
    return out


def config_of_arm(spec: ArmSpec, *, raw_topk: int = 1000, min_sep_tiles: float = 1.0) -> Any:
    """The `MultiPhasesConfig` of the chain `spec` names (one phase for an arm of one segment): what
    `--stage2 arm:<text>` runs and what the bench files the arm under."""
    from stage2_retrieval.MultiPhases import MultiPhasesConfig     # noqa: PLC0415
    return MultiPhasesConfig(phases=tuple(phase_configs(spec, raw_topk=raw_topk, min_sep_tiles=min_sep_tiles)))


def arm_topk(spec: ArmSpec) -> Optional[int]:
    """The consumer's `topk` of the arm's last list (its `@K`), None for the whole list."""
    return spec.segments[-1].cut or None


def parse_segment(text: str) -> Segment:
    """One reranker on its own (`CSLS::grid2x2+csls`, `gigapath:Dual::cls+dual@50`), for a list of them."""
    tokens: List[str] = [t.strip() for t in text.strip().split(SEP)]
    if len(tokens) != 2:
        raise ArmSyntaxError(f'{text!r}: a reranker is `[encoder:]Method::pooling[+op]`')
    seg: Segment = _segment(tokens[0], tokens[1], '', first=False)
    return seg

