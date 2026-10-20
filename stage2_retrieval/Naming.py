"""One name for a stage 2 entry, whoever writes it: the pipeline, the bench, the demo.

    {kind}_{method}-{encoder}-{pooling}[-{operation}][-k{K}]-{own hash}[_after-{upstream hash}].{csv|json}

`kind` is the table (`output`, `record`, `tile_sims`, `truth`, `truth_sim`, `rerank_scores`). `method` is the
word the method is filed under (`METHOD_SLUGS`: the pipeline's `slidewin`, and the lower-cased name of every
other method), then the encoder that reads the tiles (left out for a method that reads none: `candgraph`,
`stack`), the pooling (`raw` for none, a `-M` after it for a method that uses the main lattice only), the
operation (the config field `CONFIG_NAMES` names), and `-k{K}` when the list the phase makes or reads is not
the default 1000 long. The text is for people; the hash is the identity (everything the text does not show:
the other settings of the method, the checkpoint it reads, the mask, ...).

The name is made from the CONFIG (`config_label`), so a phase of `--stage2 <method>:<recipe>`, of
`--stage2 arm:<arm>` and of the bench's arm of the same config is one name and one file.

A reranker is an entry of its own. It reads a list a retriever (or another reranker) made, and its name
says whose with `_after-` and the upstream's hash, which is also one of its own hash's parts: the same
reranker after another list is another entry, and the retriever's files are never written by it.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

from stage2_retrieval.ArmSpec import ArmSpec, MAIN_ONLY, RAW, Segment

#: The word each method is filed under.
METHOD_SLUGS: Dict[str, str] = {
    'SlidingWindow': 'slidewin', 'BagOfTiles': 'bagoftiles', 'TileVote': 'tilevote',
    'ClusterBow': 'clusterbow', 'WinEmbed': 'winembed',
    'Dual': 'dual', 'CSLS': 'csls', 'Corr': 'corr', 'CandGraph': 'candgraph', 'Shift': 'shift',
    'TokenMatch': 'tokenmatch', 'XEnc': 'xenc', 'TileGT': 'tilegt', 'CorrNet': 'corrnet', 'Stack': 'stack'}

#: Each config class: the method it is, and the field that holds its operation ('' for none).
CONFIG_NAMES: Dict[str, Tuple[str, str]] = {
    'RetrieverSlidingWinSimRotConfig': ('SlidingWindow', 'score'),
    'RetrieverBagOfTilesConfig': ('BagOfTiles', 'mode'),
    'RetrieverTileVoteConfig': ('TileVote', 'backend'),
    'RetrieverClusterBowConfig': ('ClusterBow', ''),
    'RetrieverWinEmbedConfig': ('WinEmbed', ''),
    'RerankerDualConfig': ('Dual', 'order_by'),
    'RerankerCslsConfig': ('CSLS', 'mode'),
    'RerankerCorrConfig': ('Corr', 'order_by'),
    'RerankerCandGraphConfig': ('CandGraph', 'order_by'),
    'RerankerShiftConfig': ('Shift', ''),
    'RerankerTokenMatchConfig': ('TokenMatch', ''),
    'RerankerXEncConfig': ('XEnc', ''),
    'RerankerTileGTConfig': ('TileGT', 'order_by'),
    'RerankerCorrNetConfig': ('CorrNet', ''),
    'RerankerStackConfig': ('Stack', ''),
}

#: The methods that take no pooling: their name says `raw`.
RAW_METHODS: Tuple[str, ...] = ('CandGraph', 'TokenMatch', 'Stack')

#: The length of a list nothing is said about.
DEFAULT_LENGTH: int = 1000

#: What follows the own hash when the entry reads another entry's list.
AFTER: str = '_after-'


def slug(text: str) -> str:
    """`cls_avg` -> `cls-avg`: letters and digits, one dash between runs of anything else, case kept
    (`raw-M` keeps its capital)."""
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-')


def label(method: str, encoder: str, pooling: str, operation: str = '', cut: int = 0,
          main_only: bool = False) -> str:
    """`{method}[-{encoder}]-{pooling}[-{operation}][-k{cut}]`, without a hash. No encoder for a method that
    reads none."""
    parts: list[str] = [METHOD_SLUGS[method]]
    if encoder:
        parts.append(encoder)
    parts.append(slug(pooling) + (MAIN_ONLY if main_only else ''))
    if operation:
        parts.append(slug(operation))
    if cut:
        parts.append(f'k{cut}')
    return '-'.join(parts)


def config_label(cfg: Any) -> str:
    """The label of a phase's config: what it spells, so the same config is the same file whoever made it.
    A `MultiPhasesConfig` is named by its last phase (each phase is its own entry)."""
    if type(cfg).__name__ == 'MultiPhasesConfig':
        return config_label(cfg.phases[-1])
    name: str = type(cfg).__name__
    if name not in CONFIG_NAMES:
        raise KeyError(f'{name} is not a stage-2 method this names; they are {sorted(CONFIG_NAMES)}')
    method, op_field = CONFIG_NAMES[name]
    enc: Optional[Any] = getattr(cfg, 'encoder', None)
    encoder: str = str(getattr(enc, 'REGISTERED_AS', '') or '') if enc is not None else ''
    pooling: str = RAW if method in RAW_METHODS else (str(enc.pooling) or 'default')
    length: int = int(getattr(cfg, 'raw_topk', getattr(cfg, 'topk_in', DEFAULT_LENGTH)))
    return label(method, '' if method in ('CandGraph', 'Stack') else encoder, pooling,
                 str(getattr(cfg, op_field)) if op_field else '',
                 0 if length == DEFAULT_LENGTH else length, not bool(getattr(cfg, 'overlap', True)))


def arm_label(spec: ArmSpec, raw_topk: int = DEFAULT_LENGTH, min_sep_tiles: float = 1.0) -> str:
    """The label of the LAST phase of the arm: the entry of an arm is its own, never its chain's."""
    from stage2_retrieval.ArmSpec import phase_configs             # noqa: PLC0415
    return config_label(phase_configs(spec, raw_topk=raw_topk, min_sep_tiles=min_sep_tiles)[-1])


def entry_name(label_text: str, own: str, upstream: str = '') -> str:
    """`{label}-{own}` and, for an entry that reads another's list, `_after-{upstream}`."""
    return f'{label_text}-{own}' + (f'{AFTER}{upstream}' if upstream else '')


def hash_of(name: str) -> str:
    """The own hash of an entry name made by `entry_name` (or by `Stage.id`)."""
    return name.split(AFTER, 1)[0].rsplit('-', 1)[1]


def stage_name(method: str, cfg: Any, fallback: str) -> str:
    """The `{name}` of `Stage.id` for a stage-2 config: its label without the method's word (the stage's
    method comes first in `Stage.id`). A config this does not name keeps the recipe's name (`fallback`)."""
    try:
        text: str = config_label(cfg)
    except (KeyError, AttributeError):
        return fallback
    return text.split('-', 1)[1]


__all__ = ['AFTER', 'CONFIG_NAMES', 'DEFAULT_LENGTH', 'METHOD_SLUGS', 'RAW_METHODS', 'arm_label', 'config_label',
           'entry_name', 'hash_of', 'label', 'slug', 'stage_name']
