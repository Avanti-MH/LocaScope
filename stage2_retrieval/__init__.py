'''Stage 2 -- retrieve candidate windows. A real package: import
its modules fully qualified, `from stage2_retrieval.RetrieverSlidingWinSimRot
import ...`. The directory is not on sys.path; see utilities/_paths.py.

Every method is a `Retriever` (`Retriever.py`). `METHODS` is what a
`--stage2 <method>:<recipe>` names; `RETRIEVERS` is the class each config builds
(what `MultiPhases` and the bench construct a phase from); `--stage2 arm:<arm>`
is an arm of the bench's grammar (`ArmSpec.py`), built by `ArmSpec.config_of_arm`.
'''

from typing import Any, Dict, Tuple

from ConfigIdentity import method_recipe

#: The stage-2 methods a recipe names: method -> (module, its recipe table, the retriever).
#: `--stage2 slidewin:gigapath`, `--stage2 multiphases:uni2-avg-dual`.
METHODS: Dict[str, Tuple[str, str, str]] = {
    'slidewin': ('stage2_retrieval.RetrieverSlidingWinSimRot', 'SLIDEWIN_RECIPES', 'RetrieverSlidingWinSimRot'),
    'bagoftiles': ('stage2_retrieval.RetrieverBagOfTiles', 'BAGOFTILES_RECIPES', 'RetrieverBagOfTiles'),
    'tilevote': ('stage2_retrieval.RetrieverTileVote', 'TILEVOTE_RECIPES', 'RetrieverTileVote'),
    'clusterbow': ('stage2_retrieval.RetrieverClusterBow', 'CLUSTERBOW_RECIPES', 'RetrieverClusterBow'),
    'winembed': ('stage2_retrieval.RetrieverWinEmbed', 'WINEMBED_RECIPES', 'RetrieverWinEmbed'),
    'multiphases': ('stage2_retrieval.MultiPhases', 'MULTIPHASES_RECIPES', 'MultiPhases'),
}

#: The retriever each config class builds: config class name -> (module, class).
RETRIEVERS: Dict[str, Tuple[str, str]] = {
    'RetrieverSlidingWinSimRotConfig': ('stage2_retrieval.RetrieverSlidingWinSimRot', 'RetrieverSlidingWinSimRot'),
    'RetrieverBagOfTilesConfig': ('stage2_retrieval.RetrieverBagOfTiles', 'RetrieverBagOfTiles'),
    'RetrieverTileVoteConfig': ('stage2_retrieval.RetrieverTileVote', 'RetrieverTileVote'),
    'RetrieverClusterBowConfig': ('stage2_retrieval.RetrieverClusterBow', 'RetrieverClusterBow'),
    'RetrieverWinEmbedConfig': ('stage2_retrieval.RetrieverWinEmbed', 'RetrieverWinEmbed'),
    'RerankerDualConfig': ('stage2_retrieval.RerankerScorers', 'RerankerDual'),
    'RerankerCslsConfig': ('stage2_retrieval.RerankerScorers', 'RerankerCsls'),
    'RerankerCorrConfig': ('stage2_retrieval.RerankerScorers', 'RerankerCorr'),
    'RerankerCandGraphConfig': ('stage2_retrieval.RerankerScorers', 'RerankerCandGraph'),
    'RerankerShiftConfig': ('stage2_retrieval.RerankerEncode', 'RerankerShift'),
    'RerankerTokenMatchConfig': ('stage2_retrieval.RerankerEncode', 'RerankerTokenMatch'),
    'RerankerXEncConfig': ('stage2_retrieval.RerankerEncode', 'RerankerXEnc'),
    'RerankerTileGTConfig': ('stage2_retrieval.RerankerNeural', 'RerankerTileGT'),
    'RerankerCorrNetConfig': ('stage2_retrieval.RerankerNeural', 'RerankerCorrNet'),
    'RerankerStackConfig': ('stage2_retrieval.RerankerNeural', 'RerankerStack'),
    'MultiPhasesConfig': ('stage2_retrieval.MultiPhases', 'MultiPhases'),
}

#: `--stage2 arm:<arm>`: the arm grammar instead of a recipe name.
ARM: str = 'arm'


def retriever_class(cfg: Any) -> type:
    """The retriever class `cfg` builds."""
    import importlib                                              # noqa: PLC0415
    name: str = type(cfg).__name__
    if name not in RETRIEVERS:
        raise KeyError(f'no stage-2 retriever builds {name}; the configs with one are {sorted(RETRIEVERS)}')
    module, cls = RETRIEVERS[name]
    return getattr(importlib.import_module(module), cls)


def recipe(spec: str):
    """`(method, recipe, config, retriever class)` for `<method>:<recipe>`, or for `arm:<arm>` (the recipe is
    the arm's canonical text)."""
    method, sep, rest = str(spec).partition(':')
    if sep and method == ARM:
        from stage2_retrieval.ArmSpec import config_of_arm, parse_arm   # noqa: PLC0415
        arm = parse_arm(rest)
        cfg = config_of_arm(arm)
        return 'multiphases', arm.text(), cfg, retriever_class(cfg)
    return method_recipe(spec, METHODS)
