'''Stage 2 -- retrieve candidate windows. A real package: import
its modules fully qualified, `from stage2_retrieval.RetrieverSlidingWinSimRot
import ...`. The directory is not on sys.path; see locascope/paths.py.

Every method is a `Retriever` (`Retriever.py`). `METHODS` is what a
`--stage2 <method>:<recipe>` names; `RETRIEVERS` is the class each config builds
(what `MultiPhases` and the bench construct a phase from); `--stage2 arm:<arm>`
is an arm of the bench's grammar (`ArmSpec.py`), built by `ArmSpec.config_of_arm`.
'''

from typing import Any, Dict, Tuple

from locascope.core.config.ConfigIdentity import method_recipe

#: The stage-2 methods a recipe names: method -> (module, its recipe table, the retriever).
#: `--stage2 slidewin:gigapath`, `--stage2 multiphases:uni2-avg-dual`.
METHODS: Dict[str, Tuple[str, str, str]] = {
    'slidewin': ('locascope.core.stage2_retrieval.RetrieverSlidingWinSimRot', 'SLIDEWIN_RECIPES', 'RetrieverSlidingWinSimRot'),
    'bagoftiles': ('locascope.core.stage2_retrieval.RetrieverBagOfTiles', 'BAGOFTILES_RECIPES', 'RetrieverBagOfTiles'),
    'tilevote': ('locascope.core.stage2_retrieval.RetrieverTileVote', 'TILEVOTE_RECIPES', 'RetrieverTileVote'),
    'clusterbow': ('locascope.core.stage2_retrieval.RetrieverClusterBow', 'CLUSTERBOW_RECIPES', 'RetrieverClusterBow'),
    'winembed': ('locascope.core.stage2_retrieval.RetrieverWinEmbed', 'WINEMBED_RECIPES', 'RetrieverWinEmbed'),
    'multiphases': ('locascope.core.stage2_retrieval.MultiPhases', 'MULTIPHASES_RECIPES', 'MultiPhases'),
}

#: The retriever each config class builds: config class name -> (module, class).
RETRIEVERS: Dict[str, Tuple[str, str]] = {
    'RetrieverSlidingWinSimRotConfig': ('locascope.core.stage2_retrieval.RetrieverSlidingWinSimRot', 'RetrieverSlidingWinSimRot'),
    'RetrieverBagOfTilesConfig': ('locascope.core.stage2_retrieval.RetrieverBagOfTiles', 'RetrieverBagOfTiles'),
    'RetrieverTileVoteConfig': ('locascope.core.stage2_retrieval.RetrieverTileVote', 'RetrieverTileVote'),
    'RetrieverClusterBowConfig': ('locascope.core.stage2_retrieval.RetrieverClusterBow', 'RetrieverClusterBow'),
    'RetrieverWinEmbedConfig': ('locascope.core.stage2_retrieval.RetrieverWinEmbed', 'RetrieverWinEmbed'),
    'RerankerDualConfig': ('locascope.core.stage2_retrieval.RerankerScorers', 'RerankerDual'),
    'RerankerCslsConfig': ('locascope.core.stage2_retrieval.RerankerScorers', 'RerankerCsls'),
    'RerankerCorrConfig': ('locascope.core.stage2_retrieval.RerankerScorers', 'RerankerCorr'),
    'RerankerCandGraphConfig': ('locascope.core.stage2_retrieval.RerankerScorers', 'RerankerCandGraph'),
    'RetrieverGloRetConfig': ('locascope.core.stage2_retrieval.RetrieverGloRet', 'RetrieverGloRet'),
    'RerankerGloRetConfig': ('locascope.core.stage2_retrieval.RetrieverGloRet', 'RetrieverGloRet'),
    'RerankerLogLoRetConfig': ('locascope.core.stage2_retrieval.RetrieverGloRet', 'RerankerLogLoRet'),
    'RerankerShiftConfig': ('locascope.core.stage2_retrieval.RerankerEncode', 'RerankerShift'),
    'RerankerTokenMatchConfig': ('locascope.core.stage2_retrieval.RerankerEncode', 'RerankerTokenMatch'),
    'RerankerXEncConfig': ('locascope.core.stage2_retrieval.RerankerEncode', 'RerankerXEnc'),
    'RerankerTileGTConfig': ('locascope.core.stage2_retrieval.RerankerNeural', 'RerankerTileGT'),
    'RerankerCorrNetConfig': ('locascope.core.stage2_retrieval.RerankerNeural', 'RerankerCorrNet'),
    'RerankerStackConfig': ('locascope.core.stage2_retrieval.RerankerNeural', 'RerankerStack'),
    'MultiPhasesConfig': ('locascope.core.stage2_retrieval.MultiPhases', 'MultiPhases'),
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
        from locascope.core.stage2_retrieval.ArmSpec import config_of_arm, parse_arm   # noqa: PLC0415
        arm = parse_arm(rest)
        cfg = config_of_arm(arm)
        return 'multiphases', arm.text(), cfg, retriever_class(cfg)
    return method_recipe(spec, METHODS)
