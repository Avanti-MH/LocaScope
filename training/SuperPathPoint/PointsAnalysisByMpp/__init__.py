"""Stage B — 存亡分析. spec.md 3.2.

    MppStack        read a co-registered stack ('F' from the store, 'R' derived)
    SurvivalProcess stack + detector -> the [N, L] columns
    SurvivalTable   the store those columns live in
    Patterns        存活向量 -> one of the six 樣態      PURE
    Attribution     兩軸 -> 新生歸因                     PURE

`Patterns` and `Attribution` import nothing but numpy on purpose. They are the
two places in this stage that can be wrong without raising -- a classifier that
calls a flickering vector a band, an attribution branch with no column behind
it -- and the numbers they corrupt are the ones spec.md 3.3 uses to decide
Stage C's head. Pure means they can be tested to exhaustion on hand-written
vectors, with no GPU and no slide.
"""

# NOTHING IS IMPORTED HERE, AND THAT IS THE DECISION, NOT AN OMISSION.
# `from PointsAnalysisByMpp import Patterns` already works -- Python imports the
# submodule -- so an `__init__` that re-exported the five would add nothing
# except a cost: it would make importing `Patterns` also import `MppStack` and
# `SurvivalTable`, and with them safetensors, `PreTileStore` and `TileSampler`.
# The two pure modules would stop being cheap to import, which is the property
# that lets their tests run on a login node in a second with no GPU and no
# store. Re-exporting for tidiness would take that away silently.
