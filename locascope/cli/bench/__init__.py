"""The benches: scripts that RUN something and write what it computed to the cache.

`bench_stage1_mpp` and `bench_stage2_retriever` write the stage 1 and stage 2 entries of a set of
FoVs; `bench_locascope` runs the whole pipeline. They print the run's progress and its gates and
nothing more -- every metric, CSV and table is read off the entries afterwards by the scripts of
`locascope/cli/metrics/` (`analyze_stage1_metrics`, `analyze_stage2_metrics`), which take the bench's
own flags and compute the same addresses, so a table is redrawn or regrouped without a GPU.

`BenchCommon` is what the benches share (the slides, the FoV supply, the definition of an answer and
its ranks); `ArmTables` and `RetrieverArms` are the stage 2 bench's entries and arm engine.
"""
