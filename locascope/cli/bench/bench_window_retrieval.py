"""The old name of `bench_stage2_retriever`, kept so a script that still does
`import bench_window_retrieval` (DemoLocaScope/export_window_demo.py) keeps running.
It IS the new module: `sys.modules` hands the importer that object, so every
attribute, private ones included, is the new bench's. Delete this file once nothing
imports the old name."""
import sys

import bench_stage2_retriever

if __name__ == '__main__':
    sys.exit(bench_stage2_retriever.main())
sys.modules[__name__] = bench_stage2_retriever
