"""What a run emits, and the definitions the emissions are printed against.

Figures, tables, summary files -- and, deliberately in the same place, the
definitions of the metrics and formats those tables carry. `RetrievalReport`
is the reason the two live together: it is not a printer with metrics attached
but the ONE definition of what "@1%" means, which two benches at two different
scales both quote. A format that is defined twice is a format that eventually
disagrees with itself, and the disagreement shows up as two plausible-looking
numbers under the same column heading.

A package rather than another entry in `_paths.setup_import_paths()`, because
none of the seven importers calls that function -- every one of them inserts
`utilities/` into sys.path by hand -- so a path entry would have cost the same
seven edits and bought nothing. `from dump_function.X import ...` says where
the code lives at the call site instead.

`__init__.py` exists rather than leaning on implicit namespace packages: a
namespace package MERGES every directory of that name found anywhere on
sys.path, and this repo puts six directories on it per entry point.

What does NOT belong here
-------------------------
Anything a stage module imports, because `<N>_<stage>/` is pure logic by the
repo's own rule and may not pull matplotlib in through the side door. The
homography code is split along exactly that line: `query_quad` draws, so it
lives in `_sift_plot` here, while `is_invertible` decides whether an H is
usable at all and therefore lives in `3_localization/SIFT_RANSAC.py`, where
`SIFT_RANSAC` and `SlideWinSift` can both reach it.
"""
