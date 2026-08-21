# Sourced by every jobscript, right after `conda activate`. Not executable and
# not a job: it sets what has to be true BEFORE python starts.
#
# ---------------------------------------------------------------------------
# HF_HOME, and why a Python-side default cannot do this job
# ---------------------------------------------------------------------------
# huggingface_hub reads HF_HOME and HF_HUB_CACHE into module-level constants at
# ITS OWN import (huggingface_hub/constants.py -- os.getenv at import time, not
# at download time). Every value written after that is read by nobody.
#
# Each encoder module therefore does os.environ.setdefault('HF_HOME', ...) above
# its `import timm`, which is correct for the process that imports the encoder
# first -- and that process is the exception. bench_slidewin_pooling imports
# TissueSegFunc, which imports HestSegFunc, which imports transformers, which
# imports huggingface_hub, all before the encoder module is named. The constants
# were frozen to ~/.cache/huggingface, and 2.6 GB of UNI2 was re-downloaded on
# 2026-08-21 while a complete copy sat in /work.
#
# Exporting here dissolves the whole race: the value is in the environment
# before the interpreter starts, so every setdefault sees it already set and
# leaves it alone, whichever module wins the import. That is also why the
# modules use setdefault and not `=` -- this line has to be able to win.
#
# One directory for all encoders, not one per model. Blobs are addressed by
# hash under $HF_HOME/hub, so sharing costs nothing and splitting bought
# nothing: the split was three chances to freeze the wrong one.
#
# It lives in /work rather than $HOME because $HOME is a small quota and the
# weights are ~7 GB before CONCH. LOCASCOPE_OUTPUT_ROOT is honoured for the same
# reason _paths.py honours it -- one knob moves everything a run touches.
export HF_HOME="${HF_HOME:-${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}/model_weights}"

# Offline runs are NOT set here. A missing weight file would then fail with a
# connection error rather than downloading, and the first run of a new encoder
# is exactly when that would bite. Set HF_HUB_OFFLINE=1 in the environment when
# you want the network refusal.
