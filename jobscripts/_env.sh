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
# first -- and that process is the exception. bench_window_retrieval imports
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

# ---------------------------------------------------------------------------
# wandb: writing outside the checkout, and WANDB_MODE made to work
# ---------------------------------------------------------------------------
# WANDB_MODE IS DELIBERATELY NOT SET HERE. A real run streams, and normal2
# nodes DO reach api.wandb.ai -- `jobscripts/SuperPathPointJobs/
# TrainSuperPathPoint.sh` records that, verified 2026-08-28 after the opposite
# assumption had been written there as a fact and cost one run. Offline is for
# a smoke run, not for the project.
#
# What DID change on 2026-09-16 is that the variable now works at all. Every
# training loop calls `wandb.init(mode=...)` with an EXPLICIT argument, and an
# explicit `mode=` beats WANDB_MODE -- so the literal 'online' those files
# carried made the environment variable look like it was being ignored. Their
# defaults now read it (`SuperPoint/Trainer.py`'s `wandb_mode`,
# `FewShotEoMT/cli/train.py` and `cli/train_superpathpoint.py`'s
# `--wandb-mode`), so
#
#     sbatch --export=ALL,WANDB_MODE=offline ...
#
# is how a smoke run stays off the server, and `wandb sync <run dir>` uploads
# it afterwards if it turns out to be worth keeping.

# wandb writes its run directories to $WANDB_DIR/wandb, and $WANDB_DIR defaults
# to the CURRENT DIRECTORY -- which is the repo, since jobscripts are submitted
# from the checkout. That put 101 MB of run output inside the working tree by
# 2026-09-16, against CLAUDE.md's rule that everything a run produces lives
# under /work/u26130998 where an `rm -rf` of the tree or a `git clean` cannot
# take it and nothing can be staged by accident. Same knob, same reason, as
# HF_HOME above.
export WANDB_DIR="${WANDB_DIR:-${LOCASCOPE_OUTPUT_ROOT:-/work/u26130998}}"
