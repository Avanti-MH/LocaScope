.PHONY: help merge-logs clean clean-results clean-logs clean-pycache clean-job clean-smoke list-jobs list-smoke lint

# =============================================================================
#  HOW TO USE THIS MAKEFILE
# =============================================================================
#
#   make help             print this target list
#   make list-jobs        every job name on disk (log side + result side)
#   make list-smoke       the Smoke* leftovers clean-smoke would remove, with sizes (a dry run)
#   make clean-smoke      remove every Smoke* log, result and cache (see SMOKE below)
#   make merge-logs JOB=X merge the per-task logs log/X_<n> of an array job into log/X.merged
#   make clean-job JOB=X  remove one real job's log and result (JOB="A B" for several)
#   make clean-results    the png/jpg/html at the top of result/
#   make clean-logs       every SLURM log at the top of log/
#   make clean-pycache    __pycache__ under the checkout (not result/cache/)
#   make clean            clean-results + clean-logs + clean-pycache
#   make lint             ruff check, read-only
#
# RULES
#   1. Run make on the login node only for these short file operations. Anything that computes
#      goes through Slurm (sbatch / srun), never through a make target.
#   2. Every clean-* target DELETES and none of them asks. Run `make list-smoke` / `make list-jobs`
#      first. Claude never runs a clean-* target: the user does.
#   3. clean-job is for one named job. An empty JOB aborts. It does not touch result/cache/<job>/
#      (the caches are expensive to rebuild: remove one by hand, on purpose).
#   4. Nothing here reaches ./log/ (TODO.log, MILESTONE.log: the tracked decision record).
#
# SMOKE
#   A smoke run is submitted as `sbatch -J Smoke<JobName> ...` (SmokeStage1MppBench,
#   SmokeStage2RetrieverBench, ...). The log (log/%x), the result (result/<job>/) and the cache
#   (result/cache/<job>/) are all keyed on the job name, so a smoke lands in its own Smoke*
#   trees and never mixes with the real run. clean-smoke removes exactly those three places:
#       $(LOG_DIR)/Smoke*   $(RESULT_DIR)/Smoke*   $(RESULT_DIR)/cache/Smoke*
#   plus the older suffix spelling (*Smoke, e.g. Stage2HeavySmoke). A smoke run submitted under the
#   real job name has no such handle: its leftovers sit inside the real trees and make cannot
#   tell them apart.

# Runs write OUTSIDE the checkout. OUTPUT_ROOT is the repo's parent, matching
# utilities/_paths.py -- the one definition on the python side -- and it is
# overridable the same way. Keeping results out of the working tree
# means an `rm -rf` of the checkout, a `git clean`, or a fresh clone cannot
# take 60 GB of feature stores with them, and nothing under result/ can be
# staged by accident.
OUTPUT_ROOT ?= $(if $(LOCASCOPE_OUTPUT_ROOT),$(LOCASCOPE_OUTPUT_ROOT),$(abspath ..))
RESULT_DIR  := $(OUTPUT_ROOT)/result
LOG_DIR     := $(OUTPUT_ROOT)/log

# There are two log directories and they hold different kinds of thing:
#
#   $(LOG_DIR)   SLURM stdout/stderr. Disposable, untracked, outside the repo.
#   ./log/       TODO.log and MILESTONE.log. The decision record, tracked in
#                git, not reproducible by re-running anything, and NEVER a
#                target of any rule here.
#
# The separation used to be a filter -- every sweep spelled `! -name TODO.log
# ! -name MILESTONE.log` because both kinds shared one directory. Now it is
# positional: nothing below names the decision logs, because no rule below can
# reach them.

clean: clean-results clean-logs clean-pycache

# A job leaves artefacts on both sides: SLURM writes stdout/stderr to
# $(LOG_DIR)/<name> and job_result_dir() puts the analysis products in
# $(RESULT_DIR)/<name>/, both keyed on SLURM_JOB_NAME. clean-job removes the
# pair so a rerun starts clean.
#
#   make list-jobs
#   make clean-job JOB=TestTissueMask
#   make clean-job JOB="TestTissueMask RealTest"
#
# An empty JOB aborts rather than expanding to rm -rf $(RESULT_DIR)/.
clean-job:
	@test -n "$(JOB)" || { echo 'usage: make clean-job JOB=<name>   (make list-jobs to see them)'; exit 1; }
	@for j in $(JOB); do \
	  case "$$j" in \
	    .|..|*/*) echo "refusing to clean: $$j"; exit 1 ;; \
	  esac; \
	  rm -f  "$(LOG_DIR)/$$j"; \
	  rm -rf "$(RESULT_DIR)/$$j"; \
	  echo "cleaned $(LOG_DIR)/$$j and $(RESULT_DIR)/$$j/"; \
	done

# Every job name currently on disk, from either side.
list-jobs:
	@{ find "$(LOG_DIR)"    -maxdepth 1 -mindepth 1 -type f -printf '%f\n'; \
	   find "$(RESULT_DIR)" -maxdepth 1 -mindepth 1 -type d -printf '%f\n'; } 2>/dev/null | sort -u

clean-results:
	rm -f "$(RESULT_DIR)"/*.png "$(RESULT_DIR)"/*.jpg "$(RESULT_DIR)"/*.html

# SLURM stdout/stderr is disposable. -maxdepth 1 -type f keeps the sweep off
# any subdirectory, and $(LOG_DIR) is outside the repo, so the decision logs in
# ./log/ are not reachable from here at all.
clean-logs:
	find "$(LOG_DIR)" -maxdepth 1 -type f -delete 2>/dev/null; true

clean-pycache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# ruff.toml holds the rule set (reverse-engineered from the noqa comments
# already in the tree, see that file's own header). Read-only -- use
# `ruff check --fix .` by hand if a fix pass is actually wanted.
lint:
	ruff check .

# ---- smoke ------------------------------------------------------------------
# Smoke* (the rule) and *Smoke (the old spelling) under the three roots that are keyed on the
# job name. rm -rf because a job's log is a file but an array job's, a result's and a cache's are
# directories.
SMOKE_ROOTS := $(LOG_DIR) $(RESULT_DIR) $(RESULT_DIR)/cache

list-smoke:
	for d in $(SMOKE_ROOTS); do \
	  for p in "$$d"/Smoke* "$$d"/*Smoke; do \
	    [ -e "$$p" ] && du -sh "$$p"; \
	  done; \
	done; true

clean-smoke:
	for d in $(SMOKE_ROOTS); do \
	  for p in "$$d"/Smoke* "$$d"/*Smoke; do \
	    [ -e "$$p" ] && rm -rf "$$p" && echo "removed $$p"; \
	  done; \
	done; true

help:
	sed -n '/^#   make help/,/^#   make lint/p' $(firstword $(MAKEFILE_LIST)) | sed 's/^#   //'

# ---- merge the per-task logs of an array job into one ------------------------
# An array job (BenchLocaScope.sh, realtest.sh) writes one log per task, log/<job>_<n>. This
# appends them, in task order, each under a "######## <file>" header, to log/<job>.merged, and
# only if that succeeded removes the pieces. Appending keeps the earlier runs of the job; the
# pieces of a job still running are not finished, so merge after it ends.
#
#   make merge-logs JOB=BenchLocaScope
#   make merge-logs JOB="BenchLocaScope RealTest"
merge-logs:
	@test -n "$(JOB)" || { echo 'usage: make merge-logs JOB=<name>'; exit 1; }
	@for j in $(JOB); do \
	  case "$$j" in .|..|*/*) echo "refusing to merge: $$j"; exit 1 ;; esac; \
	  set -- $$(ls "$(LOG_DIR)/$$j"_[0-9]* 2>/dev/null | sort -V); \
	  [ $$# -gt 0 ] || { echo "no $${j}_<n> logs in $(LOG_DIR)"; continue; }; \
	  out="$(LOG_DIR)/$$j.merged"; \
	  { echo "######## merged $$(date '+%F %T')  $$# task logs"; \
	    for f in "$$@"; do echo "######## $${f##*/}"; cat "$$f"; done; } >> "$$out" \
	    && rm -f "$$@" && echo "merged $$# logs into $$out"; \
	done
