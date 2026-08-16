.PHONY: clean clean-results clean-logs clean-cache clean-job list-jobs

# Runs write OUTSIDE the checkout. OUTPUT_ROOT is the repo's parent, matching
# utilities/test_modules/_paths.py and query_sim/cli/__init__.py, and it is
# overridable the same way they are. Keeping results out of the working tree
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

clean: clean-results clean-logs clean-cache

# A job leaves artefacts on both sides: SLURM writes stdout/stderr to
# $(LOG_DIR)/<name> and job_result_dir() puts the analysis products in
# $(RESULT_DIR)/<name>/, both keyed on SLURM_JOB_NAME. clean-job removes the
# pair so a rerun starts clean.
#
#   make list-jobs
#   make clean-job JOB=TissueMaskTest
#   make clean-job JOB="TissueMaskTest RealTest"
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

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
