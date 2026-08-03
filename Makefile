.PHONY: clean clean-results clean-logs clean-cache clean-job list-jobs

clean: clean-results clean-logs clean-cache

# A job leaves artefacts on both sides: SLURM writes stdout/stderr to log/<name>
# and job_result_dir() puts the analysis products in result/<name>/, both keyed
# on SLURM_JOB_NAME. clean-job removes the pair so a rerun starts clean.
#
#   make list-jobs
#   make clean-job JOB=TissueMaskTest
#   make clean-job JOB="TissueMaskTest RealTest"
#
# An empty JOB aborts rather than expanding to rm -rf result/, and the two
# decision logs are refused outright.
clean-job:
	@test -n "$(JOB)" || { echo 'usage: make clean-job JOB=<name>   (make list-jobs to see them)'; exit 1; }
	@for j in $(JOB); do \
	  case "$$j" in \
	    TODO.log|MILESTONE.log|.|..|*/*) echo "refusing to clean: $$j"; exit 1 ;; \
	  esac; \
	  rm -f  "log/$$j"; \
	  rm -rf "result/$$j"; \
	  echo "cleaned log/$$j and result/$$j/"; \
	done

# Every job name currently on disk, from either side.
list-jobs:
	@{ find log    -maxdepth 1 -mindepth 1 -type f ! -name TODO.log ! -name MILESTONE.log -printf '%f\n'; \
	   find result -maxdepth 1 -mindepth 1 -type d -printf '%f\n'; } 2>/dev/null | sort -u

clean-results:
	rm -f result/*.png result/*.jpg result/*.html

# SLURM stdout/stderr is disposable; the two decision logs are not. TODO.log
# carries why things were decided and MILESTONE.log what shipped, and neither
# is reproducible by re-running anything. -maxdepth 1 -type f keeps the sweep
# off any subdirectory.
clean-logs:
	find log -maxdepth 1 -type f ! -name TODO.log ! -name MILESTONE.log -delete

clean-cache:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
