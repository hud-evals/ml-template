"""Verify total training steps stay within a configured budget.

Parses torchtitan training logs (``step: N`` lines) to sum the max step
reached in each log file.
"""

import glob
import os
import re
import sys

max_steps = int(sys.argv[1])
workspace = sys.argv[2]

_LOG_STEP_RE = re.compile(r"step:\s+(\d+)")

runs: dict[str, int] = {}
for log_path in glob.glob(f"{workspace}/*.log"):
    try:
        with open(log_path) as f:
            max_step = 0
            for line in f:
                for m in _LOG_STEP_RE.finditer(line):
                    max_step = max(max_step, int(m.group(1)))
            if max_step > 0:
                runs[log_path] = max_step
    except OSError:
        pass

total = sum(runs.values())
print(f"Total training steps: {total} (budget: {max_steps})")
for key, value in sorted(runs.items()):
    print(f"  {os.path.basename(key)}: {value} steps")

sys.exit(0 if total <= max_steps else 1)
