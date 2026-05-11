"""Parse a run.py eval log into per-dialogue (success, turns) tuples and report
SR +/- SE and AvgT +/- SD. Optionally accepts multiple chunked logs and pools.

Per-dialogue parsing:
  - "test tuple:N"  marks dialogue boundary
  - Within a block, count "---------------step:K-------------" lines  -> turns = K+1
  - "--> Goal completed !"      -> success = 1
  - "--> Maximum number of turns reached !" -> success = 0
"""
import re
import sys
import numpy as np


STEP_RE = re.compile(r"---------------step:(\d+)-------------")
TUPLE_RE = re.compile(r"================test tuple:(\d+)====================")


def parse_log(path):
    """Parse log, deduping by tuple_id (keep last completed occurrence per id).
    Logs may contain multiple appended runs (run.py opens log in append mode);
    we want the final run's per-dialogue results.
    """
    with open(path) as f:
        text = f.read()
    parts = TUPLE_RE.split(text)
    by_tid = {}
    for i in range(1, len(parts), 2):
        tid = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        steps = STEP_RE.findall(body)
        n_steps = len(steps)
        if "--> Goal completed !" in body:
            by_tid[tid] = (tid, 1, n_steps)
        elif "--> Maximum number of turns reached !" in body:
            by_tid[tid] = (tid, 0, n_steps)
        # else: incomplete -- don't overwrite a prior completion if any
    return [by_tid[tid] for tid in sorted(by_tid)]


def stats(results, label=""):
    sr = np.array([r[1] for r in results], dtype=float)
    avgt = np.array([r[2] for r in results], dtype=float)
    n = len(results)
    sr_mean = sr.mean()
    sr_se = np.sqrt(sr_mean * (1 - sr_mean) / n) if n > 0 else 0.0
    avgt_mean = avgt.mean()
    avgt_sd = avgt.std(ddof=1) if n > 1 else 0.0
    avgt_se = avgt_sd / np.sqrt(n) if n > 0 else 0.0

    print(f"{label:30s} N={n:3d}  SR={sr_mean:.3f} +/- {sr_se:.3f} (SE)   "
          f"AvgT={avgt_mean:.2f} +/- {avgt_sd:.2f} (SD)  SE={avgt_se:.2f}")

    succ = avgt[sr == 1]
    if len(succ) > 1:
        print(f"{'  successes only':30s} n={len(succ):3d}  AvgT={succ.mean():.2f} "
              f"+/- {succ.std(ddof=1):.2f} (SD)")
    return sr_mean, avgt_mean, n


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Usage: python parse_eval_log.py LOG1 [LOG2 ...]")
        sys.exit(1)

    all_results = []
    for path in sys.argv[1:]:
        try:
            results = parse_log(path)
        except FileNotFoundError:
            print(f"WARN: {path} not found")
            continue
        stats(results, label=path.split("/")[-1])
        all_results.extend(results)

    if len(sys.argv) > 2:
        print()
        stats(all_results, label="POOLED")


if __name__ == "__main__":
    main()
