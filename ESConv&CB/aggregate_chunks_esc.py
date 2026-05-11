"""Aggregate per-chunk eval logs into pooled SR / AvgT / reward.

Usage: python aggregate_chunks_esc.py LOG1:N1 LOG2:N2 ...
  where Ni is the dialogue count of chunk i (must match the --eval_sample_times
  passed when launching that chunk).
"""
import re
import sys
import numpy as np


def parse_log(path):
    with open(path) as f:
        text = f.read()
    m = re.search(r"SR:([\d\.]+),\s*AvgT:([\d\.]+),\s*reward:(-?[\d\.]+)", text)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def main():
    chunks = []
    for arg in sys.argv[1:]:
        path, n = arg.split(":")
        n = int(n)
        parsed = parse_log(path)
        if parsed is None:
            print(f"WARN: could not parse final SR line in {path}")
            continue
        sr, avgt, rw = parsed
        chunks.append((path, n, sr, avgt, rw))

    if not chunks:
        print("no chunks parsed")
        return

    print(f"\n{'chunk':<40} {'N':>4} {'SR':>7} {'AvgT':>7}  reward")
    print("-" * 75)
    for p, n, sr, avgt, rw in chunks:
        print(f"{p:<40} {n:>4} {sr:>7.3f} {avgt:>7.2f}  {rw:>+.3f}")

    total_n = sum(c[1] for c in chunks)
    successes = sum(round(c[2] * c[1]) for c in chunks)
    pooled_sr = successes / total_n
    pooled_avgt = sum(c[3] * c[1] for c in chunks) / total_n
    pooled_rw = sum(c[4] * c[1] for c in chunks) / total_n
    sr_se = np.sqrt(pooled_sr * (1 - pooled_sr) / total_n)

    print("-" * 75)
    print(f"{'POOLED':<40} {total_n:>4} {pooled_sr:>7.3f} {pooled_avgt:>7.2f}  {pooled_rw:>+.3f}")
    print(f"successes: {successes}/{total_n}, SR pooled SE ~{sr_se:.3f}")


if __name__ == "__main__":
    main()
