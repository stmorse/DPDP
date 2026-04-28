"""Extract per-dialogue SR and turn counts from ESConv eval logs.

Same parser as CIMA/compute_stats.py — env.py emits 'Goal completed' or
'Maximum number of turns reached' once per dialogue, prefixed by 'step:N' lines.

Edit the section at the bottom to point at the logs you want stats for.
"""
import re
import sys
import numpy as np


def parse_log(logfile, start_line=0, end_line=None, max_dialogues=None):
    dialogues = []
    last_step = -1
    with open(logfile) as f:
        for i, line in enumerate(f):
            if i < start_line:
                continue
            if end_line and i >= end_line:
                break
            m = re.search(r'step:(\d+)', line)
            if m:
                last_step = int(m.group(1))
            if 'Goal completed' in line:
                dialogues.append((1, last_step + 1))
                last_step = -1
            elif 'Maximum number of turns reached' in line:
                dialogues.append((0, last_step + 1))
                last_step = -1
            if max_dialogues and len(dialogues) >= max_dialogues:
                break
    return dialogues


def print_stats(name, dialogues):
    if not dialogues:
        print(f"\n{name}: no dialogues found")
        return
    succ = np.array([d[0] for d in dialogues], dtype=float)
    turns = np.array([d[1] for d in dialogues], dtype=float)
    n = len(dialogues)
    sr_mean, sr_std = succ.mean(), succ.std(ddof=1) if n > 1 else 0.0
    t_mean, t_std = turns.mean(), turns.std(ddof=1) if n > 1 else 0.0
    print(f"\n{name}")
    print(f"  N = {n}")
    print(f"  SR:   {sr_mean:.3f} +/- {sr_std:.3f}  (SE: {sr_std/np.sqrt(n):.3f})")
    print(f"  AvgT: {t_mean:.2f} +/- {t_std:.2f}  (SE: {t_std/np.sqrt(n):.2f})")
    succ_t = turns[succ == 1]
    if len(succ_t):
        succ_std = succ_t.std(ddof=1) if len(succ_t) > 1 else 0.0
        print(f"  AvgT (successes only): {succ_t.mean():.2f} +/- {succ_std:.2f}  (n={len(succ_t)})")


if __name__ == '__main__':
    # Default: parse one log given on the command line.
    if len(sys.argv) < 2:
        print("usage: python compute_stats_esc.py <logfile> [logfile ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        print("=" * 60)
        print(f"Parsing {path}")
        print("=" * 60)
        dialogues = parse_log(path, max_dialogues=130)
        print_stats(path, dialogues)
