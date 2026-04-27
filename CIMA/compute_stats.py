"""Extract per-dialogue SR and turn counts from eval logs, compute mean ± std."""
import re
import numpy as np

def parse_log(logfile, start_line=0, end_line=None):
    """Parse a log (or section), return list of (success: 0/1, turns: int) per dialogue."""
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

    return dialogues

def print_stats(name, dialogues):
    successes = [d[0] for d in dialogues]
    turns = [d[1] for d in dialogues]
    n = len(dialogues)
    if n == 0:
        print(f"\n{name}: no dialogues found")
        return

    sr_mean = np.mean(successes)
    sr_std = np.std(successes, ddof=1)
    sr_se = sr_std / np.sqrt(n)

    turns_mean = np.mean(turns)
    turns_std = np.std(turns, ddof=1)
    turns_se = turns_std / np.sqrt(n)

    print(f"\n{name}")
    print(f"  N = {n}")
    print(f"  SR:   {sr_mean:.3f} +/- {sr_std:.3f}  (SE: {sr_se:.3f})")
    print(f"  AvgT: {turns_mean:.2f} +/- {turns_std:.2f}  (SE: {turns_se:.2f})")

    success_turns = [d[1] for d in dialogues if d[0] == 1]
    if success_turns:
        print(f"  AvgT (success only): {np.mean(success_turns):.2f} +/- {np.std(success_turns, ddof=1):.2f}  (n={len(success_turns)})")

# --- System 2 eval (standalone log) ---
print("=" * 60)
print("DPDP System 2 Eval (MCTS always on, Epoch 2 checkpoint)")
print("=" * 60)
dialogues = parse_log('eval_system2.log')
print_stats("System 2", dialogues)

# --- Stage 2 per-epoch evals (System 1, policy-only) from spt_train.log ---
# Eval sections start at these line numbers:
eval_starts = [104556, 236759, 364350, 479261, 594114]
# Each eval ends at the next training epoch or EOF
eval_ends = eval_starts[1:] + [None]

print("\n" + "=" * 60)
print("Stage 2 Per-Epoch Evals (System 1, policy-only)")
print("=" * 60)
for epoch, (start, end) in enumerate(zip(eval_starts, eval_ends), 1):
    dialogues = parse_log('spt_train.log', start_line=start, end_line=end)
    # Only take first 113 (eval section), not training dialogues from next epoch
    dialogues = dialogues[:113]
    print_stats(f"Epoch {epoch} eval (System 1)", dialogues)

# --- Stage 1 best epoch eval from sft_train.log ---
# The Stage 1 evals are interleaved in sft_train.log.
# Epoch 7 was the best (SR=0.469). Let's find it.
print("\n" + "=" * 60)
print("Stage 1 Evals (System 1, policy-only) from sft_train.log")
print("=" * 60)

import subprocess
result = subprocess.run(['grep', '-n', 'Evaluation Epoch', 'sft_train.log'],
                       capture_output=True, text=True)
if result.stdout:
    lines = result.stdout.strip().split('\n')
    sft_eval_starts = []
    for line in lines:
        linenum = int(line.split(':')[0])
        sft_eval_starts.append(linenum)
    sft_eval_ends = sft_eval_starts[1:] + [None]

    for epoch, (start, end) in enumerate(zip(sft_eval_starts, sft_eval_ends), 1):
        dialogues = parse_log('sft_train.log', start_line=start, end_line=end)
        dialogues = dialogues[:113]
        print_stats(f"Stage 1 Epoch {epoch} eval", dialogues)
