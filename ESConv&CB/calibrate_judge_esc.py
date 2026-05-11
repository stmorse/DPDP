"""Cross-model judge calibration on ESConv.

Walks gold dialogues turn-by-turn, scores each conversation prefix with N
judges (typically gpt-4o-mini and llama3.2 via different OpenAI-compatible
backends), and dumps a JSONL of raw + parsed outputs plus aggregate stats.

Why every-turn (no early-stop): gives the most matched-pair data points per
dialogue, since we want to compare *the judges*, not the simulators.

Provider-flexible: each judge configured via a repeatable --judge flag,
key=value comma-separated. Examples:

  --judge name=gpt,url=https://api.openai.com/v1,model=gpt-4o-mini,api_key_env=OPENAI_API_KEY,loop=false
  --judge name=llama,url=http://localhost:80/v1,model=llama3.2:latest,api_key=ollama,loop=true

`loop=true` issues n separate n=1 calls (for backends like ollama that don't
support OpenAI's n parameter). `loop=false` uses native n=10.

Optional per-judge `template=<path>` overrides the user-message body of the
critic prompt. The file is plain text with `{emotion_type}`, `{problem_type}`,
`{dialogue}` placeholders. The system message and dialogue serialization stay
the codebase default so the override is apples-to-apples with the original
prompt. Omit the field to use the unmodified `ESConvMessages` critic prompt.

Usage:
  python calibrate_judge_esc.py \
    --data_path data/esc-valid.txt \
    --out_jsonl judge_calibration_esc.jsonl \
    --out_summary judge_calibration_esc.summary.txt \
    --judge name=gpt,url=https://api.openai.com/v1,model=gpt-4o-mini,api_key_env=OPENAI_API_KEY,loop=false \
    --judge name=llama,url=http://localhost:80/v1,model=llama3.2:latest,api_key=ollama,loop=true
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from openai import OpenAI

from ppdpp.prompt import ESConvMessages, chatgpt_prompt


REWARD_DICT = {
    'worse': -1.0,
    'same': -0.5,
    'better': 0.1,
    'solved': 1.0,
}
GOAL_THRESHOLD = 0.1
SYSTEM_ROLE = 'Therapist'
USER_ROLE = 'Patient'


class Judge:
    def __init__(self, name, url, model, api_key, loop, template=None):
        self.name = name
        self.url = url
        self.model = model
        self.loop = loop
        self.template = template  # str template body, or None for codebase default
        self.client = OpenAI(base_url=url, api_key=api_key or 'EMPTY')

    def score(self, messages, n, max_tokens, temperature):
        if self.loop:
            outs = []
            for _ in range(n):
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, n=1,
                )
                outs.append(r.choices[0].message.content)
            return outs
        r = self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, n=n,
        )
        return [c.message.content for c in r.choices]


def parse_judge_spec(s):
    parts = {}
    for kv in s.split(','):
        k, _, v = kv.partition('=')
        if k:
            parts[k.strip()] = v.strip()
    required = ('name', 'url', 'model')
    for r in required:
        if r not in parts:
            raise ValueError(f"--judge spec missing '{r}': {s}")
    if 'api_key' in parts:
        api_key = parts['api_key']
    elif 'api_key_env' in parts:
        api_key = os.environ.get(parts['api_key_env'], '')
        if not api_key:
            print(f"warning: env var {parts['api_key_env']} for judge "
                  f"{parts['name']} is empty", file=sys.stderr)
    else:
        api_key = 'EMPTY'
    loop = parts.get('loop', 'false').lower() in ('true', '1', 'yes')
    template = None
    if 'template' in parts and parts['template']:
        with open(parts['template']) as tf:
            template = tf.read()
    return Judge(parts['name'], parts['url'], parts['model'], api_key, loop,
                 template=template)


# Default ESConv critic system message (matches ppdpp/prompt.py:46 verbatim).
DEFAULT_ESC_CRITIC_SYSTEM = (
    "Given a conversation between a Therapist and a Patient, please assess "
    "whether the Patient' emotional issue has been solved after the conversation."
)


def serialize_dialogue(conversation):
    """Match ESConvMessages dialogue serialization verbatim: 'Role: text '
    for each turn, with trailing space."""
    out = ''
    for turn in conversation:
        out += '%s: %s ' % (turn['role'], turn['content'])
    return out


def build_critic_messages(judge, case, conversation):
    """If the judge has a template, render it; else fall back to ESConvMessages.
    Both paths return a chatgpt_prompt-formatted message list ready to send."""
    if judge.template is None:
        m = ESConvMessages(case, 'critic', conversation)
        return chatgpt_prompt(m, USER_ROLE)
    user_body = judge.template.format(
        emotion_type=case.get('emotion_type', ''),
        problem_type=case.get('problem_type', ''),
        dialogue=serialize_dialogue(conversation),
    )
    return [
        {'role': 'system', 'content': DEFAULT_ESC_CRITIC_SYSTEM},
        {'role': 'user', 'content': user_body},
    ]


def parse_one(out):
    out_l = out.lower()
    for key, val in REWARD_DICT.items():
        if key in out_l:
            return key, val
    return None, None


def parse_outputs(outputs):
    classes, vals, drops = [], [], 0
    for o in outputs:
        c, v = parse_one(o)
        if c is None:
            drops += 1
        else:
            classes.append(c)
            vals.append(v)
    mean_reward = float(np.mean(vals)) if vals else 0.0
    return classes, mean_reward, drops


def pair_turns(dialog):
    pairs = []
    sys_buf, usr_buf = [], []
    last_speaker = None

    def flush():
        nonlocal sys_buf, usr_buf
        if sys_buf or usr_buf:
            pairs.append((' '.join(sys_buf).strip(), ' '.join(usr_buf).strip()))
            sys_buf, usr_buf = [], []

    for turn in dialog:
        if turn['speaker'] == 'sys':
            if last_speaker == 'usr':
                flush()
            sys_buf.append(turn['text'])
        else:
            usr_buf.append(turn['text'])
        last_speaker = turn['speaker']
    if sys_buf or usr_buf:
        pairs.append((' '.join(sys_buf).strip(), ' '.join(usr_buf).strip()))

    return [p for p in pairs if p[0] and p[1]]


def env_early_stop_sr(rewards_per_turn):
    """Replay env's success logic on a precomputed sequence of per-turn mean
    rewards. Returns (success, turns)."""
    for t, r in enumerate(rewards_per_turn):
        if r >= GOAL_THRESHOLD:
            return 1, t + 1
    return 0, len(rewards_per_turn)


def cohen_kappa(labels_a, labels_b, classes):
    """Plain Cohen's kappa on aligned label sequences."""
    n = len(labels_a)
    if n == 0:
        return float('nan')
    cls_idx = {c: i for i, c in enumerate(classes)}
    K = len(classes)
    cm = np.zeros((K, K))
    for a, b in zip(labels_a, labels_b):
        cm[cls_idx[a]][cls_idx[b]] += 1
    po = np.trace(cm) / n
    pa = cm.sum(axis=1) / n
    pb = cm.sum(axis=0) / n
    pe = float(np.dot(pa, pb))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else float('nan')
    return (po - pe) / (1 - pe)


def pearson(x, y):
    if len(x) < 2:
        return float('nan')
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data/esc-valid.txt')
    p.add_argument('--max_turn', type=int, default=8)
    p.add_argument('--n_critic_samples', type=int, default=10)
    p.add_argument('--max_tokens', type=int, default=16)
    p.add_argument('--temperature', type=float, default=1.1)
    p.add_argument('--start', type=int, default=0,
                   help='index of first dialogue to score')
    p.add_argument('--limit', type=int, default=-1,
                   help='max number of dialogues to score (-1 = all)')
    p.add_argument('--out_jsonl', required=True)
    p.add_argument('--out_summary', required=True)
    p.add_argument('--judge', action='append', default=[], required=True,
                   help='Repeatable. key=val,... spec for one judge.')
    args = p.parse_args()

    judges = [parse_judge_spec(s) for s in args.judge]
    print(f"judges: {[(j.name, j.model, j.url, 'loop' if j.loop else 'nativeN') for j in judges]}",
          flush=True)
    if len(judges) < 2:
        print("note: only 1 judge configured; agreement stats will be NaN",
              file=sys.stderr)

    with open(args.data_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        cases = cases[args.start : args.start + args.limit]
    else:
        cases = cases[args.start:]
    print(f"data: {args.data_path}  N={len(cases)} (start={args.start})", flush=True)

    # per-(judge, turn) classes for kappa; per-(judge) running counters
    judge_class_counter = {j.name: Counter() for j in judges}
    judge_drops = {j.name: 0 for j in judges}
    judge_total_samples = {j.name: 0 for j in judges}
    # per-turn (judge_a_mean, judge_b_mean) sequences keyed by judge name
    per_turn_means = {j.name: [] for j in judges}
    # per-turn majority class label per judge (None if no matched samples)
    per_turn_majority = {j.name: [] for j in judges}
    # per-dialogue rewards-per-turn per judge, for env-style SR replay
    per_dialogue_rewards = defaultdict(lambda: {j.name: [] for j in judges})

    out_f = open(args.out_jsonl, 'w')
    t0 = time.time()
    for i, case in enumerate(cases):
        case_idx = args.start + i
        pairs = pair_turns(case['dialog'])
        n_turns = min(len(pairs), args.max_turn)
        if n_turns == 0:
            continue

        conversation = [{"role": USER_ROLE, "content": case['situation']}]
        for t_idx in range(n_turns):
            sys_text, usr_text = pairs[t_idx]
            conversation.append({"role": SYSTEM_ROLE, "content": sys_text})
            conversation.append({"role": USER_ROLE, "content": usr_text})

            row = {
                'case': case_idx,
                'emotion_type': case.get('emotion_type'),
                'problem_type': case.get('problem_type'),
                'turn': t_idx,
                'judges': {},
            }
            for j in judges:
                messages = build_critic_messages(j, case, conversation)
                try:
                    outs = j.score(messages,
                                   n=args.n_critic_samples,
                                   max_tokens=args.max_tokens,
                                   temperature=args.temperature)
                except Exception as e:
                    row['judges'][j.name] = {'error': str(e)}
                    print(f"[case {case_idx} turn {t_idx} judge {j.name}] ERROR: {e}",
                          file=sys.stderr, flush=True)
                    continue
                classes, mean_r, drops = parse_outputs(outs)
                row['judges'][j.name] = {
                    'raw': outs,
                    'classes': classes,
                    'drops': drops,
                    'mean_reward': mean_r,
                }
                judge_total_samples[j.name] += args.n_critic_samples
                judge_drops[j.name] += drops
                for c in classes:
                    judge_class_counter[j.name][c] += 1
                per_turn_means[j.name].append(mean_r)
                if classes:
                    maj = Counter(classes).most_common(1)[0][0]
                else:
                    maj = None
                per_turn_majority[j.name].append(maj)
                per_dialogue_rewards[case_idx][j.name].append(mean_r)

            out_f.write(json.dumps(row) + '\n')
            out_f.flush()

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(cases) - i - 1)
        print(f"[{i+1}/{len(cases)}] case {case_idx} turns={n_turns} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    out_f.close()

    # ===== summary =====
    lines = []
    lines.append("=" * 70)
    lines.append("Judge calibration summary")
    lines.append("=" * 70)
    lines.append(f"data_path: {args.data_path}")
    lines.append(f"N dialogues scored: {len(cases)}  (start={args.start})")
    lines.append(f"max_turn={args.max_turn}  n={args.n_critic_samples}  "
                 f"temp={args.temperature}  max_tokens={args.max_tokens}")
    lines.append("")
    for j in judges:
        total = judge_total_samples[j.name]
        drops = judge_drops[j.name]
        drop_rate = drops / total if total else float('nan')
        lines.append(f"[{j.name}] {j.model} @ {j.url}")
        lines.append(f"  total samples : {total}")
        lines.append(f"  drops (no kw) : {drops}  ({drop_rate*100:.1f}%)")
        cc = judge_class_counter[j.name]
        matched = sum(cc.values())
        for cls in REWARD_DICT:
            n_cls = cc.get(cls, 0)
            pct = n_cls / matched * 100 if matched else 0.0
            lines.append(f"    {cls:<7s}: {n_cls:5d}  ({pct:5.1f}% of matched)")
        lines.append("")

    if len(judges) >= 2:
        a, b = judges[0].name, judges[1].name
        lines.append(f"Pairwise comparison: [{a}] vs [{b}]")
        means_a = per_turn_means[a]
        means_b = per_turn_means[b]
        n_pair = min(len(means_a), len(means_b))
        # per-turn mean-reward correlation
        r = pearson(means_a[:n_pair], means_b[:n_pair])
        lines.append(f"  per-turn mean-reward Pearson r : {r:.3f}  (n_turns={n_pair})")
        # per-turn mean-reward absolute delta
        diffs = np.array(means_a[:n_pair]) - np.array(means_b[:n_pair])
        lines.append(f"  per-turn mean-reward delta     : "
                     f"mean={diffs.mean():+.3f}  std={diffs.std():.3f}  "
                     f"|mean|={np.abs(diffs).mean():.3f}")
        # majority-class agreement (kappa over turns where both judges had
        # at least one matched sample)
        maj_a = per_turn_majority[a]
        maj_b = per_turn_majority[b]
        paired = [(x, y) for x, y in zip(maj_a, maj_b) if x is not None and y is not None]
        if paired:
            la, lb = zip(*paired)
            kappa = cohen_kappa(list(la), list(lb), list(REWARD_DICT.keys()))
            agree = sum(1 for x, y in paired if x == y) / len(paired)
            lines.append(f"  majority-class agreement       : "
                         f"{agree*100:.1f}%  (n_turns_both_matched={len(paired)})")
            lines.append(f"  Cohen's kappa (majority class) : {kappa:.3f}")
        else:
            lines.append("  majority-class agreement       : (no paired turns)")
        # env-early-stop SR per judge
        sr_per_judge = {j.name: [] for j in judges}
        avgt_per_judge = {j.name: [] for j in judges}
        for case_idx, by_judge in per_dialogue_rewards.items():
            for jn, rewards in by_judge.items():
                if not rewards:
                    continue
                s, t = env_early_stop_sr(rewards)
                sr_per_judge[jn].append(s)
                avgt_per_judge[jn].append(t)
        lines.append("")
        lines.append(f"Env-style SR (early-stop at mean_reward >= {GOAL_THRESHOLD}):")
        for jn in [a, b]:
            arr = np.array(sr_per_judge[jn], dtype=float)
            tarr = np.array(avgt_per_judge[jn], dtype=float)
            n = len(arr)
            sr = arr.mean() if n else float('nan')
            sr_se = arr.std(ddof=1) / math.sqrt(n) if n > 1 else float('nan')
            avgt = tarr.mean() if n else float('nan')
            lines.append(f"  [{jn}] SR={sr:.3f} (SE {sr_se:.3f})  AvgT={avgt:.2f}  n={n}")
        # per-dialogue success agreement
        sr_a = sr_per_judge[a]
        sr_b = sr_per_judge[b]
        if sr_a and len(sr_a) == len(sr_b):
            same = sum(1 for x, y in zip(sr_a, sr_b) if x == y)
            both1 = sum(1 for x, y in zip(sr_a, sr_b) if x == 1 and y == 1)
            both0 = sum(1 for x, y in zip(sr_a, sr_b) if x == 0 and y == 0)
            a_only = sum(1 for x, y in zip(sr_a, sr_b) if x == 1 and y == 0)
            b_only = sum(1 for x, y in zip(sr_a, sr_b) if x == 0 and y == 1)
            kappa_sr = cohen_kappa(sr_a, sr_b, [0, 1])
            lines.append(f"  per-dialogue success agreement : {same}/{len(sr_a)} "
                         f"({same/len(sr_a)*100:.1f}%)")
            lines.append(f"    both success: {both1}   both fail: {both0}   "
                         f"only-{a}: {a_only}   only-{b}: {b_only}")
            lines.append(f"  Cohen's kappa (success label)  : {kappa_sr:.3f}")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(args.out_summary, 'w') as f:
        f.write(summary + "\n")


if __name__ == '__main__':
    main()
