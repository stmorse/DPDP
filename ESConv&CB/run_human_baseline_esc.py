"""Human baseline for ESConv: replay each gold validation dialogue through the
same critic the env uses, measure SR (goal completed within max_turn) and AvgT.

Mirrors ppdpp/env.py's compute_reward and step() goal logic exactly:
- Build the critic prompt via ESConvMessages(role='critic')
- Sample n=10 outputs at temperature 1.1
- Parse keyword rewards: worse(-1), same(-0.5), better(0.1), solved(1.0)
- Average; goal completed if avg > 0.1
- Replays at most max_turn=8 sys-usr exchanges per dialogue.

Backend selection via env vars (matches the env-based eval scripts):
- LLM_BASE_URL  default http://localhost:11434/v1 (ollama). Set to
                https://api.openai.com/v1 for OpenAI.
- LLM_MODEL     default llama3.2:latest. Set to gpt-4o-mini etc. for OpenAI.
- OPENAI_API_KEY required when LLM_BASE_URL points at api.openai.com.

CLI flags --ollama_url / --ollama_model still work and override the env vars.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from openai import OpenAI

from ppdpp.prompt import ESConvMessages, chatgpt_prompt


REWARD_DICT = {
    'worse': -1.0,
    'same': -0.5,
    'better': 0.1,
    'solved': 1.0,
}
GOAL_THRESHOLD = 0.1  # env.py: reward > 0.1
SYSTEM_ROLE = 'Therapist'
USER_ROLE = 'Patient'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path', default='data/esc-valid.txt')
    p.add_argument('--max_turn', type=int, default=8)
    p.add_argument('--ollama_url', default=os.environ.get('LLM_BASE_URL', 'http://localhost:11434/v1'))
    p.add_argument('--ollama_model', default=os.environ.get('LLM_MODEL', 'llama3.2:latest'))
    p.add_argument('--n_critic_samples', type=int, default=10)
    p.add_argument('--max_tokens', type=int, default=16)
    p.add_argument('--temperature', type=float, default=1.1)
    p.add_argument('--out_log', default='human_baseline_esc.log')
    return p.parse_args()


def query_critic(client, model, messages, n, max_tokens, temperature, looks_like_ollama):
    """Ollama doesn't support n>1, so loop. Real OpenAI: use native n."""
    if looks_like_ollama:
        outputs = []
        for _ in range(n):
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, n=1,
            )
            outputs.append(resp.choices[0].message.content)
        return outputs
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, n=n,
    )
    return [c.message.content for c in resp.choices]


def parse_reward(outputs):
    rewards = []
    counts = defaultdict(int)
    for out in outputs:
        for key, val in REWARD_DICT.items():
            if key in out.lower():
                rewards.append(val)
                counts[key] += 1
                break
    avg = sum(rewards) / len(rewards) if rewards else 0.0
    return avg, counts


def pair_turns(dialog):
    """Group consecutive sys turns into one (concatenated) and consecutive usr
    turns into one. Returns alternating (sys_text, usr_text) pairs."""
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

    pairs = [p for p in pairs if p[0] and p[1]]
    return pairs


def replay(client, args, case, log, looks_like_ollama):
    pairs = pair_turns(case['dialog'])
    conversation = [{"role": USER_ROLE, "content": case['situation']}]

    for t, (sys_text, usr_text) in enumerate(pairs[: args.max_turn]):
        conversation.append({"role": SYSTEM_ROLE, "content": sys_text})
        conversation.append({"role": USER_ROLE, "content": usr_text})

        messages = ESConvMessages(case, 'critic', conversation)
        messages = chatgpt_prompt(messages, USER_ROLE)
        outputs = query_critic(
            client, args.ollama_model, messages,
            n=args.n_critic_samples,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            looks_like_ollama=looks_like_ollama,
        )
        reward, counts = parse_reward(outputs)
        log.write(f"  turn {t}: reward={reward:.3f}  counts={dict(counts)}\n")

        if reward >= GOAL_THRESHOLD:
            return 1, t + 1
    return 0, min(len(pairs), args.max_turn)


def main():
    args = parse_args()
    api_key = os.environ.get('OPENAI_API_KEY', 'ollama')
    client = OpenAI(base_url=args.ollama_url, api_key=api_key)
    looks_like_ollama = any(s in args.ollama_url for s in ('localhost', '11434', 'ollama'))
    print(f"backend: base_url={args.ollama_url}  model={args.ollama_model}  ollama_loop={looks_like_ollama}", flush=True)

    with open(args.data_path) as f:
        cases = [json.loads(line) for line in f if line.strip()]

    results = []
    with open(args.out_log, 'w') as log:
        for i, case in enumerate(cases):
            log.write(f"\n=== case {i} ({case.get('emotion_type')}/{case.get('problem_type')}) ===\n")
            log.flush()
            try:
                success, turns = replay(client, args, case, log, looks_like_ollama)
            except Exception as e:
                log.write(f"  ERROR: {e}\n")
                continue
            log.write(f"  result: success={success}  turns={turns}\n")
            log.flush()
            results.append((success, turns))
            print(f"[{i+1}/{len(cases)}] success={success} turns={turns}", flush=True)

    if not results:
        print("No results to summarize.")
        return

    sr = np.array([r[0] for r in results], dtype=float)
    avgt = np.array([r[1] for r in results], dtype=float)
    n = len(results)
    sr_mean, sr_std = sr.mean(), sr.std(ddof=1) if n > 1 else 0.0
    avgt_mean, avgt_std = avgt.mean(), avgt.std(ddof=1) if n > 1 else 0.0

    print("\n" + "=" * 60)
    print("Human Baseline (ESConv valid, replayed through ollama critic)")
    print("=" * 60)
    print(f"  N    = {n}")
    print(f"  SR   = {sr_mean:.3f} +/- {sr_std:.3f}  (SE: {sr_std/np.sqrt(n):.3f})")
    print(f"  AvgT = {avgt_mean:.2f} +/- {avgt_std:.2f}  (SE: {avgt_std/np.sqrt(n):.2f})")
    succ_t = avgt[sr == 1]
    if len(succ_t):
        print(f"  AvgT (successes only): {succ_t.mean():.2f} +/- {succ_t.std(ddof=1) if len(succ_t)>1 else 0:.2f}  (n={len(succ_t)})")


if __name__ == '__main__':
    main()
