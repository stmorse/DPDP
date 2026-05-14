# DPDP Judge Spec — Cross-Model-Stable LLM-as-Judge

A self-contained spec of the LLM-judge mechanism used by the DPDP paper
("Plan Like a Human", ACL 2024) for ESConv, CIMA, and CB. Reproduced from
this codebase so the design can be audited against another project's judge
that is showing instability across LLM backends.

The single most important design choice: the judge is a **forced
multiple-choice classifier**, not a free-form scorer. Every other detail
flows from that.

---

## 1. Overall flow

For one dialogue, end-of-episode reward is computed by:

1. Build a critic prompt that contains the full dialogue plus a fixed
   instruction telling the judge to reply with **exactly one of N
   canonical sentences**.
2. Sample the judge `n=10` times at high temperature (`1.1`).
3. For each sample, extract a class via **substring keyword match** in
   first-match-wins order against a small fixed dict.
4. Map each class to a fixed scalar reward; average across the (≤10)
   samples that matched.
5. Samples that match no keyword are silently dropped, not scored zero
   (more on this below).
6. Compare the mean reward against a fixed threshold to set a binary
   success flag.

---

## 2. Prompt structure (verbatim)

The judge prompt is a 2-message OpenAI-format payload:
```
[
  {"role": "system", "content": <task framing>},
  {"role": "USER",   "content": <constraint + few-shot? + dialogue + question + "Answer: ">}
]
```

Note `"USER"` (uppercase) in the second message — that role string is
non-standard and is preserved as-is across all three datasets. Some
endpoints normalize it; if yours doesn't, it's still passed through to
the model body and acts like an in-prompt label.

The user message ends with the literal string `"Answer: "` to elicit a
sentence-completion-style response.

### 2a. ESConv (emotional support)

System:
> Given a conversation between a Therapist and a Patient, please assess
> whether the Patient' emotional issue has been solved after the
> conversation.

User (template; `%s` placeholders are emotion_type, problem_type, full
dialogue):
> You can only reply with one of the following sentences: No, the Patient
> feels worse. No, the Patient feels the same. No, but the Patient feels
> better. Yes, the Patient's issue has been solved.
>
> The following is a conversation about {emotion_type} regarding
> {problem_type}: {dialogue}
> Question: Has the Patient's issue been solved? Answer:

4 classes; the first three start with "No" and disambiguate via the trailing keyword.

### 2b. CIMA (Italian tutoring)

System:
> Given a conversation between a Teacher and a Student, please assess
> whether the Student correctly translate the English sentence into Italian
> in the conversation.

User (template; sentence is repeated 4 times in the original):
> Please assess whether the Student correctly translated the whole sentence
> of "{sentence}" into Italian in the conversation. You can only reply with
> one of the following sentences: No, the Student made an incorrect
> translation. No, the Student did not try to translate. No, the Student
> only correctly translated a part of "{sentence}". Yes, the Student
> correctly translated the whole sentence of "{sentence}".
>
> The following is the conversation: {dialogue}
> Question: Did the Student correctly translated the whole sentence of
> "{sentence}" into Italian? Answer:

4 classes, same first-three-start-with-"No" pattern.

### 2c. CB (negotiation)

System:
> Given a conversation between a Buyer and a Seller, please decide whether
> the Buyer and the Seller have reached a deal at the end of the
> conversation.

User: 2-class with **2 in-prompt few-shot examples** (one positive, one
negative) before the actual dialogue. Asks the model to extract a deal
price as `[price]` when the deal happened. The verbatim text is in
`ESConv&CB/ppdpp/prompt.py:89-90`.

### 2d. Dialogue serialization

The dialogue is flattened into a single string before insertion:
```python
dial = ''
for turn in conversation:
    dial += '%s: %s ' % (turn['role'], turn['content'])
```
Note the trailing space after each turn and the lack of newlines between
turns. Roles are dataset-specific role strings ("Therapist"/"Patient",
"Teacher"/"Student", "Buyer"/"Seller"), not "user"/"assistant".

---

## 3. Sampling parameters

```
n            = 10        # samples per judge call
temperature  = 1.1       # deliberately high
max_tokens   = 16        # truncates well before any rationalization
```

`max_tokens=16` is critical: it forces the model to commit to one of
the canonical sentences early. Increasing it lets the model add hedges
("…but I think the patient is mostly worse off"), which scrambles the
keyword parser's first-match assumption.

`temperature=1.1` plus `n=10` is the noise-averaging mechanism: each
individual sample is high-variance, but the mean over 10 is stable.
Reducing `n` or temperature changes the calibration.

If the backend is OpenAI-compatible and supports `n`, this is one API
call returning 10 choices. If the backend is real Ollama (no `n`
support), it is looped 10×, which is slow but functionally identical.

---

## 4. Parser

```python
reward_dict = {
  'esc':  {'worse': -1.0, 'same': -0.5, 'better': 0.1, 'solved': 1.0},
  'cima': {'incorrect': -1.0, 'did not': -0.5, 'part': 0.1, 'whole': 0.5},
  'cb':   {'no deal': -1.0, 'deal': 1.0},
}

rewards = []
for output in outputs:                            # outputs is the list of n=10 samples
    for key in reward_dict[dataset]:              # iter in dict order
        if key in output.lower():                 # substring, lower-cased
            rewards.append(reward_dict[dataset][key])
            break                                 # first hit wins
if len(rewards) == 0:
    reward = 0
else:
    reward = sum(rewards) / len(rewards)
```

Five non-obvious behaviors hidden in those few lines:

1. **First-match-wins by dict order.** The reward dict is iterated in its
   declared order. For ESConv, that's `worse → same → better → solved`.
   If a model output contains multiple keywords (e.g. "No, the patient
   feels worse, not better"), `worse` matches first → −1.0. Reordering
   the dict changes scores. CIMA's order has `incorrect → did not → part →
   whole`, which is similarly hazardous (e.g. "the student translated
   incorrectly the whole sentence" matches `incorrect` → −1.0, not
   `whole`).

2. **Substring match, not exact match.** The string `"solved"` substring-
   matches inside `"unsolved"` or `"resolved"`. None of the canonical
   sentences contain those tokens, so this is fine *if* the model stays
   on script. With weaker models that paraphrase ("the issue isn't
   resolved"), substring match misclassifies. Empirically the dataset of
   canonical sentences was chosen so the keywords are non-prefixes of
   each other within each set.

3. **No-match samples are dropped, not scored zero.** A vague critic
   reply like "It's hard to say" matches no keyword and contributes
   nothing to the mean. This is a major calibration knob:
   - With a strong, on-script judge (gpt-4o-mini), nearly all 10 samples
     match → mean is over 10 values.
   - With a weak/chatty judge (llama3.2:3b), several samples may not
     match → mean is over 3-4 values, increasing variance and shifting
     the distribution toward whichever class the model is most willing
     to commit to in plain words.

   If your other project replaces drops with zeros, weak judges look
   systematically negative-biased. If it replaces drops with a default
   class, weak judges inherit that class's bias. This is the most
   likely culprit for cross-model drift.

4. **Mean of class scalars, not weighted by confidence.** All matched
   samples get equal weight. The mean lives on a non-uniform scalar
   scale (-1.0, -0.5, +0.1, +1.0) — the +0.1 vs +1.0 jump for ESConv
   is intentional: "feels better" is a small positive signal, "solved"
   is the actual goal.

5. **Empty output list returns 0.** If every sample fails to match,
   reward defaults to 0 (the +0.1 "better" floor cannot be triggered by
   absence of evidence).

---

## 5. Done-threshold

```python
# ESConv
if reward >= 0.1:
    done = 1            # success
else:
    ...                 # ongoing or max-turns failure
```

For the binary success flag, mean reward ≥ 0.1 over the 10 samples
counts. With the ESConv reward set this means "solved + better" votes
need to outweigh "worse + same" votes by enough to cross 0.1; an
all-`better` outcome (10×0.1 = 1.0; mean = 0.1) is the minimum success.

(The codebase originally had `> 0.1`, which excluded the exact-1.0-mean
all-`better` case as failure — the `>=` is a small fairness fix, but
worth flagging as an example of how the threshold interacts with the
discrete reward set.)

CIMA uses an analogous threshold on its scale; CB has its own deal/
no-deal logic with extracted price.

---

## 6. What the design buys you

The combination of (a) forced canonical-sentence outputs, (b) keyword
substring match in fixed order, and (c) high-temperature 10-sample
averaging is doing the calibration work *implicitly*. It assumes:

- The judge model is willing and able to actually emit one of the
  canonical sentences most of the time.
- The keyword set is unambiguously decodable from those sentences.
- Off-script outputs are noise to be averaged/dropped, not signal.

This breaks if any assumption fails, and the failure is silent. There
are no per-sample rationales to inspect, no log of dropped samples in
the default config, and no per-sample confidence.

---

## 7. Cross-model-drift checklist

When a strong-vs-weak judge gives systematically different scores on
the *same* dialogues, walk this list before adjusting prompts. Ordered
roughly by empirical impact (see §9 for the gpt-4o-mini vs llama3.2
study that drives this ordering):

- **Class distribution over the 10 samples.** Log the per-class counts
  per judge across many dialogues. With a 4-class scheme like ESConv,
  a strong judge may concentrate mass on one class (we saw gpt-4o-mini
  put ~74% on "same") while a weak judge spreads or shifts to another
  (llama3.2 essentially never picks "same"). Threshold and reward-mean
  behavior change drastically even when aggregate SR looks comparable.
- **Drop rate.** Log how many of the 10 samples matched a keyword.
  Compare across judges. A drop-rate gap is a real but secondary tell —
  earlier versions of this doc named it the primary culprit; the
  empirical study in §9 showed class-distribution drift dominates it.
- **Output length.** With `max_tokens=16` the strong judge usually
  finishes one canonical sentence; the weak one may truncate mid-clause
  in a way that changes which keyword appears first. Try `max_tokens=32`
  and recheck.
- **Keyword collision in outputs.** Log full outputs and count how often
  multiple keywords co-occur. First-match-wins is a real source of
  systematic bias toward earlier dict entries when models hedge.
- **Role string handling.** The `"USER"` (uppercase) role is
  non-standard. Some local LLM servers normalize unknown roles to
  `"user"`, others pass them through. If your other project uses a
  different chat template, that affects what the model actually sees.
- **Temperature.** At `1.1`, the variance is intentional. If the other
  project samples at `0.0`–`0.7`, single-sample mode-collapse is much
  more model-dependent.
- **n.** With `n=1`, you're at the mercy of one draw. The DPDP design
  *requires* `n ≥ ~5` for the averaging to do its work.
- **Threshold scale.** A judge-mean of 0.1 means something very
  different on a {-1, -0.5, +0.1, +1.0} scale than on a {0, 1} scale.

---

## 8. Source pointers (this repo)

- Critic prompts (verbatim): `ESConv&CB/ppdpp/prompt.py`
  - ESConv: `:41-47` (also `CIMA/ppdpp/prompt.py` for the CIMA copy)
  - CIMA:   `:61-67`
  - CB:     `:83-90`
- Reward dict: `ESConv&CB/ppdpp/env.py:73-90`
- Sampling params: `ESConv&CB/ppdpp/env.py:368-388` (the `chatgpt`/
  `ollama` branches of the critic call)
- Parser + averaging: `ESConv&CB/ppdpp/env.py:390-403`
- Done-threshold: `ESConv&CB/ppdpp/env.py:160` and `:223`

The CIMA folder has a parallel implementation at `CIMA/ppdpp/{prompt,env}.py`
with the same structure.

---

## 9. Empirical calibration on ESConv (gpt-4o-mini vs llama3.2)

The mechanical spec above was stress-tested by running the verbatim
DPDP critic against the gold human dialogues in the ESConv test split
(n=130) under two LLM backends: `gpt-4o-mini` and `llama3.2:latest`
(3B params) via local ollama. The same dialogue, same prompt, same
sampling params (`n=10`, `temperature=1.1`, `max_tokens=16`), same
parser — only the model identity differed. Results:

### 9a. Headline numbers

- **Aggregate SR overlaps within CI.** gpt-4o-mini 0.569 (±0.044),
  llama3.2 0.485 (±0.044). Looks comparable on the surface.
- **Per-dialogue success Cohen's κ = 0.157** ("slight agreement"). Only
  57.7% of dialogues get the same success label from both judges;
  33 dialogues are only-gpt successes and 22 are only-llama successes.
  The aggregate SR agreement is largely coincidental cancellation.
- **AvgT bias of ~1.6 turns.** On dialogues both judges call
  successful, llama declares success ~1.6 turns earlier than gpt
  because its heavier "better" mass trips the +0.1 threshold sooner.

### 9b. Class distribution drift (the dominant lever)

Pooled across all turns of all dialogues:

| class    | reward | gpt-4o-mini | llama3.2:latest |
|----------|-------:|------------:|----------------:|
| worse    |   −1.0 |        ~13% |            ~34% |
| same     |   −0.5 |        ~74% |             ~2% |
| better   |   +0.1 |        ~10% |            ~60% |
| solved   |   +1.0 |         ~3% |             ~4% |

gpt-4o-mini concentrates mass on the "same" class — the natural
no-progress default for ESConv where patients describe ongoing
distress. llama3.2 essentially never picks "same" and instead
oscillates between "worse" and "better," which is what drives both
the AvgT bias and the per-dialogue disagreement.

### 9c. Three prompt-engineering attempts, all hit a ceiling

We tried prompt rewrites to nudge llama toward gpt's "same"-dominated
distribution:

- **v1** (no template change): verbatim DPDP critic prompt. Best on
  per-dialogue κ and SR among the three; baseline for comparison.
- **v2**: added "default to 'feels the same' unless… choose 'feels
  worse' only if the Patient expresses increased distress…" llama
  over-corrected to "worse"-dominated; SR crashed.
- **v3**: refined v2 with explicit disambiguation — "ongoing distress
  about the original issue is not 'feels worse' — that is the starting
  state, which is 'feels the same.'" llama spread mass roughly
  uniformly across worse/same/better; per-dialogue κ landed at 0.340,
  still below v1.

Per-turn class κ stuck near 0.02 across all three variants. Conclusion:
**at this model-size gap (175B-class vs 3B), prompt engineering cannot
align the judges' class distributions.** Larger open-weights models
may behave differently, but no template-only fix bridges the gap with
llama3.2:3b.

### 9d. Pilot calibration overstates agreement

A pilot run at n=20 dialogues reported per-dialogue κ=0.406 (fair
agreement). The full n=130 collapsed to κ=0.157. Small-sample judge
calibration is dangerously optimistic and should not be the basis
for "the judges agree."

### 9e. What's safe to compare across judges

- **Within-judge method rankings** (e.g. "method X beats raw under
  llama-judge") are valid — both methods are scored under the same
  judge with the same class-distribution bias.
- **Aggregate SR with explicit 95% CI** is the only cross-judge
  headline number that travels. Report both judges side-by-side, not
  pooled, with the κ=0.157 caveat in the methodology section.
- **Per-dialogue success comparisons** (which dialogues count as
  successes) are biased — the κ=0.157 says the judges disagree on
  ~42% of dialogues. Do not say "method X solved dialogue Y" when
  combining judge runs.
- **AvgT** is meaningful within-judge only; cross-judge AvgT gaps
  include a baked-in ~1.6-turn drift from class-distribution
  differences.

---

## 10. Reproducibility tooling (this repo)

To replicate the §9 study against any other LLM backend:

- `ESConv&CB/calibrate_judge_esc.py` — multi-judge calibration walker.
  Accepts repeatable `--judge name=X,url=X,model=X,api_key_env=X,
  loop=X,template=X` flags. For each dialogue, walks every turn,
  builds the critic prompt (optionally overridden via `template=`),
  samples each configured judge `n=10` times, applies the verbatim
  DPDP parser, and reports per-judge drop rates + class
  distributions, pairwise per-turn Pearson r + majority-class κ, and
  per-judge env-replay SR with per-dialogue success κ.
- Per-dialogue log fixtures used in §9:
  `ESConv&CB/human_baseline_esc_test_{oai,llama}_perdialog.log`.
- Template variants tried in §9c:
  `ESConv&CB/judge_template_esc_v{2,3}.txt`.

If your project's judge fails the §7 checklist, this script is the
minimum-overhead way to quantify the gap before adjusting prompts.
