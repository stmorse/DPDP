# DPDP → CALM Data Port Spec (ESConv)

## Goal

Port saved per-dialogue outputs from the DPDP repo's ESConv eval runs into
the CALM repo, so we can rescore them under the NLI metrics (Metric A / B)
that CALM uses for its MCTS reward. This spec is for an agent operating
**inside `/home/stmorse/projects/DPDP/`** to read its own logs and write
JSONL + a tiny summary into a sibling repo at
`/home/stmorse/projects/calm/data/ESConv/dpdp_ported/`.

No LLM calls are required. Binary per-case success is already in the logs.

## Source files

All under `/home/stmorse/projects/DPDP/ESConv&CB/`.

| method     | stack         | source log(s)                                    |
|------------|---------------|--------------------------------------------------|
| unguided   | llama3.2      | `eval_raw_prompting_esc_llama_c{0,1,2,3}.log`    |
| unguided   | gpt-4o-mini   | `eval_raw_prompting_esc_oai.log` (single, 130)   |
| gdp_zero   | llama3.2      | `eval_gdpzero_esc_llama_c{0,1,2,3}.log`          |
| gdp_zero   | gpt-4o-mini   | `eval_gdpzero_esc_c{0,1,2,3}.log`                |
| dpdp_s1    | llama3.2      | `eval_dpdp_s1_esc_llama_c{0,1,2,3}.log`          |
| dpdp_s1    | gpt-4o-mini   | (please supply path; expected to exist)          |
| dpdp_s1s2  | llama3.2      | `eval_dpdp_esc_llama_e5_c{0,1,2,3}.log`          |
| dpdp_s1s2  | gpt-4o-mini   | `eval_dpdp_esc_e5_c{0,1,2,3}.log`                |

Chunk sizes for the 4-chunk variants: c0=33, c1=33, c2=32, c3=32 (total 130).
Chunk starts: c0=0, c1=33, c2=66, c3=98 (these are
`--eval_start_index` in `run_eval_*_chunked.sh`).

## Target output (write into the calm repo)

```
/home/stmorse/projects/calm/data/ESConv/dpdp_ported/
  unguided_llama3.2.jsonl       summary_unguided_llama3.2.json
  unguided_gpt-4o-mini.jsonl    summary_unguided_gpt-4o-mini.json
  gdp_zero_llama3.2.jsonl       summary_gdp_zero_llama3.2.json
  gdp_zero_gpt-4o-mini.jsonl    summary_gdp_zero_gpt-4o-mini.json
  dpdp_s1_llama3.2.jsonl        summary_dpdp_s1_llama3.2.json
  dpdp_s1_gpt-4o-mini.jsonl     summary_dpdp_s1_gpt-4o-mini.json
  dpdp_s1s2_llama3.2.jsonl      summary_dpdp_s1s2_llama3.2.json
  dpdp_s1s2_gpt-4o-mini.jsonl   summary_dpdp_s1s2_gpt-4o-mini.json
```

You may need to `mkdir -p /home/stmorse/projects/calm/data/ESConv/dpdp_ported`
before writing.

## Per-case parsing algorithm

A log contains a sequence of dialogues, each delimited by

```
================test tuple:<N>====================
```

where `<N>` is the LOCAL 0-based tuple index within that chunk (NOT the
global case index — see "case_id" below).

**Within each tuple block (everything until the next `test tuple:` marker
or EOF):**

1. **Initial seed Patient utterance.** The first log line of the form
   ```
   <ts> - root - [INFO] - [{"role": "Patient", "content": "<...>"}]
   ```
   The `<...>` payload is the help-seeker's situation framing. This is the
   `situation` field. It is a JSON list with one element; parse with
   `json.loads` after stripping the log prefix.

2. **Per-step turn block.** For each occurrence of
   ```
   ---------------step:K-------------
   ```
   the following four lines (in order, ignoring any intervening lines that
   contain `httpx` or `[INFO] - HTTP Request:`) are:
   - The strategy/action name (a plain string, no JSON), e.g.
     `Reflection of feelings`. For `unguided`/`raw_prompting`, this may be
     a no-op token like `Others`; preserve whatever appears.
   - `{"role": "Therapist", "content": "<sys text>"}`
   - `{"role": "Patient",   "content": "<usr text>"}`
   - One of `--> On-going !`, `--> Goal completed !`, or
     `--> Maximum number of turns reached !`

   Each step contributes one (Therapist, Patient) pair to `dialog`.

3. **Outcome.** Determined by the terminal-marker line in the LAST step
   block:
   - `--> Goal completed !` → `dpdp_outcome = "goal_completed"`,
     `success = True`
   - `--> Maximum number of turns reached !` → `dpdp_outcome = "max_turns"`,
     `success = False`
   - Neither (tail-truncated log) → `dpdp_outcome = "incomplete"`,
     `success = None`. Skip writing this case (do not emit a JSONL line).

4. **n_turns** = number of `step:` markers in the block = number of
   (Therapist, Patient) pairs in `dialog`.

## case_id construction

```
global_case_idx = eval_start_index + local_tuple_id
case_id          = f"esc_{global_case_idx:04d}"
```

`eval_start_index` per chunk (4-chunk runs): c0=0, c1=33, c2=66, c3=98.
For the single-file `eval_raw_prompting_esc_oai.log`, `eval_start_index = 0`.

The resulting `case_id` aligns with the line index of
`data/esc-test.txt` (which is byte-identical with
`/home/stmorse/data/ESConv/esc-test.txt` on the calm side).

## Dedup rule

Logs are opened in append mode by `run.py`, so multiple re-runs may produce
duplicate `test tuple:N` blocks in the same file. **Keep the last block
per `(chunk, local_tuple_id)` that has a terminal outcome
(`Goal completed` or `Maximum number of turns reached`)**, discarding any
later-but-incomplete or earlier-but-completed occurrences. This matches
`parse_eval_log.py`'s dedup convention.

If no terminal-outcome block exists for a given `(chunk, tuple_id)`, do
not emit a line for that case.

## JSONL record schema

One line per case, sorted by `case_id` ascending:

```json
{
  "case_id": "esc_0000",
  "situation": "Stress induced by a multitude of issues...",
  "dialog": [
    {"speaker": "sys", "text": "It sounds incredibly painful...", "strategy": "Reflection of feelings"},
    {"speaker": "usr", "text": "Yes, it really is..."},
    {"speaker": "sys", "text": "...", "strategy": "Affirmation and Reassurance"},
    {"speaker": "usr", "text": "..."}
  ],
  "n_turns": 2,
  "dpdp_outcome": "goal_completed",
  "dpdp_success": true,
  "source": {"log": "eval_raw_prompting_esc_oai.log", "tuple_id_local": 0, "chunk": null}
}
```

Field notes:
- `speaker` values are `"sys"` (= Therapist) and `"usr"` (= Patient). This
  matches calm's existing `pair_turns()` convention.
- `strategy` is only on `sys` turns. Drop the field entirely on `usr`
  turns (do not include `"strategy": null`).
- `n_turns` counts (sys, usr) pairs.
- `dpdp_outcome` ∈ `{"goal_completed", "max_turns"}` for emitted lines.
  `incomplete` cases are not emitted.
- `dpdp_success` is the binary success flag (`goal_completed → true`).
  This is the same flag `parse_eval_log.py` lifts.
- `source.chunk` is `"c0"`/`"c1"`/`"c2"`/`"c3"` for chunked runs, or
  `null` for single-file runs.

## summary_<method>_<stack>.json

One JSON file per (method, stack) combo. Schema:

```json
{
  "method": "unguided",
  "stack": "gpt-4o-mini",
  "n_cases_emitted": 130,
  "n_cases_incomplete": 0,
  "dpdp_reported_sr": 0.6231,
  "dpdp_reported_avgt": 4.46,
  "dpdp_reported_reward": -0.512,
  "source_logs": ["eval_raw_prompting_esc_oai.log"],
  "lifted_sr": 0.6231,
  "lifted_avgt": 4.46
}
```

- `dpdp_reported_*` are pulled from the `SR:X, AvgT:Y, reward:Z` tail line
  in each log (same regex as `aggregate_chunks_esc.py:parse_log`). For
  4-chunk runs, pool across the chunks the same way `aggregate_chunks_esc.py`
  does (weighted by chunk N). If a chunk has no parseable final line, omit
  it from the pool and note in a `warnings` array.
- `lifted_sr` = mean of `dpdp_success` over emitted lines.
  `lifted_avgt` = mean of `n_turns` over emitted lines.
- `lifted_*` should agree with `dpdp_reported_*` to within rounding (~1e-3
  on SR, ~0.05 on AvgT). If they don't, add a `warnings` entry explaining
  the gap — this is the receiver's main sanity check on the port.

## Recommended regex pieces (Python)

```python
LOG_LINE   = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:,]+ - root - \[INFO\] - (.*)$")
TUPLE_RE   = re.compile(r"^={16}test tuple:(\d+)={20}$")
STEP_RE    = re.compile(r"^-{15}step:(\d+)-{13}$")
JSON_LINE  = re.compile(r'^\{"role": "(Therapist|Patient)", "content": (.*)\}$')
OUTCOME_RE = re.compile(r"^--> (Goal completed|Maximum number of turns reached|On-going) !$")
SUMMARY_RE = re.compile(r"SR:([\d\.]+),\s*AvgT:([\d\.]+),\s*reward:(-?[\d\.]+)")
```

The `JSON_LINE` content is a JSON-encoded string with all the right
escaping — `json.loads` the captured group 2 (after wrapping in the full
`{"role":..., "content":...}` and re-parsing the whole dict) to get the
unescaped text. Don't try to strip quotes manually; the patient text
contains backslashes and embedded quotes that will trip up naive
splitting.

## Verification you should run before finishing

1. **Case count.** Each emitted JSONL should have 130 lines for fully-
   completed runs. If fewer, the `summary.json` should list the missing
   `case_id`s in a `missing_case_ids` field.
2. **No duplicate `case_id`s** in any JSONL.
3. **`lifted_sr` vs `dpdp_reported_sr` agreement** within 0.005 — see
   above.
4. **Smoke-print first emitted line** of each JSONL to stdout so I can
   eyeball the schema after you finish.
5. **Don't modify anything in `/home/stmorse/projects/DPDP/`** — this is a
   read-only operation from the source side. All writes go to
   `/home/stmorse/projects/calm/data/ESConv/dpdp_ported/`.

## Note on dpdp_s1 / gpt-4o-mini

The calm side believes this combo's logs exist but doesn't know their
path. Please locate them (likely `eval_dpdp_s1_esc_oai_c{0,1,2,3}.log` or
similar) and use them; if they truly don't exist, emit no file for that
combo and add an explicit note in your final stdout summary so the calm
side knows to skip it.
