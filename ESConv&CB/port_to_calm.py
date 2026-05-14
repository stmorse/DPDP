"""Port DPDP ESConv eval logs into JSONL + summary JSON for the CALM repo.

Reads per-dialogue outputs from eval logs under ESConv&CB/ and emits one JSONL
file per (method, stack) into /home/stmorse/projects/calm/data/ESConv/dpdp_ported/.
No LLM calls; binary per-case success is in the logs.

Dedup, outcome detection: mirrors parse_eval_log.py's algorithm
(keep last block per (chunk, tuple_id) with a terminal outcome).

Strategy extraction (standardized across all combos):
  1. First non-httpx content line after step:K. If it matches one of 8 ESConv
     system acts, use it.
  2. Else look back from step:K for the most recent
     `Choose action "X" by (Policy Network|MCTS)...` line and extract X.
  3. Else null + warning.

Spec: /home/stmorse/projects/DPDP/DATA_PORT_SPEC.md
"""
import json
import os
import re
import sys
from collections import OrderedDict

REPO_ROOT = "/home/stmorse/projects/DPDP/ESConv&CB"
OUT_DIR   = "/home/stmorse/projects/calm/data/ESConv/dpdp_ported"

SYSTEM_ACTS = {
    "Affirmation and Reassurance",
    "Information",
    "Others",
    "Providing Suggestions",
    "Question",
    "Reflection of feelings",
    "Restatement or Paraphrasing",
    "Self-disclosure",
}

LOG_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} [\d:,]+ - [\w\.]+ - \[\w+\] - ")
TUPLE_RE      = re.compile(r"^={16}test tuple:(\d+)={20}$")
STEP_RE       = re.compile(r"^-{15}step:(\d+)-{13}$")
OUTCOME_RE    = re.compile(r"^--> (Goal completed|Maximum number of turns reached|On-going) !$")
CHOOSE_RE     = re.compile(r'^Choose action "([^"]+)" by (Policy Network|MCTS)')
CHOOSE_PATH_RE = re.compile(r'^Choose action "([^"]+)" from searched successful path')
SEED_RE       = re.compile(r'^\[\{"role": "Patient", "content":')
JSON_TURN_RE  = re.compile(r'^\{"role": "(Therapist|Patient)", "content":')
SUMMARY_RE    = re.compile(r"SR:([\d\.]+),\s*AvgT:([\d\.]+),\s*reward:(-?[\d\.]+)")
HTTPX_RE      = re.compile(r"HTTP Request:|httpx")


CHUNK_STARTS = {"c0": 0, "c1": 33, "c2": 66, "c3": 98}
CHUNK_SIZES  = {"c0": 33, "c1": 33, "c2": 32, "c3": 32}


def strip_prefix(line):
    """Strip the `YYYY-MM-DD HH:MM:SS,SSS - <logger> - [LEVEL] - ` prefix.
    Returns None if line doesn't have that prefix or contains httpx noise."""
    m = LOG_PREFIX_RE.match(line)
    if not m:
        return None
    content = line[m.end():].rstrip("\n")
    if HTTPX_RE.search(line):
        return None
    return content


def parse_log(path, chunk_tag):
    """Parse one log file. Returns list of case dicts (one per dialogue with
    a terminal outcome), and a list of (tuple_id, reason) for skipped ones.

    Dedup: for each local tuple_id, keep the last block in the file that has
    a terminal outcome ("Goal completed" or "Maximum number of turns reached").
    Append-mode reruns of the same chunk leave duplicate blocks — we want the
    latest completed run.
    """
    with open(path) as f:
        raw_lines = f.read().split("\n")

    # First pass: strip prefixes (None for non-prefixed or httpx lines).
    content_lines = [strip_prefix(ln) for ln in raw_lines]

    # Find tuple-marker indices in raw_lines.
    tuple_markers = []
    for i, ln in enumerate(raw_lines):
        # The tuple marker itself is logged with the log prefix on the LINE BEFORE
        # the actual ====...===== text. Looking at the logs: the marker appears
        # as its own line (no log prefix) of the form
        #     ================test tuple:N====================
        if TUPLE_RE.match(ln.rstrip()):
            tuple_markers.append(i)

    # Build per-tuple blocks: each block is the slice of lines from this marker
    # up to (but not including) the next marker.
    by_tid = OrderedDict()  # tid -> dict (last completed only)
    incomplete = []         # list of (tid, reason) for tuples seen but skipped
    for idx, start in enumerate(tuple_markers):
        m = TUPLE_RE.match(raw_lines[start].rstrip())
        tid = int(m.group(1))
        end = tuple_markers[idx + 1] if idx + 1 < len(tuple_markers) else len(raw_lines)
        block_raw = raw_lines[start + 1 : end]
        block_content = content_lines[start + 1 : end]

        case = build_case(tid, block_raw, block_content, chunk_tag, path)
        if case is None:
            incomplete.append((tid, "incomplete (no terminal outcome)"))
            continue
        by_tid[tid] = case

    return list(by_tid.values()), incomplete


def build_case(tid, block_raw, block_content, chunk_tag, source_log_path):
    """Build a case dict from one tuple block. Returns None if the dialogue
    didn't reach a terminal outcome."""
    # First seed line: situation (a 1-elt JSON list).
    situation = None
    for c in block_content:
        if c is None:
            continue
        if SEED_RE.match(c):
            try:
                payload = json.loads(c)
                if isinstance(payload, list) and payload and payload[0].get("role") == "Patient":
                    situation = payload[0]["content"]
                    break
            except json.JSONDecodeError:
                pass

    step_idxs = []
    for i, c in enumerate(block_content):
        if c is None:
            continue
        if STEP_RE.match(c.rstrip()):
            step_idxs.append(i)

    if not step_idxs:
        return None

    dialog = []
    # For each step: extract strategy, therapist text, patient text, outcome.
    terminal_outcome = None
    for si, step_start in enumerate(step_idxs):
        step_end = step_idxs[si + 1] if si + 1 < len(step_idxs) else len(block_raw)
        step_content = block_content[step_start + 1 : step_end]

        strategy = extract_strategy(
            step_content,
            backstop_content=block_content[: step_start],
        )

        therapist_text = None
        patient_text = None
        outcome = None
        for c in step_content:
            if c is None:
                continue
            if therapist_text is None and JSON_TURN_RE.match(c):
                try:
                    d = json.loads(c)
                    if d.get("role") == "Therapist":
                        therapist_text = d["content"]
                        continue
                except json.JSONDecodeError:
                    pass
            if patient_text is None and JSON_TURN_RE.match(c):
                try:
                    d = json.loads(c)
                    if d.get("role") == "Patient":
                        patient_text = d["content"]
                        continue
                except json.JSONDecodeError:
                    pass
            om = OUTCOME_RE.match(c)
            if om:
                outcome = om.group(1)
                # Don't break — we want the LAST outcome in case there are
                # stray markers; but in practice there's exactly one.

        if therapist_text is None or patient_text is None:
            # Malformed step. Skip the whole dialogue rather than emit a
            # half-built one.
            return None

        sys_turn = {"speaker": "sys", "text": therapist_text}
        if strategy is not None:
            sys_turn["strategy"] = strategy
        dialog.append(sys_turn)
        dialog.append({"speaker": "usr", "text": patient_text})

        if si == len(step_idxs) - 1:
            terminal_outcome = outcome

    if terminal_outcome == "Goal completed":
        dpdp_outcome = "goal_completed"
        success = True
    elif terminal_outcome == "Maximum number of turns reached":
        dpdp_outcome = "max_turns"
        success = False
    else:
        return None  # On-going (truncated) or missing outcome

    case_id = f"esc_{tid:04d}"

    return {
        "case_id": case_id,
        "situation": situation,
        "dialog": dialog,
        "n_turns": len(step_idxs),
        "dpdp_outcome": dpdp_outcome,
        "dpdp_success": success,
        "source": {
            "log": os.path.basename(source_log_path),
            "tuple_id": tid,
            "chunk": chunk_tag if chunk_tag in CHUNK_STARTS else None,
        },
    }


def extract_strategy(step_content, backstop_content):
    """Standardized strategy extraction.

    Primary: first non-httpx content line in this step that matches an act.
    Fallback: most recent `Choose action "X"` line in `backstop_content`.
    """
    # Primary.
    for c in step_content:
        if c is None:
            continue
        c_stripped = c.strip()
        if c_stripped in SYSTEM_ACTS:
            return c_stripped
        if JSON_TURN_RE.match(c_stripped):
            # Reached the Therapist line without seeing a strategy.
            break

    # Fallback: walk backstop in reverse for "Choose action ..." line.
    for c in reversed(backstop_content):
        if c is None:
            continue
        m = CHOOSE_RE.match(c) or CHOOSE_PATH_RE.match(c)
        if m:
            act = m.group(1)
            if act in SYSTEM_ACTS:
                return act

    return None


def parse_reported(path):
    """Extract SR/AvgT/reward from the last summary line in a log. Returns
    (sr, avgt, reward) or None if not found."""
    try:
        with open(path) as f:
            text = f.read()
    except FileNotFoundError:
        return None
    m = None
    for hit in SUMMARY_RE.finditer(text):
        m = hit
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def pool_chunks_reported(chunk_paths):
    """N-weighted pooling of per-chunk SR/AvgT/reward, matching
    aggregate_chunks_esc.py's behavior. Returns (sr, avgt, reward, warnings)."""
    rows = []
    warnings = []
    for tag, path in chunk_paths.items():
        n = CHUNK_SIZES[tag]
        parsed = parse_reported(path)
        if parsed is None:
            warnings.append(f"could not parse summary line in {os.path.basename(path)}")
            continue
        sr, avgt, rw = parsed
        rows.append((tag, n, sr, avgt, rw))

    if not rows:
        return None, None, None, warnings

    total_n = sum(r[1] for r in rows)
    successes = sum(round(r[2] * r[1]) for r in rows)
    pooled_sr = successes / total_n
    pooled_avgt = sum(r[3] * r[1] for r in rows) / total_n
    pooled_rw = sum(r[4] * r[1] for r in rows) / total_n
    return pooled_sr, pooled_avgt, pooled_rw, warnings


# (method, stack) -> {chunk_tag_or_None: log_path_basename}
COMBOS = {
    ("unguided", "llama3.2"): {
        "c0": "eval_raw_prompting_esc_llama_v2_c0.log",
        "c1": "eval_raw_prompting_esc_llama_v2_c1.log",
        "c2": "eval_raw_prompting_esc_llama_v2_c2.log",
        "c3": "eval_raw_prompting_esc_llama_v2_c3.log",
    },
    ("unguided", "gpt-4o-mini"): {
        # Unchanged from v1: this combo was a single-file run with
        # eval_start_index=0, so the env.test_num bug didn't affect it.
        None: "eval_raw_prompting_esc_oai.log",
    },
    ("gdp_zero", "llama3.2"): {
        "c0": "eval_gdpzero_esc_llama_v2_c0.log",
        "c1": "eval_gdpzero_esc_llama_v2_c1.log",
        "c2": "eval_gdpzero_esc_llama_v2_c2.log",
        "c3": "eval_gdpzero_esc_llama_v2_c3.log",
    },
    ("gdp_zero", "gpt-4o-mini"): {
        "c0": "eval_gdpzero_esc_v2_c0.log",
        "c1": "eval_gdpzero_esc_v2_c1.log",
        "c2": "eval_gdpzero_esc_v2_c2.log",
        "c3": "eval_gdpzero_esc_v2_c3.log",
    },
    ("dpdp_s1", "llama3.2"): {
        "c0": "eval_dpdp_s1_esc_llama_v2_c0.log",
        "c1": "eval_dpdp_s1_esc_llama_v2_c1.log",
        "c2": "eval_dpdp_s1_esc_llama_v2_c2.log",
        "c3": "eval_dpdp_s1_esc_llama_v2_c3.log",
    },
    ("dpdp_s1", "gpt-4o-mini"): {
        # NEW in this pass: standalone v2 eval of the Stage 1 best_checkpoint
        # against gpt-4o-mini. Confirms the 0.846 from Stage 1 SFT per-epoch.
        "c0": "eval_dpdp_s1_esc_v2_c0.log",
        "c1": "eval_dpdp_s1_esc_v2_c1.log",
        "c2": "eval_dpdp_s1_esc_v2_c2.log",
        "c3": "eval_dpdp_s1_esc_v2_c3.log",
    },
    ("dpdp_s1s2", "llama3.2"): {
        "c0": "eval_dpdp_esc_llama_v2_e5_c0.log",
        "c1": "eval_dpdp_esc_llama_v2_e5_c1.log",
        "c2": "eval_dpdp_esc_llama_v2_e5_c2.log",
        "c3": "eval_dpdp_esc_llama_v2_e5_c3.log",
    },
    ("dpdp_s1s2", "gpt-4o-mini"): {
        "c0": "eval_dpdp_esc_v2_e5_c0.log",
        "c1": "eval_dpdp_esc_v2_e5_c1.log",
        "c2": "eval_dpdp_esc_v2_e5_c2.log",
        "c3": "eval_dpdp_esc_v2_e5_c3.log",
    },
}


def port_combo(method, stack, chunks):
    """Returns (cases, summary_dict). cases sorted by case_id."""
    chunk_paths_abs = {tag: os.path.join(REPO_ROOT, name) for tag, name in chunks.items()}

    all_cases = []
    all_incomplete = []
    for tag, path in chunk_paths_abs.items():
        if not os.path.exists(path):
            print(f"  MISSING LOG: {path}")
            continue
        cases, incomplete = parse_log(path, tag if tag is not None else "_single")
        # tag=None case: stamp chunk to None in output (single-file run).
        if tag is None:
            for c in cases:
                c["source"]["chunk"] = None
        all_cases.extend(cases)
        all_incomplete.extend([(tag, tid, reason) for tid, reason in incomplete])

    # Sort.
    all_cases.sort(key=lambda c: c["case_id"])

    # Detect dups (shouldn't happen across chunks since they're disjoint).
    seen = set()
    dups = []
    for c in all_cases:
        if c["case_id"] in seen:
            dups.append(c["case_id"])
        seen.add(c["case_id"])

    # Lifted metrics.
    if all_cases:
        lifted_sr = sum(1 for c in all_cases if c["dpdp_success"]) / len(all_cases)
        lifted_avgt = sum(c["n_turns"] for c in all_cases) / len(all_cases)
    else:
        lifted_sr = None
        lifted_avgt = None

    # Reported metrics (pooled across chunks).
    warnings = []
    if None in chunks:
        # Single file.
        parsed = parse_reported(chunk_paths_abs[None])
        if parsed:
            rep_sr, rep_avgt, rep_rw = parsed
        else:
            rep_sr = rep_avgt = rep_rw = None
            warnings.append(f"could not parse summary line in {chunks[None]}")
    else:
        rep_sr, rep_avgt, rep_rw, pool_warnings = pool_chunks_reported(chunk_paths_abs)
        warnings.extend(pool_warnings)

    # Sanity: lifted vs reported.
    if lifted_sr is not None and rep_sr is not None and abs(lifted_sr - rep_sr) > 0.005:
        warnings.append(
            f"lifted_sr ({lifted_sr:.4f}) and dpdp_reported_sr ({rep_sr:.4f}) "
            f"differ by > 0.005"
        )
    if lifted_avgt is not None and rep_avgt is not None and abs(lifted_avgt - rep_avgt) > 0.05:
        warnings.append(
            f"lifted_avgt ({lifted_avgt:.3f}) and dpdp_reported_avgt ({rep_avgt:.3f}) "
            f"differ by > 0.05"
        )

    # Missing case_ids.
    expected = set(f"esc_{i:04d}" for i in range(130))
    emitted = set(c["case_id"] for c in all_cases)
    missing = sorted(expected - emitted)

    # Strategy-null audit.
    null_strategy_steps = sum(
        1 for c in all_cases for t in c["dialog"]
        if t["speaker"] == "sys" and "strategy" not in t
    )

    summary = {
        "method": method,
        "stack": stack,
        "n_cases_emitted": len(all_cases),
        "n_cases_incomplete": len(all_incomplete),
        "missing_case_ids": missing,
        "dpdp_reported_sr": round(rep_sr, 4) if rep_sr is not None else None,
        "dpdp_reported_avgt": round(rep_avgt, 4) if rep_avgt is not None else None,
        "dpdp_reported_reward": round(rep_rw, 4) if rep_rw is not None else None,
        "lifted_sr": round(lifted_sr, 4) if lifted_sr is not None else None,
        "lifted_avgt": round(lifted_avgt, 4) if lifted_avgt is not None else None,
        "source_logs": [chunks[k] for k in sorted(chunks.keys(), key=lambda x: (x is None, x))],
        "duplicate_case_ids": dups,
        "null_strategy_sys_turns": null_strategy_steps,
        "warnings": warnings,
    }
    return all_cases, summary


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Writing into {OUT_DIR}")
    print()
    skipped = []

    for (method, stack), chunks in COMBOS.items():
        tag = f"{method}_{stack}"
        print(f"=== {tag} ===")
        cases, summary = port_combo(method, stack, chunks)

        jsonl_path = os.path.join(OUT_DIR, f"{tag}.jsonl")
        with open(jsonl_path, "w") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")

        summary_path = os.path.join(OUT_DIR, f"summary_{tag}.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Smoke-print first line.
        first = (json.dumps(cases[0], ensure_ascii=False)[:240] + "...") if cases else "<no cases>"
        print(f"  emitted {len(cases)} cases -> {os.path.basename(jsonl_path)}")
        print(f"  lifted SR={summary['lifted_sr']}  AvgT={summary['lifted_avgt']}    "
              f"reported SR={summary['dpdp_reported_sr']}  AvgT={summary['dpdp_reported_avgt']}")
        if summary["warnings"]:
            print(f"  WARNINGS:")
            for w in summary["warnings"]:
                print(f"    - {w}")
        print(f"  null-strategy sys turns: {summary['null_strategy_sys_turns']}")
        print(f"  first line: {first}")
        print()

    print("=== SKIPPED ===")
    for method, stack, reason in skipped:
        print(f"  {method}/{stack}: {reason}")


if __name__ == "__main__":
    main()
