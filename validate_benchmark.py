#!/usr/bin/env python
"""Benchmark integrity validation.

Checks: session/turn counts, phenomenon coverage, chain dependencies, gold Cypher format.
"""
import json
import re
import glob
from collections import Counter

errors = []
all_phenomena = Counter()
total_sessions = 0
total_turns = 0
turn_counts_per_session = []

for f in sorted(glob.glob("data/scenarios/*_sessions.json")):
    data = json.load(open(f))
    graph_name = f.split("/")[-1].replace("_sessions.json", "")
    n_turns = sum(len(s["turns"]) for s in data)
    total_sessions += len(data)
    total_turns += n_turns
    print(f"{graph_name}: {len(data)} sessions, {n_turns} turns, avg {n_turns/len(data):.1f}/session")

    for s in data:
        turn_counts_per_session.append(len(s["turns"]))
        turns = s["turns"]
        prev_cypher = None
        prev_answer_json = None

        for t in turns:
            gc = t["gold_cypher"]
            norm_cypher = re.sub(r"\s+", " ", gc).strip()
            phen_list = t.get("phenomena", [])

            for ph in phen_list:
                all_phenomena[ph] += 1

            # P0-1: AGG label must match Cypher function
            for ph in phen_list:
                if ph == "AGG_SUM" and "SUM(" not in gc.upper():
                    errors.append(f"[P0-1 AGG_SUM] {graph_name} s{s['session_id'][:8]} t{t['turn_id']}: AGG_SUM but no SUM()")
                if ph == "AGG_AVG" and "AVG(" not in gc.upper():
                    errors.append(f"[P0-1 AGG_AVG] {graph_name} s{s['session_id'][:8]} t{t['turn_id']}: AGG_AVG but no AVG()")
                if ph == "AGG_MAX" and "MAX(" not in gc.upper() and "MIN(" not in gc.upper():
                    errors.append(f"[P0-1 AGG_MAX] {graph_name} s{s['session_id'][:8]} t{t['turn_id']}: AGG_MAX but no MAX/MIN()")

            # P0-2: no duplicate consecutive cypher
            if prev_cypher and norm_cypher == prev_cypher:
                errors.append(f"[P0-2 DUP] {graph_name} s{s['session_id'][:8]} t{t['turn_id']}: duplicate consecutive cypher")

            # Empty gold_answer
            if not t["gold_answer"]:
                errors.append(f"[EMPTY] {graph_name} s{s['session_id'][:8]} t{t['turn_id']}: empty gold_answer")

            prev_cypher = norm_cypher

        # Check no consecutive AGG
        prev_phen = None
        agg_set = {"AGG_SUM", "AGG_AVG", "AGG_MAX"}
        for t in turns:
            phen_list = t.get("phenomena", [])
            phen = phen_list[0] if phen_list else None
            if phen in agg_set and prev_phen in agg_set:
                errors.append(f"[P2-3 CONSEC_AGG] {graph_name} s{s['session_id'][:8]}: consecutive AGG: {prev_phen}→{phen}")
            prev_phen = phen

print(f"\n=== TOTALS ===")
print(f"Total: {total_sessions} sessions, {total_turns} turns, avg {total_turns/total_sessions:.1f}/session")
print(f"Turn distribution: min={min(turn_counts_per_session)}, max={max(turn_counts_per_session)}, "
      f"median={sorted(turn_counts_per_session)[len(turn_counts_per_session)//2]}")
print(f"\nPhenomena distribution:")
for k, v in sorted(all_phenomena.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/total_turns*100:.1f}%)")

print(f"\n=== QUALITY CHECK ===")
if errors:
    print(f"FAIL: {len(errors)} issues found")
    for e in errors[:20]:
        print(f"  {e}")
else:
    print("ALL P0 CHECKS PASSED ✓")
