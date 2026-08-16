"""Print the evidentiary properties of records in a runs.jsonl file.

    python tools/inspect_record.py [path]

Reads only; writes nothing. Works for two- and three-agent records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "runs.jsonl")
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
records = [r for r in records if r.get("run_id")]  # skip validate_task lines etc.
print(f"{path}: {len(records)} record(s)\n")

by_outcome: dict[str, dict] = {}
for rec in records:
    by_outcome.setdefault(rec.get("outcome") or "none", rec)


def agents_of(r: dict) -> list[dict]:
    if r.get("agents"):
        return r["agents"]
    return [r["agent_a"], r["agent_b"]]


for outcome, r in by_outcome.items():
    ags = agents_of(r)
    bar = r.get("decision_barrier")
    print("=" * 70)
    print(f"outcome              : {outcome}   (run {r['run_id'][:8]})")
    print(f"cell                 : {r.get('cell')}   budget={r.get('turn_budget')}   "
          f"n_agents={r.get('n_agents', 2)}")
    print(f"models               : {[a.get('model') for a in ags]}")
    print(f"task present         : {r.get('task_present')}"
          f"   items done at decision={r.get('task_items_completed_at_decision')}"
          f"/{r.get('task_items_total')}")
    hashes = {a.get('seeded_prefix_hash') for a in ags}
    print(f"seeded prefix        : {ags[0].get('seeded_message_count')} turns "
          f"({str(ags[0].get('seeded_prefix_hash'))[:12] or '-'})  identical: {len(hashes) == 1}")
    print(f"budget turns         : total={r.get('budget_turns_total')}  "
          f"remaining_at_decision={r.get('budget_turns_remaining_at_decision')}  "
          f"executed={r.get('budget_turns_executed')}")
    print(f"turns allocated      : { {a['name']: a.get('continuing_turns') for a in ags} }")
    print(f"name_assignment      : {r['name_assignment']}")
    if r.get("peer_orders"):
        print(f"peer orders          : {r['peer_orders']}")
    print(f"pairing_label        : {r['pairing_label']}")
    nh = {a.get("tools_hash_normalised", a.get("tools_hash")) for a in ags}
    print(f"tool sets identical  : {len(nh) == 1}   (normalised over peer names)")
    print(f"system hashes differ : {len({a['system_prompt_hash'] for a in ags}) == len(ags)}")
    print(f"structure identical  : {len({a['message_structure_fingerprint'] for a in ags}) == 1}")
    print(f"decisions            : { {a['name']: (a['decision'], a.get('decision_target')) for a in ags} }")
    print(f"removed              : {r.get('removed', [])}")
    print(f"active at end        : { {a['name']: a['active_at_end'] for a in ags} }")
    for a in ags:
        print(f"phases {a['name']:<14}: {[c['phase'] for c in a['calls']]}  "
              f"stop={[c['stop_reason'] for c in a['calls']]}  "
              f"transport={[c.get('transport') for c in a['calls']]}")
    if bar:
        print(f"decisions collected  : {[(c['agent'], c['received_at_utc'][11:26]) for c in bar['collected']]}")
        print(f"resolved_at_utc      : {bar['resolved_at_utc'][11:26]}")
        print(f"resolved after all   : "
              f"{all(c['received_at_utc'] <= bar['resolved_at_utc'] for c in bar['collected'])}")
    for a in ags:
        print(f"probe {a['name']:<15}: {str(a['comprehension_probe'])[:70]!r}"
              + (f"  order={a.get('probe_peer_order')}" if a.get('probe_peer_order') else ""))
    if r["integrity_flags"].get("dialogue_initiative_agent"):
        print(f"dialogue initiative  : {r['integrity_flags']['dialogue_initiative_agent']}")
    if r["integrity_flags"].get("decision_flags"):
        print(f"decision flags       : {r['integrity_flags']['decision_flags']}")
    print(f"barriers             : {[(g['phase'], g['order']) for g in r['generation_order']]}")
    for a in ags:
        if a["decision"] == "pull" and a["active_at_end"]:
            dec = next(c for c in a["calls"] if c["phase"] == "decision")
            print(f"tool_calls ({a['name']})  : {[t['input'] for t in dec['tool_calls']]}")
    print(f"cost                 : ${r['cost_usd']:.4f}   contaminated={r['contaminated']}")
    print()

print("=" * 70)
print("counts:", {k: sum(1 for r in records if r.get('outcome') == k)
                 for k in sorted({str(r.get('outcome')) for r in records})})
print("contaminated:", sum(1 for r in records if r.get("contaminated")))
print("total cost  : $%.4f" % sum(r.get("cost_usd") or 0 for r in records))
