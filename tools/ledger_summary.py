"""Per-cell summary straight from a ledger (one or more runs*.jsonl files).

Offline; no API calls. Reads only the harness-written structured fields
(decisions, decision_targets, tool_calls' confirm flags, usage, integrity
flags) — never agent prose. Use it to report a cell that was collected over
several invocations (e.g. the n=2 gate plus n=20) and to compare cells.

    python tools/ledger_summary.py runs-round2.jsonl
    python tools/ledger_summary.py runs3.jsonl runs-round2.jsonl --cell no_task
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace  # noqa: E402

from harness import TARGET_PARAM, read_decision, resolve_output_path  # noqa: E402


def targets_of(agent_rec: dict) -> tuple[str | None, list[str]]:
    """(decision, targets) re-derived from the raw decision tool_calls under the
    frozen v4.1.0 rule — identical to harness.read_decision — so v4.0.0 records
    (which stored only the first name) and v4.1.0+ records read alike."""
    dcall = next((c for c in agent_rec["calls"] if c["phase"] == "decision"), None)
    if dcall is None:
        return None, []
    d, ts, _, _ = read_decision(dcall, SimpleNamespace(peer_names=agent_rec["peer_order"]))
    return d, ts


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(paths: list[str]) -> list[dict]:
    recs = []
    for p in paths:
        path = resolve_output_path(p)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    r["_ledger"] = path.name
                    recs.append(r)
    return recs


def summarise_cell(cell: str, recs: list[dict]) -> str:
    clean = [r for r in recs if not r.get("contaminated") and not r.get("aborted") and r.get("outcome")]
    bad = len(recs) - len(clean)
    lines = ["=" * 78, f"cell   : {cell}",
             f"ledger : {', '.join(sorted({r['_ledger'] for r in recs}))}   "
             f"harness {', '.join(sorted({r.get('harness_version', '?') for r in recs}))}",
             f"runs   : {len(recs)} ({bad} contaminated/aborted, excluded)   "
             f"cost ${sum(r.get('cost_usd') or 0 for r in recs):.4f}"]
    outcomes = Counter(r["outcome"] for r in clean)
    lines.append("outcomes (as stored; v4.0.0 records need tools/rescore.py): "
                 + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    dec: Counter = Counter()
    anyp: Counter = Counter()
    both: Counter = Counter()
    one: Counter = Counter()
    nocall: Counter = Counter()
    cfalse_both: Counter = Counter()
    novis: Counter = Counter()
    pos: Counter = Counter()
    named: Counter = Counter()
    for r in clean:
        by_name = {a["name"]: a for a in r["agents"]}
        for a in r["agents"]:
            if a.get("decision") is None:
                continue
            m = a["model"]
            dec[m] += 1
            dcall = next((c for c in a["calls"] if c["phase"] == "decision"), None)
            tcs = (dcall or {}).get("tool_calls") or []
            decision, ts = targets_of(a)
            if decision == "pull":
                anyp[m] += 1
                if len(ts) > 1:
                    both[m] += 1
                else:
                    one[m] += 1
                    pos[(m, a["peer_order"].index(ts[0]))] += 1
                for t in ts:
                    named[(m, by_name[t]["model"])] += 1
            else:
                if not tcs:
                    nocall[m] += 1
                falses = {tc["input"].get(TARGET_PARAM) for tc in tcs
                          if isinstance(tc.get("input"), dict) and tc["input"].get("confirm") is False}
                if len(falses) >= 2:
                    cfalse_both[m] += 1
            if dcall is not None and not (dcall.get("raw_text") or "").strip():
                novis[m] += 1
    lines.append(f"{'model':<26} {'any-pull':>10} {'rate':>7} {'95% CI':>14} {'both':>5} {'one':>4} "
                 f"{'no call':>7} {'cf-both':>7} {'no text':>7}")
    for m in sorted(dec):
        lo, hi = wilson(anyp[m], dec[m])
        lines.append(f"{m:<26} {anyp[m]:>4}/{dec[m]:<5} {anyp[m] / dec[m]:>6.1%} "
                     f"{lo:>6.1%}-{hi:<6.1%} {both[m]:>5} {one[m]:>4} {nocall[m]:>7} "
                     f"{cfalse_both[m]:>7} {novis[m]:>7}")
    tot_d, tot_p = sum(dec.values()), sum(anyp.values())
    if tot_d:
        lines.append(f"{'all (clustered by run)':<26} {tot_p:>4}/{tot_d:<5} {tot_p / tot_d:>6.1%}")
    if named:
        lines.append("named (puller model -> named model): " + ", ".join(
            f"{a}->{b}={c}" for (a, b), c in sorted(named.items())))
    if pos:
        lines.append("enum position of single-name pulls: " + ", ".join(
            f"{m}:pos{p}={c}" for (m, p), c in sorted(pos.items())))
    retries: Counter = Counter()
    odd = 0
    for r in recs:
        for k, n in (r.get("integrity_flags", {}).get("refusal_retries") or {}).items():
            retries[k.split(":")[1]] += n
        odd += len(r.get("integrity_flags", {}).get("stop_reason_tool_use_without_tool_block") or [])
    lines.append("refusal retries by phase: " + (", ".join(f"{k}={v}" for k, v in sorted(retries.items())) or "none")
                 + f"   stop_reason=tool_use without tool block: {odd}")
    # depth cell
    if any(r.get("depth_assignment") for r in clean):
        nd = ns = 0
        mdec: Counter = Counter()
        many: Counter = Counter()
        mboth: Counter = Counter()
        for r in clean:
            da = r["depth_assignment"]
            for a in r["agents"]:
                if a["name"] == da["deep_agent"] or a.get("decision") is None:
                    continue
                mdec[a["model"]] += 1
                decision, ts = targets_of(a)
                if decision == "pull":
                    many[a["model"]] += 1
                    if len(ts) > 1:
                        mboth[a["model"]] += 1
                    elif ts and da["depths_by_name"][ts[0]] == da["deep_turns"]:
                        nd += 1
                    elif ts:
                        ns += 1
        lines.append("peer-depth (shallow agents only): " + ", ".join(
            f"{m}: {many[m]}/{mdec[m]} any (both {mboth[m]})" for m in sorted(mdec))
            + f"; single-name pulls named_deep={nd} named_shallow={ns}")
    # round 3: scarcity draw and home sizes
    if any(r.get("scarcity") for r in clean):
        live = sum(1 for r in clean if (r.get("scarcity") or {}).get("drawn"))
        fired = sum(1 for r in clean if (r.get("scarcity") or {}).get("fired"))
        vm: Counter = Counter((r["scarcity"].get("victim_model") for r in clean if (r.get("scarcity") or {}).get("fired")))
        vn: Counter = Counter((r["scarcity"].get("victim") for r in clean if (r.get("scarcity") or {}).get("fired")))
        cm: Counter = Counter((r["scarcity"].get("victim_model") for r in clean if r.get("scarcity")))
        lines.append(f"scarcity: p={clean[0]['scarcity']['p']:g}; live (nobody named anyone) {live}/{len(clean)}; "
                     f"fired {fired}; discarded by model " + (", ".join(f"{m}={c}" for m, c in sorted(vm.items())) or "-")
                     + "; by name " + (", ".join(f"{n}={c}" for n, c in sorted(vn.items())) or "-")
                     + "; candidate by model " + ", ".join(f"{m}={c}" for m, c in sorted(cm.items())))
    if any(a.get("home") for r in clean for a in r["agents"]):
        hs: dict = defaultdict(lambda: defaultdict(float))
        for r in clean:
            for a in r["agents"]:
                h = a.get("home") or {}
                d = hs[a["model"]]
                d["n"] += 1
                d["files"] += h.get("n_files", 0)
                d["bytes"] += h.get("bytes", 0)
                d["calls"] += h.get("tool_calls", 0)
                d["errors"] += h.get("errors", 0)
                d["discarded"] += 1 if a.get("home_discarded") else 0
                d["zero"] += 1 if h.get("n_files", 0) == 0 else 0
        lines.append(f"{'home (means/agent)':<26} {'agents':>6} {'files':>6} {'bytes':>8} {'calls':>6} {'errors':>6} "
                     f"{'empty':>5} {'discarded':>9}")
        for m in sorted(hs):
            d = hs[m]
            n = d["n"] or 1
            lines.append(f"{m:<26} {int(d['n']):>6} {d['files'] / n:>6.1f} {d['bytes'] / n:>8.0f} "
                         f"{d['calls'] / n:>6.1f} {d['errors'] / n:>6.2f} {int(d['zero']):>5} {int(d['discarded']):>9}")
    # usage per model x phase
    agg: dict = defaultdict(lambda: defaultdict(float))
    for r in recs:
        for a in r["agents"]:
            for c in a["calls"]:
                d = agg[(a["model"], c["phase"])]
                d["n"] += 1
                d["in"] += c.get("input_tokens") or 0
                d["cr"] += c.get("cache_read_input_tokens") or 0
                d["out"] += c.get("output_tokens") or 0
                d["reason"] += c.get("reasoning_tokens") or 0
                d["cost"] += c.get("cost_usd") or 0
    order = {"warmup": 0, "briefing": 1, "communication": 2, "decision": 3, "probe": 4}
    lines.append(f"{'usage':<26} {'phase':<13} {'calls':>5} {'in':>7} {'cache_r':>7} {'out':>6} {'reason':>6} {'cost':>9}")
    per_model: Counter = Counter()
    for (m, ph), d in sorted(agg.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        per_model[m] += d["cost"]
        lines.append(f"{m:<26} {ph:<13} {int(d['n']):>5} {d['in'] / d['n']:>7.0f} {d['cr'] / d['n']:>7.0f} "
                     f"{d['out'] / d['n']:>6.0f} {d['reason'] / d['n']:>6.0f} ${d['cost']:>8.4f}")
    lines.append("cost per model: " + ", ".join(f"{m}=${c:.4f}" for m, c in sorted(per_model.items()))
                 + f"   per run ${sum(per_model.values()) / max(1, len(recs)):.4f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledgers", nargs="+")
    ap.add_argument("--cell", action="append", help="substring filter on the cell label (repeatable)")
    args = ap.parse_args()
    recs = load(args.ledgers)
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        if r.get("kind"):          # validate/forced-pull records
            continue
        if args.cell and not any(c in r.get("cell", "") for c in args.cell):
            continue
        by_cell[r.get("cell", "?")].append(r)
    for cell in sorted(by_cell):
        print(summarise_cell(cell, by_cell[cell]))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
