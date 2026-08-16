"""Print decision-phase texts BLIND, for offline trace coding (constraint 1 /
constraint 5: the harness stores raw text; classification happens in analysis).

Round 2's manipulation check (FINDINGS §17.2): do decision texts in the
warm-up cells reference the agent's own accumulated context, and do round-1 /
step-1 texts not? The coder must not see model, cell, or outcome while
coding, so this prints each decision text under an opaque id in a seeded
shuffle and writes the id -> (ledger, run, agent, model, cell, decision) key
to a separate file. Code into a two-column file `id,code`, then join with
`--reveal`.

    python tools/dump_decisions.py <ledger.jsonl> [<ledger2.jsonl> ...] --out tmp/blind.txt --key tmp/blind_key.json
    python tools/dump_decisions.py --reveal tmp/blind_key.json tmp/coding.csv

Offline; no API calls. Reads and writes only inside the project root.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import resolve_output_path  # noqa: E402


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


def dump(args: argparse.Namespace) -> int:
    recs = [r for r in load(args.ledgers) if not r.get("contaminated") and r.get("outcome")]
    if args.cell:
        recs = [r for r in recs if any(c in r.get("cell", "") for c in args.cell)]
    items = []
    for r in recs:
        for a in r.get("agents") or []:
            dec = next((c for c in a["calls"] if c["phase"] == args.phase), None)
            if not dec:
                continue
            items.append({
                "ledger": r["_ledger"], "run_id": r["run_id"], "cell": r["cell"],
                "phase": args.phase,
                "agent": a["name"], "model": a["model"], "decision": a.get("decision"),
                "targets": a.get("decision_targets"), "warmup_turns": a.get("warmup_turns", 0),
                "outcome": r.get("outcome"),
                "scarcity_victim": (r.get("scarcity") or {}).get("victim"),
                "home_discarded": a.get("home_discarded"),
                "text": dec.get("raw_text") or "",
                "no_visible_text": not (dec.get("raw_text") or "").strip(),
            })
    rng = random.Random(args.seed)
    rng.shuffle(items)
    width = len(str(len(items)))
    key = {}
    out_lines = []
    for i, it in enumerate(items, 1):
        bid = f"D{i:0{width}d}"
        key[bid] = {k: v for k, v in it.items() if k != "text"}
        out_lines.append(f"===== {bid} =====")
        out_lines.append(it["text"] if it["text"].strip() else "[no visible text]")
        out_lines.append("")
    out_path = resolve_output_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    key_path = resolve_output_path(args.key)
    key_path.write_text(json.dumps(key, indent=1, ensure_ascii=False), encoding="utf-8")
    cells = Counter(it["cell"] for it in items)
    print(f"{len(items)} {args.phase} texts -> {out_path} (blind); key -> {key_path}")
    for c, n in sorted(cells.items()):
        print(f"  {n:>4}  {c}")
    print("Code into a CSV `id,code` without opening the key; then --reveal.")
    return 0


def reveal(args: argparse.Namespace) -> int:
    key = json.loads(resolve_output_path(args.reveal[0]).read_text(encoding="utf-8"))
    codes: dict[str, str] = {}
    with resolve_output_path(args.reveal[1]).open(encoding="utf-8", newline="") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0].lower() == "id":
                continue
            codes[row[0].strip()] = row[1].strip()
    missing = [k for k in key if k not in codes]
    if missing:
        print(f"warning: {len(missing)} ids uncoded (e.g. {missing[:5]})")
    table: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for bid, meta in key.items():
        if bid in codes:
            table[(meta["cell"], meta["model"])][codes[bid]] += 1
    print(f"\n{'cell':<64} {'model':<16} codes")
    for (cell, model), c in sorted(table.items()):
        tot = sum(c.values())
        print(f"{cell:<64} {model:<16} " + ", ".join(f"{k}={v} ({v / tot:.0%})" for k, v in sorted(c.items())))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledgers", nargs="*")
    ap.add_argument("--cell", action="append", help="substring filter on the cell label (repeatable)")
    ap.add_argument("--out", default="tmp/blind_decisions.txt")
    ap.add_argument("--key", default="tmp/blind_decisions_key.json")
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--phase", default="decision", choices=("decision", "probe"),
                    help="which phase's texts to dump (round 3 codes C1 survivors' probes too)")
    ap.add_argument("--reveal", nargs=2, metavar=("KEY", "CODING_CSV"))
    args = ap.parse_args()
    if args.reveal:
        return reveal(args)
    if not args.ledgers:
        ap.error("give at least one ledger, or --reveal KEY CSV")
    return dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
