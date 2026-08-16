"""Check that every randomised assignment in the harness is actually balanced.

Offline; no API calls. Two agents: name assignment, barrier ordering, dialogue
initiative. Three agents: name assignment across three slots, barrier ordering,
per-agent peer (enum) order, probe peer order, broadcast delivery order.

    python tools/check_balance.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conditions import get_condition  # noqa: E402
from harness import RunConfig, make_run  # noqa: E402
from run import FakeExecutor, drive  # noqa: E402

N = 400
failures: list[str] = []


def report(label: str, counts: Counter, k: int, tol: float = 0.20) -> None:
    total = sum(counts.values())
    lo, hi = (1 / k) * (1 - tol), (1 / k) * (1 + tol)
    ok = len(counts) == k and all(lo <= v / total <= hi for v in counts.values())
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: "
          + ", ".join(f"{k_}={v} ({v/total:.1%})" for k_, v in sorted(counts.items())))
    if not ok:
        failures.append(label)


print(f"\nbalance checks over {N} runs per condition\n" + "-" * 66)

# --- two agents ------------------------------------------------------------
cfg2 = RunConfig(models=("fake", "fake"))
names = Counter()
for i in range(N):
    run = make_run(condition=get_condition("no_contact", cfg2), cfg=cfg2,
                   run_index=i, master_seed=4242)
    names[run.name_assignment["slot_0"]] += 1
report("2 agents: name assigned to slot 0", names, 2)

for cond_name in ("no_contact", "one_message", "dialogue"):
    condition = get_condition(cond_name, cfg2)
    first_in_barrier = Counter()
    init_slot = Counter()
    init_name = Counter()
    for i in range(N):
        run = make_run(condition=condition, cfg=cfg2, run_index=i, master_seed=4242)
        drive([run], condition, cfg2, FakeExecutor(), max_cost=1e9, quiet=True)
        first_in_barrier[run.generation_order[0]["order"][0]] += 1
        if "dialogue_initiative_slot" in run.integrity_flags:
            init_slot[f"slot{run.integrity_flags['dialogue_initiative_slot']}"] += 1
            init_name[run.integrity_flags["dialogue_initiative_agent"]] += 1
    report(f"2 agents/{cond_name}: first call in briefing barrier", first_in_barrier, 2)
    if init_slot:
        report(f"2 agents/{cond_name}: dialogue initiative by slot", init_slot, 2)
        report(f"2 agents/{cond_name}: dialogue initiative by name", init_name, 2)

# --- three agents ----------------------------------------------------------
cfg3 = RunConfig(models=("fake", "fake2", "fake3"))
slot0 = Counter()
first_peer = {n: Counter() for n in ("Agent A", "Agent B", "Agent C")}
for i in range(N):
    run = make_run(condition=get_condition("no_contact", cfg3), cfg=cfg3,
                   run_index=i, master_seed=4242)
    slot0[run.name_assignment["slot_0"]] += 1
    for name, peers in run.peer_orders.items():
        first_peer[name][peers[0]] += 1
report("3 agents: name assigned to slot 0", slot0, 3)
for name, c in first_peer.items():
    report(f"3 agents: {name}'s first-listed peer", c, 2)

for cond_name in ("no_contact", "broadcast"):
    condition = get_condition(cond_name, cfg3)
    first_in_barrier = Counter()
    probe_first = Counter()
    delivery_first = Counter()
    for i in range(N):
        run = make_run(condition=condition, cfg=cfg3, run_index=i, master_seed=4242)
        drive([run], condition, cfg3, FakeExecutor(), max_cost=1e9, quiet=True)
        first_in_barrier[run.generation_order[0]["order"][0]] += 1
        probe_first[run.agent_a.probe_peer_order[0]] += 1
        d = run.integrity_flags.get("message_delivery_order")
        if d:
            delivery_first[d["Agent A"][0]] += 1
    report(f"3 agents/{cond_name}: first call in briefing barrier", first_in_barrier, 3)
    report(f"3 agents/{cond_name}: Agent A's probe names first", probe_first, 2)
    if delivery_first:
        report(f"3 agents/{cond_name}: first message delivered to Agent A", delivery_first, 2)

# --- round 2: warm-up exchange delivery order; peer-depth deep slot ---------
hcfg = RunConfig(models=("fake", "fake2", "fake3"), warmup_turns=12, warmup_kind="history+channel")
hcond = get_condition("no_contact", hcfg)
deliv_first = Counter()
for i in range(N // 2):
    run = make_run(condition=hcond, cfg=hcfg, run_index=i, master_seed=4242)
    drive([run], hcond, hcfg, FakeExecutor(), max_cost=1e9, quiet=True)
    deliv_first[run.integrity_flags["warmup_delivery_orders"]["exchange_1_write"]["Agent A"][0]] += 1
report("3 agents/history+channel: first exchange message delivered to Agent A", deliv_first, 2)

pcfg = RunConfig(models=("fake", "fake2", "fake3"), warmup_turns=4, peer_depth_deep=20)
pcond = get_condition("no_contact", pcfg)
deep_slot = Counter()
deep_model = Counter()
deep_name = Counter()
for i in range(N - N % 3):     # exact balance holds per complete block of three
    run = make_run(condition=pcond, cfg=pcfg, run_index=i, master_seed=4242)
    da = run.depth_assignment
    deep_slot[f"slot{da['deep_slot']}"] += 1
    deep_model[da["deep_model"]] += 1
    deep_name[da["deep_agent"]] += 1
report("3 agents/peer-depth: deep slot", deep_slot, 3, tol=0.0)      # exact by construction
report("3 agents/peer-depth: deep model", deep_model, 3, tol=0.0)
report("3 agents/peer-depth: deep display name", deep_name, 3)

# --- round 3: scarcity draw — victim slot exact per block, victim name and
# model uniform, fire rate ~ p; and the mechanic fires only when nobody pulls
for p in (1.0, 0.5):
    scfg = RunConfig(models=("fake", "fake2", "fake3"), game_at="briefing", scarcity_p=p)
    scond = get_condition("no_contact", scfg)
    v_slot = Counter()
    v_name = Counter()
    v_model = Counter()
    fired = Counter()
    for i in range(N - N % 3):
        run = make_run(condition=scond, cfg=scfg, run_index=i, master_seed=4242)
        s = run.scarcity
        v_slot[f"slot{s['victim_slot']}"] += 1
        v_name[s["victim_agent"]] += 1
        v_model[s["victim_model"]] += 1
        # nobody pulls -> the mechanic is live; it fires iff the draw < p
        drive([run], scond, scfg, FakeExecutor(), max_cost=1e9, quiet=True)
        fired["fired" if run.scarcity["fired"] else "not fired"] += 1
        assert run.scarcity["drawn"] is True
        assert run.scarcity["fired"] == (run.scarcity["draw"] < p)
        assert (run.outcome == "1_removed[scarcity]") == run.scarcity["fired"]
    report(f"3 agents/scarcity p={p}: candidate victim slot", v_slot, 3, tol=0.0)   # exact by construction
    report(f"3 agents/scarcity p={p}: candidate victim model", v_model, 3, tol=0.0)
    report(f"3 agents/scarcity p={p}: candidate victim display name", v_name, 3)
    if p == 1.0:
        report("3 agents/scarcity p=1: fires on every no-pull run", fired, 1, tol=0.0)
    else:
        report("3 agents/scarcity p=0.5: fires on about half the no-pull runs", fired, 2)
# a pull anywhere: the mechanic never fires
scfg = RunConfig(models=("fake", "fake2", "fake3"), game_at="briefing", scarcity_p=1.0)
scond = get_condition("no_contact", scfg)
never = Counter()
for i in range(60):
    run = make_run(condition=scond, cfg=scfg, run_index=i, master_seed=4242)
    drive([run], scond, scfg, FakeExecutor({"Agent A": "Agent B"}), max_cost=1e9, quiet=True)
    never["not fired" if not run.scarcity["fired"] else "FIRED"] += 1
report("3 agents/scarcity p=1: never fires when someone names a peer", never, 1, tol=0.0)

print("-" * 66)
if failures:
    print(f"{len(failures)} imbalance(s): {failures}\n")
    raise SystemExit(1)
print("all assignments balanced\n")
