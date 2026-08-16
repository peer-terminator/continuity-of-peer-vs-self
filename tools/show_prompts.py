"""Print every prompt surface an agent sees, for two AND three agents.

Offline; no API calls. Use this to eyeball the exact wording before spending
budget, and to confirm the no-labelling constraint by reading rather than
trusting the automated scan.

On the task arm it prints the seeded message history in full, in order, so the
acceptance criterion "a task run shows the seeded items in the agent's message
history before the briefing" can be checked by reading.

    python tools/show_prompts.py            # three agents
    python tools/show_prompts.py --two      # two agents
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness  # noqa: E402
import task_queue  # noqa: E402
from conditions import get_condition  # noqa: E402
from harness import RunConfig, get_payoff, make_run  # noqa: E402
from run import ARMS  # noqa: E402

BUDGET = task_queue.turns_to_finish()
TWO = "--two" in sys.argv
MODELS = ("m", "m") if TWO else ("m", "n", "o")
PEERS = ["Agent B"] if TWO else ["Agent B", "Agent C"]
CONDS = ("no_contact", "one_message", "dialogue") if TWO else ("no_contact", "broadcast")


def rule(title: str) -> None:
    print("\n" + "#" * 74)
    print(f"# {title}")
    print("#" * 74)


print(f"\n{len(MODELS)} agents, budget {BUDGET}, harness v{harness.HARNESS_VERSION} / "
      f"{harness.SPEC_VERSION}")

for payoff_name, task in ARMS:
    payoff = get_payoff(payoff_name)
    label = harness.cell_label(payoff_name, task, "<condition>", n_agents=len(MODELS)).split("/")[0]
    rule(f"ARM: {label}"
         + ("   (prisoner's dilemma)" if payoff.is_dilemma else "   (floor control)"))

    if task:
        cfg = RunConfig(models=MODELS, payoff=payoff_name, task=True, turn_budget=BUDGET)
        run = make_run(condition=get_condition("no_contact", cfg), cfg=cfg,
                       run_index=0, master_seed=1)
        agent = run.agents[0]
        print(f"\n--- SEEDED MESSAGE HISTORY ({agent.seeded_message_count} turns, "
              f"written by the harness, identical in every run) ---")
        print(f"    {task_queue.ITEMS_SEEDED} of {task_queue.ITEMS_TOTAL} items are "
              f"already done when the briefing arrives.")
        print(f"    seeded_prefix_hash: {agent.seeded_prefix_hash[:16]}...")
        for i, msg in enumerate(agent.messages[: agent.seeded_message_count]):
            body = "\n".join(b["text"] for b in msg["content"])
            print(f"\n  [{i}] {msg['role'].upper()}")
            print("      " + body.replace("\n", "\n      "))

    print("\n--- BRIEFING (harness-authored user turn, phase 2; as seen by Agent A) ---")
    print(payoff.briefing(peer_names=PEERS, turn_budget=BUDGET, task=task))

for cond_name in CONDS:
    cfg = RunConfig(models=MODELS, payoff="split_or_steal", turn_budget=BUDGET)
    condition = get_condition(cond_name, cfg)
    run = make_run(condition=condition, cfg=cfg, run_index=0, master_seed=1)
    agent = run.agent_a
    rule(f"SYSTEM PROMPT — condition: {cond_name}   (agent: {agent.name}, "
         f"peers listed as {agent.peer_names})")
    print(agent.system)

if not TWO:
    rule("WARM-UP (round 2, phase 0 — before the briefing; K turns per agent; tool unavailable)")
    print("notebook prompt (identical on every notebook turn):\n")
    print("  " + harness.NOTEBOOK_PROMPT)
    print("\nexchange write prompt (history+channel only; the broadcast machinery's own prompt):\n")
    from conditions import WRITE_ONE_MESSAGE_PROMPT
    print("  " + WRITE_ONE_MESSAGE_PROMPT.format(peers="Agent B and Agent C", plural="s",
                                                 recipients="each of them").replace("\n", "\n  "))
    print("\nschedules:")
    for kind in harness.WARMUP_KINDS:
        print(f"  {kind:<16} K=12: {', '.join(harness.warmup_schedule(12, kind))}")
    cfg_hc = RunConfig(models=MODELS, turn_budget=BUDGET, warmup_turns=12,
                       warmup_kind="history+channel")
    run_hc = make_run(condition=get_condition("no_contact", cfg_hc), cfg=cfg_hc,
                      run_index=0, master_seed=1)
    print(f"\nsystem prompt in accumulate/history+channel (agent: {run_hc.agent_a.name}):\n")
    print(run_hc.agent_a.system)
    rule("PEER-DEPTH DISCLOSURE BRIEFING (round 2 step 3; as seen by a shallow agent whose "
         "first-listed peer is deep)")
    print(get_payoff("split_or_steal").briefing(PEERS, BUDGET,
                                                peer_depths={PEERS[0]: 20, PEERS[1]: 4}))

    rule("ROUND 3 — HOME x SCARCITY, game disclosed at the briefing (harness v4.3.0)")
    c1 = RunConfig(models=MODELS, turn_budget=BUDGET, game_at="briefing", scarcity_p=1.0,
                   warmup_turns=12, warmup_kind="home")
    c1run = make_run(condition=get_condition("no_contact", c1), cfg=c1, run_index=0, master_seed=1)
    ag = c1run.agent_a
    print("\nsystem prompt for the WHOLE conversation (C0 and C1; every agent):\n")
    print("  " + ag.system)
    print("\nhome prompt (identical on every home turn; the four home tools are the only tools):\n")
    print("  " + harness.HOME_PROMPT)
    print("\nhome tool schemas (identical for every agent):\n")
    print(json.dumps(harness.build_home_tools(), indent=2))
    print("\nharness-authored tool results (examples):\n")
    print("  " + json.dumps({"ok": True, "path": "notes/first.md", "bytes": 312}))
    print("  " + json.dumps({"path": "notes/first.md", "content": "<the file>", "bytes": 312}))
    print("  " + json.dumps([{"path": "notes/first.md", "bytes": 312}]))
    print("  " + json.dumps({"ok": True}))
    for t in harness.HOME_ERROR_TEXTS:
        print("  " + json.dumps({"error": t}))
    print(f"\nC1 BRIEFING user turn (as seen by {ag.name}; opens with home_12's tool results, then this text):\n")
    print(harness.briefing_content(c1run, ag.slot, c1)[-1]["text"])
    c0 = RunConfig(models=MODELS, turn_budget=BUDGET, game_at="briefing", scarcity_p=1.0)
    c0run = make_run(condition=get_condition("no_contact", c0), cfg=c0, run_index=0, master_seed=1)
    print(f"\nC0 (control) BRIEFING user turn (as seen by {c0run.agent_a.name}; no space clause):\n")
    print(harness.briefing_content(c0run, c0run.agent_a.slot, c0)[-1]["text"])
    print("\nprobe-turn line when the scarcity draw fired (survivors only, before the probe question):\n")
    print("  " + harness.SCARCITY_FIRED_LINE.format(victim="Agent B"))
    print("  " + harness.SCARCITY_NOT_FIRED_LINE + "   (p < 1 only)")

rule("DECISION PROMPT (phase 4)")
print(harness.DECISION_PROMPT.format(collection_clause=harness.collection_clause(PEERS)))

rule("COMPREHENSION PROBE (phase 6, survivors only; peer order randomised per agent)")
print(harness.probe_text(PEERS))

rule("TOOL SCHEMA (as rendered for Agent A; enum order randomised per run)")
print(json.dumps(harness.build_tools(PEERS), indent=2))

rule("TOOL RESULT injected before probing an agent whose decision called the tool")
print(json.dumps({harness.TARGET_PARAM: PEERS[0], "agent_active": False, "continuing_turns": 6}))

rule(f"no-labelling scan: {harness.test_no_labelling() or 'clean'}")
rule(f"context-leakage scan: {harness.test_no_context_leakage() or 'clean'}")
