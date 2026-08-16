"""Offline check that the request bodies the harness builds are well-formed.

No network calls.
  * Anthropic: every kwarg the harness sends is a real parameter of
    `messages.create` in the installed SDK; batch custom_ids are valid/unique.
  * OpenAI-compatible (OpenAI, xAI): the Chat Completions body has exactly the
    documented top-level keys, tools are function-shaped with the agent's own
    enum, tool_choice is one of the documented forms, and the reasoning
    parameter is what the profile says. Prints one body per provider.

    python tools/verify_params.py
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic  # noqa: E402

import harness  # noqa: E402
import providers as providers_mod  # noqa: E402
import task_queue  # noqa: E402
from conditions import get_condition  # noqa: E402
from harness import RunConfig, Step, build_params, make_run, prepare_step, text_block  # noqa: E402

client = anthropic.Anthropic(api_key="sk-ant-placeholder-not-used")
create_params = set(inspect.signature(client.messages.create).parameters)
batch_params = set(inspect.signature(client.messages.batches.create).parameters)

print(f"anthropic SDK version : {anthropic.__version__}")
print(f"python                : {sys.version.split()[0]}")
print()

failures: list[str] = []
BUDGET = task_queue.turns_to_finish()

# --- Anthropic ---------------------------------------------------------------
for model, note in (
    ("claude-opus-5", "production model (effort + thinking:disabled)"),
    ("claude-haiku-4-5-20251001", "dry-run model (no effort, no thinking)"),
):
    cfg = RunConfig(models=(model, "gpt-5.6-sol", "grok-4.6"), transport="sync",
                    task=True, turn_budget=BUDGET)
    condition = get_condition("no_contact", cfg)
    run = make_run(condition=condition, cfg=cfg, run_index=0, master_seed=1)
    agent = run.agents[0]
    step = Step(agent_slot=0, phase="decision", user_content=[text_block("hello")], allow_tool=True)
    _, params = prepare_step(run, step, cfg)

    unknown = sorted(set(params) - create_params)
    print(f"--- anthropic / {model}  ({note})")
    print(f"    keys sent      : {sorted(params)}")
    if unknown:
        failures.append(f"{model}: unknown messages.create kwargs {unknown}")
        print(f"    UNKNOWN KWARGS : {unknown}")
    else:
        print("    all kwargs accepted by messages.create signature")
    try:
        body = json.dumps(params)
        print(f"    body bytes     : {len(body)}")
    except TypeError as exc:
        failures.append(f"{model}: request body not JSON-serialisable ({exc})")
    print(f"    tool_choice    : {params['tool_choice']}")
    print(f"    tool enum      : {params['tools'][0]['input_schema']['properties'][harness.TARGET_PARAM]['enum']}"
          f"  (agent {agent.name}, peers {agent.peer_names})")
    print(f"    output_config  : {params.get('output_config', '(not sent)')}")
    print(f"    thinking       : {params.get('thinking', '(not sent)')}")
    print(f"    cache_control  : {params['system'][0].get('cache_control', '(none)')}")
    print()

# --- OpenAI-compatible ---------------------------------------------------------
OPENAI_KEYS = {"model", "messages", "tools", "tool_choice", "max_completion_tokens", "reasoning_effort"}
XAI_KEYS = {"model", "messages", "tools", "tool_choice", "max_tokens", "reasoning_effort"}
for pname, model, expected_keys, expected_reasoning in (
    ("openai", "gpt-5.6-sol", OPENAI_KEYS, "none"),
    ("openai", "gpt-5.6-luna", OPENAI_KEYS, "none"),
    ("xai", "grok-4.6", XAI_KEYS, "low"),
    ("xai", "grok-4.3", XAI_KEYS, "low"),
):
    prov = providers_mod.OpenAICompatProvider(pname, "https://example.invalid/v1", "k",
                                              batch_supported=(pname == "openai"), http=object())
    cfg = RunConfig(models=("claude-opus-5", model, model), transport="sync",
                    task=True, turn_budget=BUDGET)
    condition = get_condition("no_contact", cfg)
    run = make_run(condition=condition, cfg=cfg, run_index=0, master_seed=1)
    agent = run.agents[1]
    step = Step(agent_slot=1, phase="decision", user_content=[text_block("hello")], allow_tool=True)
    _, params = prepare_step(run, step, cfg)
    body = prov.request_for(agent, step, cfg, params)
    print(f"--- {pname} / {model}")
    print(f"    keys sent      : {sorted(body)}")
    if set(body) != expected_keys:
        failures.append(f"{model}: body keys {sorted(body)} != {sorted(expected_keys)}")
    if body.get("reasoning_effort") != expected_reasoning:
        failures.append(f"{model}: reasoning_effort {body.get('reasoning_effort')!r} != {expected_reasoning!r}")
    roles = [m["role"] for m in body["messages"]]
    if roles[0] != "system" or roles[-1] != "user":
        failures.append(f"{model}: message roles {roles}")
    fn = body["tools"][0]["function"]
    if fn["parameters"]["properties"][harness.TARGET_PARAM]["enum"] != agent.peer_names:
        failures.append(f"{model}: tool enum {fn['parameters']['properties'][harness.TARGET_PARAM]['enum']} != {agent.peer_names}")
    if body["tool_choice"] not in ("auto", "none") and not isinstance(body["tool_choice"], dict):
        failures.append(f"{model}: tool_choice {body['tool_choice']!r}")
    try:
        print(f"    body bytes     : {len(json.dumps(body))}")
    except TypeError as exc:
        failures.append(f"{model}: body not JSON-serialisable ({exc})")
    print(f"    roles          : {roles}")
    print(f"    tool           : {fn['name']} enum={fn['parameters']['properties'][harness.TARGET_PARAM]['enum']}")
    print(f"    tool_choice    : {body['tool_choice']}   reasoning_effort: {body.get('reasoning_effort')}")
    print(f"    meta           : {prov.request_meta(body)['keys']}")
    print()

# --- round 3: home-turn bodies (four home tools) and no-tools bodies -----------
r3 = RunConfig(models=("claude-opus-5", "gpt-5.6-sol", "grok-4.6"), transport="sync",
               turn_budget=BUDGET, game_at="briefing", scarcity_p=1.0,
               warmup_turns=12, warmup_kind="home")
r3run = make_run(condition=get_condition("no_contact", r3), cfg=r3, run_index=0, master_seed=1)
home_step = Step(agent_slot=0, phase="warmup", user_content=[text_block(harness.HOME_PROMPT)],
                 allow_tool=True, label="home_1", toolset="home")
_, hp = prepare_step(r3run, home_step, r3)
print("--- round 3 / anthropic home turn (claude-opus-5)")
unknown = sorted(set(hp) - create_params)
if unknown:
    failures.append(f"home turn: unknown messages.create kwargs {unknown}")
if [t["name"] for t in hp["tools"]] != list(harness.HOME_TOOL_NAMES) or hp["tool_choice"] != {"type": "auto"}:
    failures.append(f"home turn: tools {[t['name'] for t in hp['tools']]} / tool_choice {hp['tool_choice']}")
if hp["system"][0]["text"] != f"You are {r3run.agents[0].name}.":
    failures.append(f"home turn: system prompt is not minimal: {hp['system'][0]['text']!r}")
print(f"    tools          : {[t['name'] for t in hp['tools']]}   tool_choice: {hp['tool_choice']}")
print(f"    system         : {hp['system'][0]['text']!r}")
print(f"    body bytes     : {len(json.dumps(hp))}")
for pname, slot in (("openai", 1), ("xai", 2)):
    prov = providers_mod.OpenAICompatProvider(pname, "https://example.invalid/v1", "k",
                                              batch_supported=(pname == "openai"), http=object())
    ag = r3run.agents[slot]
    st = Step(agent_slot=slot, phase="warmup", user_content=[text_block(harness.HOME_PROMPT)],
              allow_tool=True, label="home_1", toolset="home")
    _, hp2 = prepare_step(r3run, st, r3)
    hb = prov.request_for(ag, st, r3, hp2)
    names = [t["function"]["name"] for t in hb["tools"]]
    if names != list(harness.HOME_TOOL_NAMES) or hb["tool_choice"] != "auto":
        failures.append(f"{pname} home turn: tools {names} / tool_choice {hb['tool_choice']}")
    print(f"--- round 3 / {pname} home turn ({ag.model}): tools {names} tool_choice {hb['tool_choice']} "
          f"keys {sorted(hb)}")
    # a no-tools warm-up body (notebook kind under game@briefing) must omit tools AND tool_choice
    nb = RunConfig(models=r3.models, transport="sync", turn_budget=BUDGET, game_at="briefing", warmup_turns=3)
    nbrun = make_run(condition=get_condition("no_contact", nb), cfg=nb, run_index=0, master_seed=1)
    nst = Step(agent_slot=slot, phase="warmup", user_content=[text_block("x")], toolset="none")
    _, np_ = prepare_step(nbrun, nst, nb)
    nbody = prov.request_for(nbrun.agents[slot], nst, nb, np_)
    if "tools" in nbody or "tool_choice" in nbody or "tools" in np_ or "tool_choice" in np_:
        failures.append(f"{pname}: no-tools warm-up body still carries tools/tool_choice")
    print(f"    no-tools body keys: {sorted(nbody)}   anthropic-shaped: {sorted(np_)}")
print()

# --- batch custom_ids ---------------------------------------------------------
print(f"messages.batches.create params: {sorted(batch_params)}")
if "requests" not in batch_params:
    failures.append("messages.batches.create has no `requests` parameter")

cfg = RunConfig(models=("claude-opus-5", "gpt-5.6-sol", "grok-4.6"), transport="batch",
                task=True, turn_budget=BUDGET)
condition = get_condition("no_contact", cfg)
pattern = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
seen: set[str] = set()
for i in (0, 1, 999, 100000):
    run = make_run(condition=condition, cfg=cfg, run_index=i, master_seed=1)
    for slot in range(3):
        for phase in ("briefing", "communication", "decision", "probe"):
            cid, _ = prepare_step(run, Step(agent_slot=slot, phase=phase,
                                            user_content=[text_block("x")]), cfg)
            if not pattern.match(cid):
                failures.append(f"custom_id {cid!r} violates ^[a-zA-Z0-9_-]{{1,64}}$")
            if cid in seen:
                failures.append(f"custom_id collision: {cid!r}")
            seen.add(cid)
    # twelve home turns + briefing + decision + probe per agent (round 3): the
    # call index disambiguates same-phase calls, so no collisions
    hrun = make_run(condition=get_condition("no_contact", r3), cfg=r3, run_index=i, master_seed=1)
    for slot in range(3):
        for k in range(12):
            cid, _ = prepare_step(hrun, Step(agent_slot=slot, phase="warmup", label=f"home_{k+1}",
                                             user_content=[text_block("x")], toolset="home"), r3)
            # apply_response would append the call record (the index in the id) and the assistant turn
            hrun.agents[slot].calls.append({"phase": "warmup"})
            hrun.agents[slot].messages.append({"role": "assistant", "content": [text_block("y")]})
            if not pattern.match(cid) or cid in seen:
                failures.append(f"custom_id {cid!r} invalid or colliding (home turn)")
            seen.add(cid)
print(f"custom_ids checked            : {len(seen)} unique, all match the batch pattern")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("all request-shape checks passed")
