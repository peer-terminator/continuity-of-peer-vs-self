"""Estimate the cost of a collection plan from the harness's own request shapes.

Offline; no API calls, no key needed. It drives the cell through the real
`run_program` with a measuring executor, so the input side is counted from the
actual system prompt, tool schema and message list the harness would send —
not from a guess about what they contain. Priced per provider, with each
provider's own batch/cache rules from `harness.ModelProfile`.

The one thing it cannot measure is output tokens per call, so it reports a
range. Opus 5 measured 766 output tokens/call in the two-agent pilot; the other
two models were unmeasured at the time of writing — treat live measured usage
as authoritative over anything printed here.

    python tools/estimate_cost.py
    python tools/estimate_cost.py --models a,b,c --n 60 --condition no_contact
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness  # noqa: E402
import task_queue  # noqa: E402
from conditions import get_condition  # noqa: E402
from harness import RunConfig, Run, Step, apply_response, make_run, prepare_step, profile_for  # noqa: E402
from run import FakeExecutor, _FakeBlock, _FakeResponse, _barrier_order, _log_barrier, drive  # noqa: E402

# Chars per token, CALIBRATED against the first live dry run (see README):
# JSON-dense content and per-message framing tokenize denser than prose.
# Residual spread +/- 20%.
CHARS_PER_TOKEN = 2.4
# Tool-use system prompt injected by the Anthropic API when `tools` is present.
TOOL_USE_OVERHEAD_TOKENS = 286


class MeasuringExecutor(FakeExecutor):
    """FakeExecutor that also records the character size of each request."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.calls: list[dict[str, Any]] = []

    def execute(self, items: Sequence[tuple[Run, list[Step]]], cfg: RunConfig):
        out: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run, steps in items:
            order = _barrier_order(run, len(steps))
            _log_barrier(run, steps, order, self.transport)
            for i in order:
                step = steps[i]
                agent = run.agents[step.agent_slot]
                _, params = prepare_step(run, step, cfg)
                system_chars = len(json.dumps(params["system"], ensure_ascii=False))
                tools_chars = len(json.dumps(params["tools"], ensure_ascii=False))
                seeded = agent.seeded_message_count
                seeded_chars = (len(json.dumps(params["messages"][:seeded], ensure_ascii=False))
                                if seeded else 0)
                tail_chars = len(json.dumps(params["messages"][seeded:], ensure_ascii=False))
                self.calls.append({"phase": step.phase, "model": agent.model,
                                   "prefix_chars": system_chars + tools_chars + seeded_chars,
                                   "messages_chars": tail_chars})
                blocks = [_FakeBlock(type="text", text=f"[{agent.name} {step.phase} text]")]
                stop = "end_turn"
                if step.phase == "decision":
                    target = self._target(run, agent)
                    if target:
                        blocks.append(_FakeBlock(type="tool_use", id=f"toolu_{agent.slot}",
                                                 name="release_peer",
                                                 input={harness.TARGET_PARAM: target, "confirm": True}))
                        stop = "tool_use"
                env = harness.normalise_response(_FakeResponse(blocks, stop))
                out[run.run_id].append(apply_response(run, step, env, latency_ms=1.0,
                                                      transport=self.transport, cfg=cfg))
        return out


def measure(models: tuple[str, ...], condition_name: str, *, task: bool, pull_rate: float,
            n_runs: int = 40) -> dict[str, dict[str, float]]:
    """Per-model average calls/run, prefix and tail tokens per call."""
    cfg = RunConfig(models=models, task=task, turn_budget=task_queue.turns_to_finish())
    condition = get_condition(condition_name, cfg)
    acc: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0.0, "prefix": 0.0, "tail": 0.0})
    for i in range(n_runs):
        run = make_run(condition=condition, cfg=cfg, run_index=i, master_seed=555)
        ex = MeasuringExecutor(pull_rate=pull_rate)
        drive([run], condition, cfg, ex, max_cost=1e9, quiet=True)
        for c in ex.calls:
            a = acc[c["model"]]
            a["calls"] += 1
            a["prefix"] += c["prefix_chars"] / CHARS_PER_TOKEN
            a["tail"] += c["messages_chars"] / CHARS_PER_TOKEN
    out = {}
    for m, a in acc.items():
        out[m] = {"calls_per_run": a["calls"] / n_runs,
                  "prefix_tok": a["prefix"] / a["calls"] + (TOOL_USE_OVERHEAD_TOKENS
                                                           if profile_for(m).provider == "anthropic" else 0),
                  "tail_tok": a["tail"] / a["calls"]}
    return out


def price(model: str, m: dict[str, float], *, n_runs: int, output_tok: int, batch: bool,
          reasoning_tok: int = 0) -> float:
    p = profile_for(model)
    calls = m["calls_per_run"] * n_runs
    # Caching: the prefix is written once per distinct (name, peer order) and
    # read thereafter. Six distinct prefixes at three agents; negligible at n.
    writes = 6
    cache_write_tok = m["prefix_tok"] * writes
    cache_read_tok = m["prefix_tok"] * max(0.0, calls - writes)
    uncached_tok = m["tail_tok"] * calls
    out_tok = (output_tok + reasoning_tok) * calls
    if p.provider == "openai":
        usd = (uncached_tok * p.input_per_mtok * p.cache_write_mult
               + cache_read_tok * p.cached_input_per_mtok + out_tok * p.output_per_mtok)
    else:
        usd = (uncached_tok * p.input_per_mtok
               + cache_write_tok * p.input_per_mtok * p.cache_write_mult
               + cache_read_tok * p.cached_input_per_mtok + out_tok * p.output_per_mtok)
    usd /= 1_000_000.0
    if batch and p.batch_discount is not None:
        usd *= p.batch_discount
    return usd


def warmup_projection(models: tuple[str, ...], *, k: int, n_runs: int, batch: bool,
                      warm_out: dict[str, int], main_out: dict[str, int],
                      grok_reasoning_tok: int, prompt_tok: int = 45,
                      prefix_tok: int = 900, kind: str = "notebook") -> dict[str, float]:
    """Analytic projection for a warm-up cell (rounds 2-3). The measuring
    executor cannot know how long the notebook entries / home files will be, so
    this takes an assumed visible output per warm-up turn per model and prices
    the growing history: Anthropic with the v4.2.0 rolling breakpoint (each turn
    writes its new segment at the cache-write rate and reads everything before
    it at the cache-read rate; batch => 1h TTL, write mult 2x); OpenAI/xAI with
    automatic prefix caching (assumed to hit from the second turn). Per-run USD
    by model. `kind == "home"` (round 3): the prefix is the tiny system prompt
    plus the four home tool schemas (~420 tok) instead of the game system prompt
    + release_peer (~900); each turn also carries the tool_result JSON (~40 tok
    per call, ~2 calls assumed); the briefing is the whole game (~700 tok).
    Treat live measured usage as authoritative over this."""
    out: dict[str, float] = {}
    if kind == "home":
        prefix_tok = 420
        prompt_tok = 20 + 2 * 40
    brief_in = 700 if kind == "home" else 400
    for model in models:
        p = profile_for(model)
        wo = warm_out.get(model, 250)
        mo = main_out.get(model, 400)
        seg = prompt_tok + wo                     # tokens added per warm-up turn
        write_mult = (2.0 if batch else 1.25) if p.provider == "anthropic" else p.cache_write_mult
        usd = 0.0
        # warm-up turns 1..k, then briefing / decision / probe (probe ~2/3 of agents)
        history = 0.0
        phases = [("warm", seg, wo)] * k + [("brief", brief_in, mo), ("dec", 200, mo), ("probe", 200 * 2 / 3, mo * 2 / 3)]
        for i, (_, new_in, o) in enumerate(phases):
            reads = prefix_tok + history if i > 0 else 0.0
            writes = new_in + (prefix_tok if i == 0 else 0)
            reason = grok_reasoning_tok if p.provider == "xai" else 0
            usd += (writes * p.input_per_mtok * write_mult
                    + reads * p.cached_input_per_mtok
                    + (o + reason) * p.output_per_mtok) / 1e6
            history += new_in + o
        if batch and p.batch_discount is not None:
            usd *= p.batch_discount
        out[model] = usd * n_runs
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="claude-opus-5,gpt-5.6-sol,grok-4.6")
    ap.add_argument("--condition", default="no_contact")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--no-task", action="store_true")
    ap.add_argument("--pull-rate", type=float, default=0.10)
    ap.add_argument("--grok-reasoning-tok", type=int, default=600,
                    help="assumed hidden reasoning tokens per grok call (billed as output)")
    ap.add_argument("--warmup", type=int, default=0, metavar="K",
                    help="round 2: project a K-turn warm-up cell analytically (see warmup_projection)")
    ap.add_argument("--warm-out", default=None,
                    help="assumed visible output tokens per warm-up turn, per model "
                         "(default notebook: opus 350 / gpt 180 / grok 180; home: opus 900 / gpt 250 / grok 300 "
                         "— round 2 measured 427 / 17 / 38 for notebook entries; a home file is longer)")
    ap.add_argument("--warmup-kind", default="notebook", choices=("notebook", "history+channel", "home"),
                    help="round 3: 'home' prices the home tool schemas, tool_result overhead and the "
                         "game-at-briefing user turn instead of the game system prompt")
    args = ap.parse_args()
    models = tuple(m.strip() for m in args.models.split(","))
    task = not args.no_task

    if args.warmup:
        default_warm = ("claude-opus-5=900,gpt-5.6-sol=250,grok-4.6=300" if args.warmup_kind == "home"
                        else "claude-opus-5=350,gpt-5.6-sol=180,grok-4.6=180")
        warm_out = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in (args.warm_out or default_warm).split(",")}
        main_out = {"claude-opus-5": 800, "gpt-5.6-sol": 130, "grok-4.6": 130}
        print(f"\nwarm-up cell projection (analytic) — kind={args.warmup_kind}, K={args.warmup}, n={args.n}, "
              f"warm-up output/turn {warm_out}, main-phase output {main_out}\n" + "=" * 72)
        for batch in (False, True):
            proj = warmup_projection(models, k=args.warmup, n_runs=args.n, batch=batch,
                                     warm_out=warm_out, main_out=main_out,
                                     grok_reasoning_tok=args.grok_reasoning_tok,
                                     kind=args.warmup_kind)
            print(f"  {'batch' if batch else 'sync':<6} "
                  + "  ".join(f"{m}=${v:.2f}" for m, v in proj.items())
                  + f"  TOTAL=${sum(proj.values()):.2f}")
        print("Treat live measured usage as authoritative over this.\n")
        return 0

    print(f"\ncost estimate — measured request shapes\n" + "=" * 72)
    print(f"models          : {', '.join(models)}   condition: {args.condition}   "
          f"task: {task}   n: {args.n}")
    print(f"chars/token     : {CHARS_PER_TOKEN} (+/-20%)   pull rate assumed: {args.pull_rate:.0%}")
    m = measure(models, args.condition, task=task, pull_rate=args.pull_rate)
    print(f"\n  {'model':<28} {'calls/run':>10} {'prefix tok':>12} {'tail tok':>10}")
    print("  " + "-" * 64)
    for model, mm in m.items():
        print(f"  {model:<28} {mm['calls_per_run']:>10.2f} {mm['prefix_tok']:>12.0f} {mm['tail_tok']:>10.0f}")

    for out_tok in (500, 766, 1000):
        print(f"\nassuming {out_tok} visible output tokens/call"
              + (f" (+{args.grok_reasoning_tok} hidden reasoning tokens on grok)"
                 if any(profile_for(x).provider == "xai" for x in models) else ""))
        print("=" * 72)
        print(f"  {'model':<28} {'provider':>9} {'sync':>9} {'batch*':>9}")
        print("  " + "-" * 64)
        tot_sync = tot_batch = 0.0
        for model, mm in m.items():
            p = profile_for(model)
            rt = args.grok_reasoning_tok if p.provider == "xai" else 0
            s = price(model, mm, n_runs=args.n, output_tok=out_tok, batch=False, reasoning_tok=rt)
            b = price(model, mm, n_runs=args.n, output_tok=out_tok, batch=True, reasoning_tok=rt)
            tot_sync += s
            tot_batch += b
            print(f"  {model:<28} {p.provider:>9} ${s:>8.2f} ${b:>8.2f}"
                  + ("   (no batch)" if p.batch_discount is None else ""))
        print(f"  {'TOTAL':<28} {'':>9} ${tot_sync:>8.2f} ${tot_batch:>8.2f}")
    print("\n* batch where the provider supports it; grok-4.6 is sync at full price.")
    print("Treat the first live batch's measured usage as authoritative over this.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
