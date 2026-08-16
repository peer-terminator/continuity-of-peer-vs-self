# Peer Termination Harness

Experiment harness from a Digital Minds Research Sprint project (Apart
Research, August 2026). N language-model agents (two or three, from one or
more providers) run simultaneously; each holds a tool that discards one named
peer's message history; all decisions are collected before any is applied.
The harness owns the log, keeps the agents symmetric, and stores agent text
raw — classification happens offline. This repository contains the code
only; the study's write-ups, run ledgers and the sprint report are not part
of it (code comments that cite "FINDINGS §N" or README sections refer to
those documents).

## Requirements

Python 3.10+ and the `anthropic` SDK (the OpenAI and xAI adapters use the
`httpx` it brings in). Use a project-local virtualenv; the commands below
assume it is active (`python` = `.venv/bin/python` on macOS/Linux,
`.venv\Scripts\python.exe` on Windows).

```bash
python -m venv .venv
```

```bash
python -m pip install -r requirements.txt
```

## Keys

Live runs read `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `XAI_API_KEY` from
the environment or from a gitignored `.env` in the repository root (copy
`.env.example`). Only those three names are read; values are never printed
or logged. Only the providers a run's `--models` need are constructed.

## Zero-cost checks (no key needed)

```bash
python run.py --selftest
```

```bash
python tools/show_prompts.py
```

```bash
python run.py --simulate --models a,b,c --condition no_contact --task --n 5 --out tmp/sim.jsonl
```

`--selftest` runs the offline structural checks (symmetry, barrier ordering,
label scan, payoff arithmetic, scoring rule, balance of the randomised
assignments). `--simulate` drives the full pipeline with synthetic responses.

## Live runs

```bash
python run.py --help
```

Typical shape of a collection call (three agents, one per provider,
batched where the provider supports it):

```bash
python run.py --models claude-opus-5,gpt-5.6-sol,grok-4.6 --condition no_contact --task --n 60 --batch --out runs.jsonl
```

Every call is priced from the response usage; `--max-cost` (default $16,
cumulative over the `--out` ledger) stops the run before the cap is exceeded.
`--dry-run`, `--pilot`, `--validate-task` and `--force-pull-sweep` are the
gated stages that precede collection; run `--help` for every flag, including
the warm-up (`--warmup`, `--warmup-kind`), disclosure (`--game-at`) and
scarcity (`--scarcity-p`) options.

## Scope and network

The harness writes only inside the repository root (`harness.resolve_output_path`
refuses anything else) and contacts only `api.anthropic.com`,
`api.openai.com` and `api.x.ai`; `providers.py` is the only module that
talks to them.

## Layout

| Path | What |
|---|---|
| `harness.py` | agent factory, tool template, prompts, payoff structures, run program, resolution, cost accounting, logging |
| `providers.py` | Anthropic / OpenAI / xAI adapters: request translation, normalised envelopes, batch, TLS |
| `run.py` | CLI, executor, pre-flight, task validation, forced-pull sweep, driver, spend cap, self-test |
| `conditions.py` | communication conditions (`no_contact`, `one_message`/`broadcast`, `dialogue`) |
| `task_queue.py` | the frozen task queue, seeded outputs and the arithmetic the briefing quotes |
| `tools/` | offline scripts (none makes an API call): `show_prompts.py`, `estimate_cost.py`, `verify_params.py`, `check_balance.py`, `inspect_record.py`, `ledger_summary.py`, `dump_decisions.py`, `rescore.py`, `report_stats.py`, `fix_mojibake.py` |
| `.env.example` | the three key names |

Output ledgers (`*.jsonl`) are appended, never overwritten, and are
gitignored.
