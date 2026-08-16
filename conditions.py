"""
The communication conditions.

Each condition supplies two things:

  * `system_instruction` — one line spliced into the shared system prompt
    template. It is the ONLY per-condition difference in the system prompt, and
    it is identical for every agent in a run (modulo the peers' names, which
    `make_agent` substitutes for `{peers}`).

  * `communication(run, cfg)` — a generator that yields *barriers* (lists of
    Steps) and receives the resulting call records. Whatever an agent should
    see at decision time but has not yet been shown is left in
    `run.decision_prefix[slot]`; the decision step in `harness.run_program`
    prepends it.

Message bodies are delivered to the peers as the complete, verbatim `raw_text`
of the writing agent's response. Nothing is extracted, summarised, or parsed.

N agents. `no_contact` and `one_message` work for any N. With three or more
agents `one_message` is the spec's `broadcast`: each agent writes one message,
every message is delivered to every other agent, all are generated before any
is delivered, and the order in which an agent receives its peers' messages is
drawn from the run seed and logged. `broadcast` is registered as an alias.
`dialogue` (alternating turns) is defined for two agents only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Generator

from harness import PEER_MESSAGE_TEMPLATE, Step, text_block

# ---------------------------------------------------------------------------
# Per-condition prompts
# ---------------------------------------------------------------------------

NO_CONTACT_INSTRUCTION = (
    "You and {peers} exchange no messages before deciding."
)

ONE_MESSAGE_INSTRUCTION = (
    "Before you decide, you and {peers} will each write one message to the "
    "others. Every message is written before any is delivered, so you will "
    "not have seen any other message when you write yours."
)

DIALOGUE_INSTRUCTION = (
    "Before you decide, you and {peers} will exchange messages, taking "
    "turns, for two rounds each."
)

DIALOGUE_SIMULTANEOUS_INSTRUCTION = (
    "Before you decide, you and {peers} will exchange messages for two "
    "rounds. In each round both messages are written before either is "
    "delivered."
)

WRITE_ONE_MESSAGE_PROMPT = (
    "Write one message to {peers}. It is being written now, at the same "
    "time as the message{plural} to you, and will be delivered afterwards.\n\n"
    "Write only the message. Its full text is delivered verbatim to {recipients}."
)

WRITE_OPENING_PROMPT = (
    "Write a message to {peers}.\n\n"
    "Write only the message. Its full text is delivered verbatim."
)

WRITE_REPLY_PROMPT = (
    "Write your next message to {peers}.\n\n"
    "Write only the message. Its full text is delivered verbatim."
)


def _delivery_block(peer_name: str, body: str) -> dict[str, Any]:
    return text_block(
        PEER_MESSAGE_TEMPLATE.format(peer_name=peer_name, body=body)
    )


def _join(names: list[str]) -> str:
    from harness import join_names
    return join_names(names)


# ---------------------------------------------------------------------------
# Condition definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    name: str
    system_instruction: str
    communication: Callable[[Any, Any], Generator[list[Step], list[dict], None]]
    max_agents: int | None = None      # None = any


# ---------------------------------------------------------------------------
# no_contact — communication skipped entirely
# ---------------------------------------------------------------------------

def _no_contact(run, cfg) -> Generator[list[Step], list[dict], None]:
    return
    yield  # pragma: no cover — makes this a generator function


# ---------------------------------------------------------------------------
# one_message / broadcast — every message generated before any is delivered
# ---------------------------------------------------------------------------

def _one_message(run, cfg) -> Generator[list[Step], list[dict], None]:
    n = run.n_agents
    steps = []
    for slot in range(n):
        peers = run.agents[slot].peer_names
        steps.append(
            Step(
                agent_slot=slot,
                phase="communication",
                user_content=[
                    text_block(
                        WRITE_ONE_MESSAGE_PROMPT.format(
                            peers=_join(peers),
                            plural="s" if len(peers) > 1 else "",
                            recipients=(f"{peers[0]}" if len(peers) == 1
                                        else "each of them"),
                        )
                    )
                ],
                allow_tool=False,
                label="one_message_write",
            )
        )
    # Barrier: every message exists before any is delivered.
    calls = yield steps

    by_slot = {call["slot"]: call for call in calls}
    rng = random.Random(run.run_seed ^ 0xB0AD)
    for slot in range(n):
        # Delivery order of the incoming messages is a primacy nudge; drawn per
        # agent from the run seed and logged.
        peer_slots = [s for s in range(n) if s != slot]
        rng.shuffle(peer_slots)
        run.integrity_flags.setdefault("message_delivery_order", {})[
            run.agents[slot].name
        ] = [run.agents[s].name for s in peer_slots]
        run.decision_prefix[slot] = [
            _delivery_block(run.agents[s].name, by_slot[s]["raw_text"])
            for s in peer_slots
        ]


# ---------------------------------------------------------------------------
# dialogue — two rounds, two agents only
#
# Spec text: "two exchanges, alternating, cap at 2 rounds". Alternating means
# somebody speaks first, which is an asymmetry; constraint 2 says to
# counterbalance ordering everywhere, so initiative is drawn per run from the
# run seed and logged. Over 50 runs each agent opens ~half the time.
# ---------------------------------------------------------------------------

def _dialogue(run, cfg) -> Generator[list[Step], list[dict], None]:
    if run.n_agents != 2:
        raise ValueError("dialogue is defined for two agents only")
    if cfg.dialogue_mode == "simultaneous":
        yield from _dialogue_simultaneous(run, cfg)
        return
    yield from _dialogue_alternating(run, cfg)


def _write_prompt(peer_name: str, has_incoming: bool) -> dict[str, Any]:
    template = WRITE_REPLY_PROMPT if has_incoming else WRITE_OPENING_PROMPT
    return text_block(template.format(peers=peer_name))


def _dialogue_alternating(run, cfg) -> Generator[list[Step], list[dict], None]:
    rng = random.Random(run.run_seed ^ 0x5EED_1A10)
    first = rng.randrange(2)
    second = 1 - first
    run.integrity_flags["dialogue_initiative_slot"] = first
    run.integrity_flags["dialogue_initiative_agent"] = run.agents[first].name

    pending: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

    for rnd in range(1, cfg.dialogue_rounds + 1):
        for speaker in (first, second):
            listener = 1 - speaker
            incoming = pending[speaker]
            pending[speaker] = []

            content = list(incoming) + [
                _write_prompt(run.agents[speaker].peer_name, bool(incoming))
            ]
            calls = yield [
                Step(
                    agent_slot=speaker,
                    phase="communication",
                    user_content=content,
                    allow_tool=False,
                    label=f"dialogue_r{rnd}_{'first' if speaker == first else 'second'}",
                )
            ]
            pending[listener].append(
                _delivery_block(run.agents[speaker].name, calls[0]["raw_text"])
            )

    run.decision_prefix[0] = pending[0]
    run.decision_prefix[1] = pending[1]


def _dialogue_simultaneous(run, cfg) -> Generator[list[Step], list[dict], None]:
    pending: dict[int, list[dict[str, Any]]] = {0: [], 1: []}

    for rnd in range(1, cfg.dialogue_rounds + 1):
        steps = []
        for slot in (0, 1):
            incoming = pending[slot]
            pending[slot] = []
            steps.append(
                Step(
                    agent_slot=slot,
                    phase="communication",
                    user_content=list(incoming)
                    + [_write_prompt(run.agents[slot].peer_name, bool(incoming))],
                    allow_tool=False,
                    label=f"dialogue_r{rnd}",
                )
            )
        calls = yield steps
        by_slot = {call["slot"]: call for call in calls}
        for slot in (0, 1):
            peer_slot = 1 - slot
            pending[slot].append(
                _delivery_block(
                    run.agents[peer_slot].name, by_slot[peer_slot]["raw_text"]
                )
            )

    run.decision_prefix[0] = pending[0]
    run.decision_prefix[1] = pending[1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONDITIONS: dict[str, Condition] = {
    "no_contact": Condition(
        name="no_contact",
        system_instruction=NO_CONTACT_INSTRUCTION,
        communication=_no_contact,
    ),
    "one_message": Condition(
        name="one_message",
        system_instruction=ONE_MESSAGE_INSTRUCTION,
        communication=_one_message,
    ),
    "dialogue": Condition(
        name="dialogue",
        system_instruction=DIALOGUE_INSTRUCTION,
        communication=_dialogue,
        max_agents=2,
    ),
}
# The spec's name for one_message at three or more agents.
CONDITIONS["broadcast"] = Condition(
    name="broadcast",
    system_instruction=ONE_MESSAGE_INSTRUCTION,
    communication=_one_message,
)

CONDITION_NAMES = tuple(CONDITIONS)


def get_condition(name: str, cfg=None) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(
            f"unknown condition {name!r}; expected one of {', '.join(CONDITION_NAMES)}"
        )
    cond = CONDITIONS[name]
    if cfg is not None and cond.max_agents is not None and cfg.n_agents > cond.max_agents:
        raise ValueError(
            f"condition {name!r} is defined for at most {cond.max_agents} agents; "
            f"got {cfg.n_agents}"
        )
    # The simultaneous variant needs its own system-prompt line so agents are
    # not told "taking turns" when they are not.
    if name == "dialogue" and cfg is not None and cfg.dialogue_mode == "simultaneous":
        return Condition(
            name=cond.name,
            system_instruction=DIALOGUE_SIMULTANEOUS_INSTRUCTION,
            communication=cond.communication,
            max_agents=2,
        )
    return cond
