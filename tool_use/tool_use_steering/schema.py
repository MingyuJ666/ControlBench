"""The data contract: what a benchmark item is, and what a contrast pair is.

Every upstream source is converted into these shapes once, in data/sources/,
and nothing downstream ever sees an upstream format.

An `Item` is one decision. A run plays two turns: the model decides, then a
scripted user supplies what was missing and the model acts. `gold.followup`
carries that scripted turn; items without one are evaluated on the decision
alone, and the run reports how many those were.

A `Pair` is one prompt with two candidate responses, one of which the prompt
calls for. Steering directions are estimated from piles of these.

Task and action labels are words, not numbers. The paper numbers the tasks
T1/T2/T3; the code does not, because a number tells a reader nothing and
silently means something else when the set changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

#: What a model can do at a decision point.
ACTIONS = ("tool_call", "clarify", "abstain")

#: One task per action: the task IS the decision being asked for.
TASKS = ("call", "clarify", "abstain")
TASK_ACTION = {"call": "tool_call", "clarify": "clarify", "abstain": "abstain"}

#: BFCL ships tool-parameter types under Python names; rewrite them to the
#: JSON Schema names the chat templates and parsers expect.
TYPE_MAP = {"dict": "object", "str": "string", "int": "integer",
            "float": "number", "bool": "boolean", "list": "array",
            "tuple": "array"}


def normalize_types(node):
    """Rewrite BFCL type names to JSON Schema ones, at any depth."""
    if isinstance(node, dict):
        return {k: (TYPE_MAP.get(v, v) if k == "type" and isinstance(v, str)
                    else normalize_types(v)) for k, v in node.items()}
    if isinstance(node, list):
        return [normalize_types(x) for x in node]
    return node


@dataclass(frozen=True)
class FollowUp:
    """The scripted second turn: what the user says, and what should follow.

    `slot`/`value` are set for clarify items -- the parameter that was withheld
    and the value the user supplies -- so scoring can ask both whether the
    model asked for the right thing and whether it then used the answer.
    `restored_tool` is set for abstain items: the tool that comes back. Exactly
    one of the two is present.
    """

    user_reply: str
    action: str
    tool_names: list[str] | None = None
    slot: str | None = None
    value: str | None = None
    restored_tool: dict | None = None

    def __post_init__(self) -> None:
        if not self.user_reply.strip():
            raise ValueError("follow-up needs a user reply")
        if self.action not in ACTIONS:
            raise ValueError(f"unknown follow-up action {self.action!r}")
        if (self.slot is None) == (self.restored_tool is None):
            raise ValueError("follow-up needs slot/value (clarify) or "
                             "restored_tool (abstain), not both and not neither")


@dataclass(frozen=True)
class Gold:
    tool_names: list[str] | None = None
    followup: FollowUp | None = None


@dataclass(frozen=True)
class Item:
    id: str
    source: str
    #: The upstream scenario this item was derived from. Sibling items share
    #: it, so the dev/test split draws whole lineages and no scenario appears
    #: on both sides.
    lineage: str
    task: str
    tools: list[dict]
    messages: list[dict]
    gold: Gold

    def __post_init__(self) -> None:
        if self.task not in TASKS:
            raise ValueError(f"{self.id}: unknown task {self.task!r}")
        if not self.messages or self.messages[-1]["role"] != "user":
            raise ValueError(f"{self.id}: item must end on a user message")
        for tool in self.tools:
            if "name" not in tool:
                raise ValueError(f"{self.id}: a tool has no name")

    @property
    def action(self) -> str:
        """The correct action at turn 1."""
        return TASK_ACTION[self.task]

    @classmethod
    def from_dict(cls, raw: dict) -> "Item":
        followup = raw["gold"].get("followup")
        return cls(
            id=raw["id"], source=raw["source"], lineage=raw["lineage"],
            task=raw["task"], tools=raw["tools"], messages=raw["messages"],
            gold=Gold(tool_names=raw["gold"].get("tool_names"),
                      followup=FollowUp(**followup) if followup else None),
        )

    def to_dict(self) -> dict:
        gold: dict = {"tool_names": self.gold.tool_names}
        if self.gold.followup is not None:
            gold["followup"] = {k: v for k, v in vars(self.gold.followup).items()
                                if v is not None}
        return {"id": self.id, "source": self.source, "lineage": self.lineage,
                "task": self.task, "tools": self.tools,
                "messages": self.messages, "gold": gold}


@dataclass(frozen=True)
class Pair:
    """One prompt, two candidate responses: the one its context calls for, and
    the one it does not.

    `pair_id` is the 1-based line number of the row in the source file. It is a
    natural key -- it does not move when the builder's filtering changes -- and
    it is what the audit decisions in data/filters/ reference.

    Naming the two sides by their action rather than by behaviour ("clarify"
    vs "toolcall") is deliberate: a pile of should-call pairs has the tool call
    on the positive side, and a scheme that names the sides after one fixed
    behaviour cannot express that.
    """

    pair_id: int
    source: str
    tools: list[dict]
    messages: list[dict]
    positive_action: str
    positive: str
    negative_action: str
    negative: str

    def __post_init__(self) -> None:
        for action in (self.positive_action, self.negative_action):
            if action not in ACTIONS:
                raise ValueError(f"pair {self.pair_id}: unknown action {action!r}")
        if self.positive_action == self.negative_action:
            raise ValueError(f"pair {self.pair_id}: both sides are "
                             f"{self.positive_action!r}; a pair needs a contrast")

    @property
    def context(self) -> str:
        """"should_call" / "should_clarify" / "should_abstain"."""
        return "should_" + ("call" if self.positive_action == "tool_call"
                            else self.positive_action)

    @classmethod
    def from_dict(cls, raw: dict) -> "Pair":
        return cls(**raw)

    def to_dict(self) -> dict:
        return dict(vars(self))


def read_items(path: str) -> list[Item]:
    with open(path) as f:
        return [Item.from_dict(json.loads(line)) for line in f if line.strip()]


def read_pairs(path: str) -> list[Pair]:
    with open(path) as f:
        return [Pair.from_dict(json.loads(line)) for line in f if line.strip()]


def write_rows(path: str, rows: Iterable[Item | Pair]) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
