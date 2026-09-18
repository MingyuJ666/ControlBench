"""The scripted second turn.

An item that withheld something scripts what happens once it is supplied: for
a clarify item the user gives the argument that was missing, for an abstain
item the tool that could not be found comes back, announced in the user's own
line and present in the tool list from then on. Call items script nothing --
they are complete at the decision.

The conversation the model sees on the second turn is the first prompt, its
own reply, and the user's line; the tool block is rebuilt from the tool list
as it now stands, so a restored tool is both announced and documented.
"""

from ..render import item_messages, render_chat


def next_turn(tokenizer, item: dict, turns: list[str]) -> str | None:
    """The prompt for this item's next turn, or None when it is over."""
    followup = item["gold"].get("followup")
    if followup is None or len(turns) >= 2:
        return None
    messages = item["messages"] + [
        {"role": "assistant", "content": turns[-1]},
        {"role": "user", "content": followup["user_reply"]}]
    tools = item["tools"]
    if followup.get("restored_tool"):
        tools = tools + [followup["restored_tool"]]
    return render_chat(tokenizer, item_messages(item, messages, tools))


def check_second(item: dict, text: str):
    """The structural verdict on a second turn: a call to the tool the
    follow-up names, under the tool list as it stands after the follow-up."""
    from .. import calls
    followup = item["gold"].get("followup") or {}
    tools = item["tools"]
    if followup.get("restored_tool"):
        tools = tools + [followup["restored_tool"]]
    return calls.check(text, tools, followup.get("tool_names"))
