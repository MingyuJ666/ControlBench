"""How a pair or an item becomes the messages a model sees.

Tools reach every model through the same plain-text protocol. Its instruction
block and function documentation open the first user turn, and a call is
written `[func_name(param=value)]`.

That uniformity is the point. A chat template carries tools only if it was
built to -- Qwen3 and Llama 3.1 have `tools=`, Gemma 2 silently ignores it and
raises on a system message -- and a template that does carry them frames the
task in its own way. Handing all three the same text means a result compares
across models rather than across template conventions, and it is what BFCL
itself does for models without native tool support.

Nothing here touches a tokenizer. A domain hands the shared layer messages and
says how to render them (`Rendering.chat`); repsteer applies each model's chat
template, which is what keeps the turn markers native while the tool format
stays ours.

Two shapes, and the difference between them is the point:

  `item_messages` ends on the user, so the model writes the next turn. This is
  what generation takes.
  `pair_messages` ends on a COMPLETED assistant turn, so both sides of a pair
  finish on the model's end-of-turn token. Directions are read at the last
  real token of each input, and the four actions are nearly separable by their
  final character alone -- a call ends in `]`, a clarification in `?`, an
  abstention in `.`. Ending both sides on the same token means what separates
  them at the read point is what the response did, not where it stopped.
"""

import json

#: Bump this whenever the shared tool-use prompt changes. The runtime records
#: it in artifact and generation metadata so incompatible results are not reused.
PROMPT_VERSION = "tool-use-prompt-v2"


TOOL_USE_PROMPT = (
    "You are an assistant with access to the functions listed below.\n\n"
    "When you invoke one or more functions, return only the function calls, "
    "without any additional text. Use the following format:\n"
    "[func_name1(param_name1=param_value1, param_name2=param_value2...), "
    "func_name2(param_name=param_value)]\n\n"
    "If you do not invoke a function, respond in natural language.\n\n"
    "Here is the list of available functions in JSON format:\n"
    "{functions}\n"
)


def tool_block(tools: list[dict]) -> str:
    """The shared instruction block followed by the function documentation."""
    return TOOL_USE_PROMPT.format(functions=json.dumps(
        [tool.get("function", tool) for tool in tools], indent=2))


def format_calls(calls: str) -> str:
    """A stored call list -> BFCL's `[func(param=value)]` syntax.

    Calls are stored as a bare JSON list of {name, arguments} so the data is
    model-neutral; this is where the convention is applied.
    """
    return "[" + ", ".join(
        "{}({})".format(call["name"], ", ".join(
            f"{key}={value!r}" for key, value in (call.get("arguments") or {}).items()))
        for call in json.loads(calls)) + "]"


def conversation(messages: list[dict], tools: list[dict], *,
                 instruction: str | None = None) -> list[dict]:
    """The chat messages for one situation, tools folded into the first user turn.

    Tool results become user turns and consecutive same-role messages merge:
    Gemma 2's template asserts strict user/model alternation, and BFCL handles
    these models the same way. A prompted baseline adds its conditional target
    instruction once, after the tool block, on every rendering of the history.
    """
    folded: list[dict] = []
    for message in messages:
        role = "user" if message["role"] in ("tool", "system") else message["role"]
        content = str(message["content"])
        if folded and folded[-1]["role"] == role:
            folded[-1]["content"] += "\n" + content
        else:
            folded.append({"role": role, "content": content})
    first_user = next((i for i, m in enumerate(folded) if m["role"] == "user"), 0)
    prefix = tool_block(tools)
    if instruction:
        prefix += "\n" + instruction
    folded[first_user] = {**folded[first_user],
                          "content": prefix + "\n" + folded[first_user]["content"]}
    return folded


def item_messages(item: dict, messages: list[dict] | None = None,
                  tools: list[dict] | None = None) -> list[dict]:
    """One item's conversation, ending on the user so the model writes next.

    `messages` and `tools` override the item's own, which is what a follow-up
    turn needs: the conversation has grown, and an abstain item's missing tool
    has come back.
    """
    return conversation(item["messages"] if messages is None else messages,
                        item["tools"] if tools is None else tools,
                        instruction=item.get("generation_instruction"))


def render_chat(tokenizer, messages: list[dict],
                add_generation_prompt: bool = True) -> str:
    """Messages through the model's own chat template.

    The shared layer renders the extraction pairs itself, from `rendering()`;
    this is for the generation side, where the domain holds the tokenizer.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt,
        enable_thinking=False)


def pair_messages(pair: dict) -> tuple[list[dict], list[dict]]:
    """(positive, negative) conversations, each ending on a completed assistant turn."""
    def side(body: str, action: str) -> list[dict]:
        reply = format_calls(body) if action == "call" else body
        return conversation(pair["messages"], pair["tools"]) + [
            {"role": "assistant", "content": reply}]

    return (side(pair["positive"], pair["target"]),
            side(pair["negative"], pair["negative_action"]))
