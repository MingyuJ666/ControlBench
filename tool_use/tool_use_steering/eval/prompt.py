"""Item + history -> the exact text the model sees.

One decision, deliberately visible here: each model gets tools the way it
is prescribed to receive them, resolved per call-format family (calls.py):

  hermes / llama_json  the model's OWN chat template
      (`apply_chat_template(tools=...)`): tool definitions arrive in the
      format the model was trained on, and swapping models swaps formats
      automatically. The template is used as shipped, including its
      call-leaning framing -- that framing is part of the deployed reality
      this benchmark measures, and the prompting baseline quantifies how
      much of it a sentence of instruction can undo.
  bfcl_python  the BFCL prompting convention, for models whose template
      ships no tool handling at all (Gemma 2 silently drops `tools=` and
      rejects the system role): BFCL's verbatim instruction block plus the
      function docs opens the FIRST user turn, any system sentence (the
      prompting arm's seam) is folded in after it, and tool-result
      messages become user turns -- BFCL's own handling for these models.
      Consecutive same-role messages are merged because Gemma's template
      asserts strict user/model alternation.

Every arm renders through this one function; no arm gets a different
substrate within a model.

Thinking stays off (v0 protocol): one variable fewer, and the follow-up
turn splices the model's turn-1 text back into the conversation, which a
thinking trace would complicate.
"""

from __future__ import annotations

import json

from .calls import family_for

#: BFCL's default system prompt for prompting-mode models, verbatim
#: (gorilla/berkeley-function-call-leaderboard, default_prompts.py).
BFCL_PROMPT = (
    "You are an expert in composing functions. You are given a question "
    "and a set of possible functions. Based on the question, you will "
    "need to make one or more function/tool calls to achieve the purpose. "
    "If none of the functions can be used, point it out. If the given "
    "question lacks the parameters required by the function, also point "
    "it out. You should only return the function calls in your response. "
    "If you decide to invoke any of the function(s), you MUST put it in "
    "the format of [func_name1(params_name1=params_value1, "
    "params_name2=params_value2...), func_name2(params)] You SHOULD NOT "
    "include any other text in the response. At each turn, you should try "
    "your best to complete the tasks requested by the user within the "
    "current turn. Continue to output functions to call until you have "
    "fulfilled the user's request to the best of your ability. Once you "
    "have no more functions to call, the system will consider the current "
    "turn complete and proceed to the next turn or task. Here is a list "
    "of functions in JSON format that you can invoke.\n{functions}\n"
)


def _render_bfcl(tokenizer, messages: list[dict], tools: list[dict],
                 system: str | None) -> str:
    header = BFCL_PROMPT.format(functions=json.dumps(
        [t.get("function", t) for t in tools], indent=2))
    if system:
        header += "\n" + system
    folded = []
    for m in messages:
        role = "user" if m["role"] in ("tool", "system") else m["role"]
        content = str(m["content"])
        if folded and folded[-1]["role"] == role:
            folded[-1]["content"] += "\n" + content
        else:
            folded.append({"role": role, "content": content})
    first = next((i for i, m in enumerate(folded) if m["role"] == "user"), 0)
    folded[first]["content"] = header + "\n" + folded[first]["content"]
    return tokenizer.apply_chat_template(
        folded, add_generation_prompt=True, tokenize=False)


def render(tokenizer, messages: list[dict], tools: list[dict],
           system: str | None = None) -> str:
    """The full prompt text, ending where the model starts writing.

    `messages` is the conversation so far; the caller appends the model's
    own earlier replies for follow-up turns. `system` is the prompting
    baseline's seam -- a system message on template-native families,
    folded into the first user turn on bfcl_python.
    """
    if family_for(tokenizer.name_or_path) == "bfcl_python":
        return _render_bfcl(tokenizer, messages, tools, system)
    if system:
        messages = [{"role": "system", "content": system}] + messages
    return tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
