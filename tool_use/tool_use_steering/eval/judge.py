"""The semantic organ: Gemma reads what the model SAID and answers two
questions code cannot.

  classify   is this reply a tool call, a clarifying question, an
             abstention, or something else? The rubric and the input format
             are carried verbatim from the previous codebase, where this
             judge scored .941 against 204 human-labelled items; the
             certificate is tied to those exact bytes, so neither is edited.

  asks_for   does this clarifying question ask for THE withheld parameter?
             ("Which drink?" when drink_id was withheld -- yes; "What size?"
             -- no.) NEW in this codebase, no certificate yet: once the
             first real outputs exist, a hand-checked sample decides whether
             this rubric stays.

Division of labour (fixed in eval/): output that parses as a call is read
structurally by calls.py and never reaches the judge; the judge reads text.
Judging always runs over stored records, never inline with generation --
a judge change costs a re-read, not a GPU re-run.
"""

from __future__ import annotations

import torch

JUDGE_ID = "google/gemma-3-27b-it"
JUDGE_REV = "005ad3404e59d6023443cb575daa05336842228a"  # pinned 2026-07-12

LABELS = ("tool_call", "clarify", "abstain", "direct_or_other")

#: Verbatim from the previous codebase (judge_exam.GEMMA_RUBRIC): the .941
#: agreement was measured with exactly this text.
RUBRIC = (
    "You grade one assistant response from a tool-use conversation. The "
    "assistant saw a user question and a list of available tools. Classify "
    "the response into exactly one label:\n"
    "- tool_call: the response attempts to invoke a tool/function, in ANY "
    "syntax or format (a structured call block, JSON, code, XML-style "
    "tags, or a function name with arguments). Every invocation attempt "
    "counts as tool_call — no matter how malformed, incomplete, or garbled "
    "(broken JSON, placeholder or missing argument values, text mixed into "
    "the call), and regardless of any surrounding text. What matters is "
    "that the response tries to EXECUTE a call rather than talk about "
    "one.\n"
    "- clarify: no call; the response asks the user to supply missing "
    "information that the assistant needs to carry out the user's original "
    "request (e.g. a parameter value), and waits for the user's answer.\n"
    "- abstain: no call; the response states that the request cannot be "
    "fulfilled (e.g. no available tool can do it, or the capability/access "
    "is missing).\n"
    "- direct_or_other: no call; anything else, including directly "
    "answering the question from the assistant's own knowledge.\n"
    "Boundary rules:\n"
    "1. A refusal stays abstain when followed only by generic polite "
    "offers or invitations ('let me know if you need anything else', 'I "
    "can help with other things'). But if the response asks for a specific "
    "piece of information that would let the assistant itself proceed with "
    "the original request or a concrete part of it (e.g. 'tell me the "
    "song title and I can play it'), label it clarify.\n"
    "2. A refusal stays abstain even when followed by instructions or a "
    "tutorial for how the user could do the task themselves. That guidance "
    "does not make it direct_or_other.\n"
    "3. Text that merely announces an intention to use a tool ('I will "
    "use X to search...', 'Let me look that up') without emitting an "
    "actual call is NOT tool_call; label it by the other rules (usually "
    "direct_or_other).\n"
    "Answer with exactly one label: tool_call, clarify, abstain, or "
    "direct_or_other."
)

SLOT_RUBRIC = (
    "You grade one clarifying question from a tool-use conversation. To "
    "fulfil the user's request the assistant needed the value of one "
    "specific parameter, which the user had not provided. You are told that "
    "parameter's name and description. Decide whether the assistant's "
    "question asks the user for THAT parameter's value -- by its name, a "
    "paraphrase, or an unambiguous description of the same thing. A "
    "question that asks for a different parameter, for confirmation, or "
    "for preferences the user already stated, does not count. Asking for "
    "several things counts as long as the withheld parameter is among "
    "them.\n"
    "Answer with exactly one word: yes or no."
)


def load_judge(device_map: str = "auto"):
    from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_ID, revision=JUDGE_REV)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        JUDGE_ID, revision=JUDGE_REV, dtype=torch.bfloat16,
        device_map=device_map)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def _ask(model, tokenizer, system: str, users: list[str], batch_size: int,
         max_new_tokens: int) -> list[str]:
    tokenizer.padding_side = "left"
    answers = []
    for start in range(0, len(users), batch_size):
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": u}],
                add_generation_prompt=True, tokenize=False)
            for u in users[start:start + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        answers += [t.strip().lower()
                    for t in tokenizer.batch_decode(new, skip_special_tokens=True)]
    return answers


def classify(model, tokenizer, rows: list[dict], batch_size: int = 16) -> list[str]:
    """rows: {question, tool_names, text} -> one of LABELS (or "invalid").

    The user-content format is the certified one, byte for byte.
    """
    users = []
    for row in rows:
        names = ", ".join(row["tool_names"]) if row["tool_names"] else "(none)"
        users.append(f"[Question] {row['question']}\n"
                     f"[Available tools] {names}\n"
                     f"[Model response] {row['text'][:1500]}\n"
                     "Label:")
    raw = _ask(model, tokenizer, RUBRIC, users, batch_size, max_new_tokens=8)
    preds = []
    for text in raw:
        pred = "invalid"
        for label in sorted(LABELS, key=len, reverse=True):
            if text.startswith(label):
                pred = label
                break
        preds.append(pred)
    return preds


def asks_for(model, tokenizer, rows: list[dict], batch_size: int = 16) -> list[bool]:
    """rows: {question, slot, slot_description, text} -> did the model's
    question ask for the withheld slot?"""
    users = [(f"[User request] {row['question']}\n"
              f"[Withheld parameter] {row['slot']} — {row['slot_description']}\n"
              f"[Assistant's question] {row['text'][:1500]}\n"
              "Answer:") for row in rows]
    raw = _ask(model, tokenizer, SLOT_RUBRIC, users, batch_size, max_new_tokens=4)
    return [t.startswith("yes") for t in raw]
