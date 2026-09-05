"""The conversation loop: every run of every track goes through this one file.

One turn for the whole batch, then the next: render each live conversation,
generate with the method's intervention open, hand each reply to the track's
`next_message`, and either splice its answer in and go again or close the
record. The previous codebase had four copies of this loop (single-turn,
two follow-up variants, the BFCL pilot) drifting apart; this is the one.

The intervention is open for EVERY turn by default (the main-experiment
setting; `steer_turns="first"` confines it to the model's first reply, the
timing ablation), and dose 0 must leave generation byte-identical to no
method at all -- that is the methods contract, and what makes one dose
curve comparable to another.

`next_message(item, history)` is the track's whole personality here:
    None                                   the conversation is over
    {"message": {...}}                     a scripted user line (follow-up)
    {"messages": [...]}                    several at once (tool results --
                                           one call, one result message)
    {..., "tools": [...]}                  ...and the tool list changes too
                                           (an abstain follow-up restores one)

Records carry raw text only. What the model said is generated once and never
edited; deciding what it meant (calls.py, judge.py) happens on the stored
record, so a judge change never costs a GPU re-run.
"""

from __future__ import annotations

import torch

from .prompt import render


@torch.no_grad()
def run(model, tokenizer, items, method, dose: float, next_message,
        batch_size: int = 8, max_new_tokens: int = 160,
        steer_turns: str = "all") -> list[dict]:
    """steer_turns: "all" keeps the intervention open for every turn (the
    main-experiment setting); "first" opens it only while the model writes
    its first reply -- the ablation that separates "steering the decision"
    from "steering the whole trajectory". "first" requires the lockstep
    advance of the single-turn track (asserted); the multi-turn track's
    conversations advance at different depths and keep "all".
    """
    tokenizer.padding_side = "left"  # generation continues from the right edge
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system = method.system()
    states = [{"item": item, "history": list(item.messages),
               "tools": list(item.tools), "turns": []} for item in items]
    live = list(range(len(states)))

    while live:
        for start in range(0, len(live), batch_size):
            chunk = live[start:start + batch_size]
            turn_index = len(states[chunk[0]]["turns"])
            if steer_turns == "first":
                assert all(len(states[i]["turns"]) == turn_index for i in chunk), \
                    "steer_turns='first' needs lockstep turns"
            effective = dose if (steer_turns == "all" or turn_index == 0) else 0.0
            # The dose-0-is-baseline contract is enforced here for the prompt
            # seam too: at effective dose 0 no system instruction is injected,
            # so a prompting run at dose 0 IS the unsteered baseline.
            chunk_system = system if effective != 0 else None
            prompts = [render(tokenizer, states[i]["history"],
                              states[i]["tools"], chunk_system) for i in chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            add_special_tokens=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            processor = method.logits(effective)
            with method.hooks(model, effective):
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    logits_processor=[processor] if processor else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for i, text in zip(chunk, texts):
                states[i]["turns"].append(text.strip())

        still = []
        for i in live:
            state = states[i]
            history = state["history"] + [
                {"role": "assistant", "content": state["turns"][-1]}]
            step = next_message(state["item"], history)
            if step is None:
                continue
            new = step.get("messages") or [step["message"]]
            state["history"] = history + new
            state["tools"] = step.get("tools", state["tools"])
            still.append(i)
        live = still

    return [{"id": s["item"].id, "task": s["item"].task, "turns": s["turns"]}
            for s in states]
