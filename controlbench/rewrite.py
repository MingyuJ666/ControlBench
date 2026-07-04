"""Dense-Rewriting: turn a sparse reasoning trace into a denser positive one.

Given the small model's own trace (x_neg), a stronger model rewrites it to be
"denser" -- adjacent trivial steps merged, higher information per step -- while
preserving every necessary logical move AND the final answer. The (x_pos, x_neg)
pair is the raw material for DenseSteer's steering vector.

Rewriter is the OpenAI API. Set OPENAI_API_KEY in the environment.
"""

import os
from openai import OpenAI

REWRITE_SYSTEM = (
    "You rewrite mathematical reasoning to be DENSER without changing its "
    "meaning or its final answer. Dense reasoning uses fewer, higher-information "
    "steps: merge adjacent trivial or redundant steps, drop filler and restated "
    "context, and keep each step doing real mathematical work. Preserve every "
    "logically necessary step and the exact final numeric answer. Keep the same "
    "natural style the model would produce (do not add headers, markdown, or "
    "meta commentary). Output ONLY the rewritten reasoning."
)

REWRITE_USER = (
    "Problem:\n{question}\n\n"
    "Original reasoning to make denser:\n{trace}\n\n"
    "Rewrite it as dense reasoning ending with 'The answer is <number>'."
)


def make_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Run:  export OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=key)


def dense_rewrite(client, question, trace, model="gpt-4o", temperature=0.3):
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": REWRITE_USER.format(
                question=question, trace=trace)},
        ],
    )
    return resp.choices[0].message.content.strip()
