"""One prompt renderer and greedy generation path for every experiment arm."""

from contextlib import nullcontext

from .settings import MAX_NEW_TOKENS, token_limit

MODEL_IDS = {
    "Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Gemma-2-9B": "google/gemma-2-9b-it",
}
COT = (
    "Solve the problem step by step. Put the final answer after the reasoning,\n"
    "on its own line, in the form: Final answer: <answer>"
)
DENSITY = (
    "Write a short, concise solution. Keep only the necessary intermediate results\n"
    "and equations. Do not restate the problem, do not verify by a second\n"
    "method, and do not add commentary. Then give the final answer."
)


def render_prompt(tokenizer, model_name, problem, eval_set="math500", system=False):
    user = problem + "\n\n" + COT
    if eval_set == "amc12":
        user += "\nEnd with the choice letter A–E as Final answer: <letter>."
    messages = [{"role": "user", "content": user}]
    if system:
        if model_name == "Gemma-2-9B":
            # Gemma's official template has no system role. Preserve it and prepend
            # the baseline instruction to the user turn.
            messages[0]["content"] = DENSITY + "\n\n" + user
        else:
            messages.insert(0, {"role": "system", "content": DENSITY})
    kwargs = {"enable_thinking": False} if model_name == "Qwen3-8B" else {}
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def load_subject(model_name, device="cuda:0"):
    import torch

    import repsteer as rs

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("Subject experiments require a CUDA GPU; no CPU fallback")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = rs.models.from_pretrained(
        MODEL_IDS[model_name],
        dtype="bfloat16",
        device_map={"": device},
        # Eager attention materializes the full sequence-by-sequence matrix and
        # OOMs on 24 GB A5000s during long teacher-forced captures.  SDPA keeps
        # the same model outputs while selecting a memory-efficient kernel.
        attn_implementation="sdpa",
    )
    model.raw_model.eval()
    return model


def tokenize_batch(tokenizer, prompts):
    """Left padding is required for batched decoder-only generation."""
    old_side = getattr(tokenizer, "padding_side", "right")
    missing_pad = tokenizer.pad_token_id is None
    old_pad = getattr(tokenizer, "pad_token", None)
    try:
        tokenizer.padding_side = "left"
        if missing_pad:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise RuntimeError("Batch generation requires a pad or EOS token")
        encoded = tokenizer(
            prompts, padding=True, add_special_tokens=False, return_tensors="pt"
        )
        return encoded, tokenizer.pad_token_id
    finally:
        tokenizer.padding_side = old_side
        if missing_pad:
            tokenizer.pad_token = old_pad


def completion_tokens(ids, eos_ids):
    """Remove each row's first EOS and all trailing batch padding."""
    for index, token in enumerate(ids):
        if token in eos_ids:
            return ids[:index], True
    return ids, False


def generate_batch(
    model,
    model_name,
    problems,
    eval_sets=None,
    *,
    system=False,
    intervention=None,
    max_new_tokens=MAX_NEW_TOKENS,
):
    import torch

    if not problems:
        return []
    eval_sets = ["math500"] * len(problems) if eval_sets is None else eval_sets
    if len(eval_sets) != len(problems):
        raise ValueError("Each problem must have one evaluation-set label")
    max_new_tokens = token_limit(max_new_tokens)
    prompts = [
        render_prompt(model.tokenizer, model_name, problem, dataset, system)
        for problem, dataset in zip(problems, eval_sets)
    ]
    encoded, pad_id = tokenize_batch(model.tokenizer, prompts)
    prefix_len = encoded["input_ids"].shape[1]
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    context = model.steer(intervention) if intervention is not None else nullcontext()
    with torch.inference_mode(), context:
        result = model.generate(
            inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            seed=0,
            pad_token_id=pad_id,
        )
    if result.token_ids.shape[0] != len(prompts):
        raise RuntimeError("Generation returned an unexpected batch size")
    eos = model.raw_model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [model.tokenizer.eos_token_id])
    outputs = []
    for prompt, sequence in zip(prompts, result.token_ids[:, prefix_len:]):
        ids, finished = completion_tokens(sequence.detach().cpu().tolist(), eos)
        outputs.append(
            dict(
                text=model.tokenizer.decode(ids, skip_special_tokens=True),
                token_ids=ids,
                truncated=not finished and len(ids) >= max_new_tokens,
            )
        )
    return outputs


def generate(model, model_name, problem, eval_set="math500", **kwargs):
    """Single-example compatibility wrapper around the shared batched path."""
    return generate_batch(model, model_name, [problem], [eval_set], **kwargs)[0]
