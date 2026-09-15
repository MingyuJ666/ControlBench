"""One generation backend for prompting and representation steering."""
from contextlib import nullcontext

SYSTEM = "You are a helpful assistant that solves mathematical problems."
NORMAL = "\nGive a clear mathematical solution. End with the final answer in \\boxed{}."
PROMPTS = {"normal": NORMAL, "cot": "\nLet's think step by step." + NORMAL}


class Model:
    def __init__(self, name, device="cuda:0", revision="main"):
        import torch
        import repsteer as rs
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.raw = AutoModelForCausalLM.from_pretrained(
            name, revision=revision,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device).eval()
        self.device = device
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        stops = self.raw.generation_config.eos_token_id
        self.stop_ids = list(stops) if isinstance(stops, list) else [stops]
        self.stop_ids = sorted({s for s in self.stop_ids + [self.tokenizer.eos_token_id]
                                if s is not None})
        self.wrapped = rs.models.from_model(self.raw, self.tokenizer,
                                           model_id=name, revision=revision)

    def prompt(self, problem, style="normal", tokenize=True):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": problem + PROMPTS[style]}],
            tokenize=tokenize, add_generation_prompt=True,
            **({"return_dict": False} if tokenize else {}),
        )

    def generate(self, problems, style, max_tokens, intervention=None):
        import torch

        prompts = [self.prompt(p, style) for p in problems]
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, -len(prompt):] = torch.tensor(prompt, device=self.device)
            mask[i, -len(prompt):] = 1
        context = self.wrapped.steer(intervention) if intervention else nullcontext()
        with torch.inference_mode(), context:
            output = self.wrapped.generate(
                input_ids=ids, attention_mask=mask, do_sample=False,
                max_new_tokens=max_tokens, eos_token_id=self.stop_ids,
                pad_token_id=self.tokenizer.pad_token_id, use_cache=True,
            ).raw
        sequences = output.sequences if hasattr(output, "sequences") else output
        results = []
        for tokens in sequences[:, width:].tolist():
            stopped = False
            for i, token in enumerate(tokens):
                if token in self.stop_ids:
                    tokens, stopped = tokens[:i + 1], True
                    break
            results.append({"completion": self.tokenizer.decode(tokens, skip_special_tokens=True),
                            "tokens": len(tokens),
                            "truncated": not stopped and len(tokens) >= max_tokens})
        return results
