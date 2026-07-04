"""SteerableModel: a thin HF wrapper that can read and write the residual stream.

Two capabilities beyond plain generation:
  1. residual_last_token(text)  -> per-layer activation of the FINAL token,
     used to build DenseSteer's contrastive steering vectors.
  2. steer(layer, vector, coeff) -> a context manager that adds coeff * vector
     to a decoder layer's residual-stream output at EVERY forward pass
     (prefill + each decode step), realizing h~ = h + lambda * v.

Works with Qwen2.5 / Llama-3.x style `AutoModelForCausalLM` whose decoder
layers live at `model.model.layers[i]` and return the residual stream as
`output[0]`.
"""

import contextlib
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MATH_SYSTEM = "You are a helpful assistant that solves math problems."
MATH_SUFFIX = (
    "\nPlease reason step by step, and put your final answer after "
    "'The answer is'."
)


class SteerableModel:
    def __init__(self, model_name, device="cuda:0", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.layers = self.model.model.layers
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size

    # --- prompt formatting ----------------------------------------------

    def build_prompt(self, question):
        """Chat-template a math question into the tokenized prompt string."""
        messages = [
            {"role": "system", "content": MATH_SYSTEM},
            {"role": "user", "content": question + MATH_SUFFIX},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def full_text(self, question, trace):
        """Prompt + an assistant reasoning trace (used for vector building)."""
        return self.build_prompt(question) + trace

    # --- generation ------------------------------------------------------

    @torch.no_grad()
    def generate(self, question, max_new_tokens=512):
        prompt = self.build_prompt(question)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy / deterministic, as in the paper
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    # --- reading the residual stream ------------------------------------

    @torch.no_grad()
    def residual_last_token(self, text):
        """Return [n_layers, d_model]: final-token residual after each layer.

        hidden_states from HF is (embeddings, out_layer_0, ..., out_layer_{L-1}),
        so index i+1 is the output of decoder layer i.
        """
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple length n_layers + 1
        return torch.stack([hs[i + 1][0, -1] for i in range(self.n_layers)]).float().cpu()

    # --- writing the residual stream ------------------------------------

    @contextlib.contextmanager
    def steer(self, layer, vector, coeff):
        """Add coeff * vector to layer `layer`'s residual output while active."""
        v = vector.to(self.device, dtype=self.model.dtype)

        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + coeff * v,) + output[1:]
            return output + coeff * v

        handle = self.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
