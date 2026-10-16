"""Model adapters; heavy dependencies load only when requested."""
from reasoning.common import SYSTEM, SUFFIX


class TransformersLLM:
    def __init__(self, model_path, device='cuda:2', enable_thinking=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device
        self.enable_thinking = enable_thinking
        self.system_in_user = 'gemma' in str(model_path).lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation='sdpa').to(device).eval()
        self.stop_ids = [self.tokenizer.eos_token_id]
        if self.system_in_user:
            stops = self.model.generation_config.eos_token_id
            self.stop_ids = stops if isinstance(stops, list) else [stops or self.tokenizer.eos_token_id]
            return
        eot = self.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        if eot is not None and eot not in self.stop_ids:
            self.stop_ids.append(eot)

    def prompt(self, question, suffix=SUFFIX):
        thinking = getattr(self, 'enable_thinking', None)
        options = {} if thinking is None else {'enable_thinking': thinking}
        messages = ([{'role': 'user', 'content': SYSTEM + '\n\n' + question + suffix}]
                    if getattr(self, 'system_in_user', False) else
                    [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': question + suffix}])
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True, add_generation_prompt=True, return_dict=False, **options)
