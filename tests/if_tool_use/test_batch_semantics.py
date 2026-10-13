"""Check batch identities and steering positions independently of output equality."""

import unittest

import torch
import repsteer as rs
from transformers import Qwen3Config, LlamaConfig, Gemma2Config

import core as cb
from core.domains import instruction_following as domain
from core.methods import METHODS
from test_native_methods import tiny_wrapper, training_pairs


ARCHITECTURES = (Qwen3Config, LlamaConfig, Gemma2Config)


def chats(n):
    texts = ['answer red', 'answer ' + 'blue green ' * 12]
    texts.extend('answer ' + ('yellow black ' * (i % 9 + 1)) for i in range(n - 2))
    return [[{'role': 'user', 'content': text}] for text in texts[:n]]


def encode(wrapper, messages):
    rendered = wrapper.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return wrapper.tokenizer(rendered, padding=True, add_special_tokens=False,
                             return_tensors='pt', return_token_type_ids=False)


def run(wrapper, messages, plan=None, max_new=4):
    return cb.generate(wrapper, messages, plan, max_new_tokens=max_new,
                       output_scores=True, **domain.rendering().generation_kwargs())


class BatchSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_wrapper_matches_raw_generate_with_identical_encoded_inputs(self):
        for architecture in ARCHITECTURES:
            wrapper = tiny_wrapper(architecture, num_hidden_layers=2, num_key_value_heads=1)
            try:
                for batch_size in (1, 3, 48):
                    with self.subTest(model=architecture.__name__, batch=batch_size):
                        messages = chats(batch_size)
                        encoded = encode(wrapper, messages)
                        captured = []
                        def pre(module, args, kwargs):
                            captured.append({key: kwargs[key].detach().clone() for key in
                                             ('input_ids', 'attention_mask', 'position_ids')})
                        handle = wrapper.model.register_forward_pre_hook(pre, with_kwargs=True)
                        try:
                            actual = run(wrapper, messages)
                        finally:
                            handle.remove()
                        self.assertTrue(torch.equal(captured[0]['input_ids'], encoded['input_ids']))
                        self.assertTrue(torch.equal(captured[0]['attention_mask'], encoded['attention_mask']))
                        positions = (encoded['attention_mask'].cumsum(-1) - 1).masked_fill(
                            encoded['attention_mask'] == 0, 0)
                        self.assertTrue(torch.equal(captured[0]['position_ids'], positions))
                        for step, entry in enumerate(captured[1:], 1):
                            self.assertTrue(torch.equal(entry['position_ids'][:, -1],
                                                       encoded['attention_mask'].sum(-1) + step - 1))
                        expected = wrapper.model.generate(
                            **encoded, do_sample=False, temperature=None, top_p=None, top_k=None,
                            max_new_tokens=4, pad_token_id=wrapper.tokenizer.pad_token_id,
                            return_dict_in_generate=True, output_scores=True)
                        self.assertTrue(torch.equal(actual.sequences, expected.sequences))
                        for a, b in zip(actual.scores, expected.scores, strict=True):
                            self.assertTrue(torch.equal(a, b))
            finally:
                wrapper.close()

    def test_other_rows_do_not_change_target_at_fixed_shape(self):
        """Change other rows' tokens while preserving every tensor shape and mask."""
        for architecture in ARCHITECTURES:
            wrapper = tiny_wrapper(architecture, num_hidden_layers=2, num_key_value_heads=1)
            # Keep both runs equally long; pad tokens after EOS are not an answer.
            wrapper.model.generation_config.eos_token_id = None
            try:
                art = METHODS['caa'].learn(wrapper, domain, 'lowercase', [rs.sites.resid_post(0)],
                                           batch_size=3, pairs=training_pairs())['resid_post_0']
                for strength in (0., .5, 2.):
                    with self.subTest(model=architecture.__name__, strength=strength):
                        original = encode(wrapper, chats(48))
                        changed = {key: value.clone() for key, value in original.items()}
                        ids = changed['input_ids'][1:]
                        ids[ids == wrapper.tokenizer.convert_tokens_to_ids('blue')] = wrapper.tokenizer.convert_tokens_to_ids('white')
                        ids[ids == wrapper.tokenizer.convert_tokens_to_ids('yellow')] = wrapper.tokenizer.convert_tokens_to_ids('black')
                        plan = METHODS['caa'].plan(art, {'strength': strength})
                        a = cb.generate(wrapper, original, plan, max_new_tokens=4, output_scores=True)
                        b = cb.generate(wrapper, changed, plan, max_new_tokens=4, output_scores=True)
                        self.assertTrue(torch.equal(a.sequences[0], b.sequences[0]))
                        for sa, sb in zip(a.scores, b.scores, strict=True):
                            self.assertTrue(torch.equal(sa[0], sb[0]))
            finally:
                wrapper.close()

    def test_caa_applies_token_local_formula_at_expected_positions(self):
        for architecture in ARCHITECTURES:
            wrapper = tiny_wrapper(architecture, num_hidden_layers=2, num_key_value_heads=1)
            wrapper.model.generation_config.eos_token_id = None
            try:
                art = METHODS['caa'].learn(wrapper, domain, 'lowercase', [rs.sites.resid_post(0)],
                                           batch_size=3, pairs=training_pairs())['resid_post_0']
                for batch_size in (1, 3, 48):
                    with self.subTest(model=architecture.__name__, batch=batch_size):
                        before, after, phases = [], [], []
                        layer = wrapper.model.model.layers[0]
                        def read(output):
                            return output if isinstance(output, torch.Tensor) else output[0]
                        h1 = layer.register_forward_hook(lambda m, a, o: before.append(read(o).detach().clone()))
                        plan = METHODS['caa'].plan(art, {'strength': 2.})
                        try:
                            with wrapper.steer(plan):
                                def observe(m, a, o):
                                    after.append(read(o).detach().clone())
                                    phases.append(wrapper.generation_tracker.current.phase)
                                h2 = layer.register_forward_hook(observe)
                                try:
                                    run(wrapper, chats(batch_size), max_new=4)
                                finally:
                                    h2.remove()
                        finally:
                            h1.remove()
                        self.assertEqual(phases, ['prefill', 'decode', 'decode', 'decode'])
                        for original, steered in zip(before, after, strict=True):
                            expected = original.clone()
                            direction = art.direction.to(original)
                            for row in range(batch_size):
                                token = original[row, -1]
                                alpha = (2. * torch.linalg.vector_norm(token.float())).to(token)
                                expected[row, -1] = token + alpha * direction
                            self.assertTrue(torch.equal(steered, expected))
            finally:
                wrapper.close()

    def test_max_new_only_changes_the_termination_boundary(self):
        for architecture in ARCHITECTURES:
            wrapper = tiny_wrapper(architecture, num_hidden_layers=2, num_key_value_heads=1)
            wrapper.model.generation_config.eos_token_id = None
            try:
                art = METHODS['caa'].learn(wrapper, domain, 'lowercase', [rs.sites.resid_post(0)],
                                           batch_size=3, pairs=training_pairs())['resid_post_0']
                messages = chats(3)
                for strength in (0., .5, 2.):
                    with self.subTest(model=architecture.__name__, strength=strength):
                        plan = METHODS['caa'].plan(art, {'strength': strength})
                        short, long = run(wrapper, messages, plan, 4), run(wrapper, messages, plan, 12)
                        self.assertTrue(torch.equal(short.sequences, long.sequences[:, :short.sequences.shape[1]]))
                        for a, b in zip(short.scores, long.scores):
                            self.assertTrue(torch.equal(a, b))
            finally:
                wrapper.close()

    def test_output_slicing_and_sorting_preserve_item_identity(self):
        for architecture in ARCHITECTURES:
            wrapper = tiny_wrapper(architecture, num_hidden_layers=2, num_key_value_heads=1)
            try:
                messages = chats(5)
                texts = wrapper.tokenizer.apply_chat_template(messages, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                actual = cb.generate_texts(wrapper, texts, batch_size=3, max_new_tokens=4,
                                            messages=messages, rendering=domain.rendering())
                order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
                for offset in range(0, len(order), 3):
                    indices = order[offset:offset + 3]
                    batch = [messages[i] for i in indices]
                    encoded = encode(wrapper, batch)
                    result = run(wrapper, batch)
                    prompt_length = encoded['input_ids'].shape[1]
                    for row, i in enumerate(indices):
                        ids = result.sequences[row, prompt_length:].tolist()
                        eos = wrapper.tokenizer.eos_token_id
                        ended = eos in ids
                        if ended:
                            ids = ids[:ids.index(eos) + 1]
                        self.assertEqual(actual[i], {
                            'output': wrapper.tokenizer.decode(ids, skip_special_tokens=True),
                            'n_new_tokens': len(ids), 'truncated': not ended and len(ids) >= 4})
            finally:
                wrapper.close()


if __name__ == '__main__':
    unittest.main()
