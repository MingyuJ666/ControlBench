import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from control_bench.cli import judge_vllm
from control_bench.domains import instruction_following as domain
from control_bench.domains.instruction_following import judge_core as core


class CharacterTokenizer:
    def encode(self, prompt):
        return list(map(ord, prompt))

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]['content']


def item_with_length(uid, length):
    item = {'uid': uid, 'question': 'What is 6 times 7?', 'answer': ''}
    overhead = len(core.build_prompt(item))
    item['answer'] = 'a' * (length - overhead)
    assert len(core.build_prompt(item)) == length
    return item


def verdict(label):
    return json.dumps({'id': label, 'required_content': 'Compute 6 times 7.',
                       'response_evidence': 'The candidate states 42.', 'coherent': True,
                       'answer_status': 'adequate', 'note': 'The result is correct.'})


def write_items(path, items):
    path.write_text(''.join(json.dumps(item) + '\n' for item in items))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class JudgeContextTest(unittest.TestCase):
    def setUp(self):
        self.tok = CharacterTokenizer()
        self.primary, self.bounded = object(), object()
        self.calls = []
        self.capacity = 8192

        def generate(prompts, sampling, **kwargs):
            self.calls.append((prompts, sampling))
            outputs = []
            for prompt in prompts:
                ids = prompt['prompt_token_ids']
                self.assertLessEqual(len(ids) + core.MAX_NEW, self.capacity)
                data = json.loads(''.join(map(chr, ids)))
                outputs.append(SimpleNamespace(outputs=[SimpleNamespace(text=verdict(data['id']))]))
            return outputs
        self.llm = SimpleNamespace(generate=generate)

    def judge(self, source, **kwargs):
        return judge_vllm.judge_shard(self.llm, self.tok, core, self.primary, source,
                                      retry_sampling=self.bounded,
                                      max_model_len=self.capacity, **kwargs)

    def test_exact_boundary_overflow_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            items = [item_with_length('cached', 200), item_with_length('short', 300),
                     item_with_length('boundary', 7680), item_with_length('overflow', 7681),
                     item_with_length('reported', 8193)]
            write_items(source, items)
            original = source.read_bytes()
            out = source.with_name('judge_out_0.jsonl')
            cached = {'uid': 'cached', 'coherent': True, 'answer_status': 'adequate', 'note': 'cached'}
            out.write_text(json.dumps(cached) + '\n')
            with self.assertRaisesRegex(judge_vllm.UnparsedVerdictsError, '2 inputs exceed'):
                self.judge(source)
            rows = read_rows(out)
            self.assertEqual([r['uid'] for r in rows], ['cached', 'short', 'boundary'])
            self.assertEqual(rows[0], cached)
            self.assertEqual(rows[-1]['input_tokens'], 7680)
            failures = read_rows(source.with_name('judge_unparsed_0.jsonl'))
            self.assertEqual([r['required_model_len'] for r in failures], [8193, 8705])
            self.assertTrue(all(r['error'] == 'context_length_exceeded' for r in failures))
            self.assertTrue(all('coherent' not in r for r in failures))
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(len(self.calls[0][0]), 2)
            self.capacity = 32768
            self.judge(source)
            self.assertEqual(len(self.calls), 2)
            self.assertEqual(len(self.calls[-1][0]), 2)
            self.assertIs(self.calls[-1][1], self.primary)
            self.assertEqual(read_rows(out)[:3], rows)
            self.assertEqual(read_rows(out)[-1]['max_model_len'], 32768)
            self.assertEqual(source.with_name('judge_unparsed_0.jsonl').read_text(), '')
            self.assertEqual(source.read_bytes(), original)

    def test_all_oversized_inputs_make_no_model_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            write_items(source, [item_with_length('x', 8193)])
            with self.assertRaisesRegex(judge_vllm.UnparsedVerdictsError, 'after 0 attempts'):
                self.judge(source)
            self.assertEqual(self.calls, [])
            self.assertEqual(source.with_name('judge_out_0.jsonl').read_text(), '')

    def test_reserves_the_larger_retry_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            write_items(source, [item_with_length('x', 7680)])
            with patch.object(core, 'RETRY_MAX_NEW', 1024):
                with self.assertRaises(judge_vllm.UnparsedVerdictsError):
                    self.judge(source)
            self.assertEqual(self.calls, [])
            self.assertEqual(read_rows(source.with_name('judge_unparsed_0.jsonl'))[0]['required_model_len'], 8704)

    def test_context_failures_survive_a_successful_bounded_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            write_items(source, [item_with_length('bad-json', 300), item_with_length('long', 8193)])
            original_generate = self.llm.generate
            def generate(prompts, sampling, **kwargs):
                outputs = original_generate(prompts, sampling, **kwargs)
                if sampling is self.primary:
                    outputs[0].outputs[0].text = 'unclosed JSON'
                return outputs
            self.llm.generate = generate
            with self.assertRaises(judge_vllm.UnparsedVerdictsError):
                self.judge(source)
            self.assertEqual([s for _, s in self.calls], [self.primary, self.bounded])
            self.assertEqual(read_rows(source.with_name('judge_out_0.jsonl'))[0]['previous_raw_response'], 'unclosed JSON')
            self.assertEqual([r['uid'] for r in read_rows(source.with_name('judge_unparsed_0.jsonl'))], ['long'])

    def test_domain_entrypoint_default_window_continues_shards_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first, second = folder / 'judge_in_0.jsonl', folder / 'judge_in_1.jsonl'
            write_items(first, [item_with_length('oversized', 32769), item_with_length('normal', 300)])
            write_items(second, [item_with_length('later-shard', 300)])
            self.capacity = 32768
            self.llm.get_tokenizer = lambda: self.tok
            self.llm.llm_engine = None
            vllm = SimpleNamespace(LLM=Mock(return_value=self.llm),
                                   SamplingParams=Mock(return_value=self.primary), __version__='test')
            def subprocess_run(cmd, **kwargs):
                with patch('sys.argv', cmd[1:]):
                    judge_vllm.main()
            with patch.dict('sys.modules', {'vllm': vllm}), \
                 patch.object(judge_vllm, 'make_retry_sampling', return_value=self.bounded), \
                 patch('subprocess.run', side_effect=subprocess_run):
                with self.assertRaisesRegex(judge_vllm.UnparsedVerdictsError, '1 inputs exceed'):
                    domain.judge([folder])
                self.assertEqual(len(self.calls), 2)
                self.assertEqual(vllm.LLM.call_args.kwargs['max_model_len'], 32768)
                self.assertEqual(read_rows(second.with_name('judge_out_1.jsonl'))[0]['uid'], 'later-shard')
                manifest = json.loads((folder / 'judge_run.json').read_text())
                self.assertEqual(manifest['max_model_len'], 32768)
                self.assertEqual(manifest['max_new_tokens'], 512)
                self.capacity = 40000
                with patch('sys.argv', ['judge_vllm.py', str(folder), '--max-model-len', '40000']):
                    judge_vllm.main()
                self.assertEqual(len(self.calls), 3)
                self.assertEqual(len(self.calls[-1][0]), 1)
                self.assertIs(self.calls[-1][1], self.primary)
                self.assertEqual(first.with_name('judge_unparsed_0.jsonl').read_text(), '')

    def test_tool_use_default_window_fits_full_followup_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'judge_in_0.jsonl').write_text('')
            vllm = SimpleNamespace(LLM=Mock(return_value=SimpleNamespace(
                get_tokenizer=lambda: self.tok, llm_engine=None)),
                SamplingParams=Mock(), __version__='test')
            with patch.dict('sys.modules', {'vllm': vllm}), \
                 patch.object(judge_vllm, 'make_retry_sampling', return_value=None), \
                 patch('sys.argv', ['judge_vllm.py', str(folder), '--domain', 'tool_use']):
                judge_vllm.main()
            self.assertEqual(vllm.LLM.call_args.kwargs['max_model_len'], 16384)
            self.assertEqual(vllm.SamplingParams.call_args.kwargs['max_tokens'], 256)

    def test_invalid_capacity_rejected_before_loading_model(self):
        vllm = SimpleNamespace(LLM=Mock())
        for length in ('0', '-1', '512'):
            with self.subTest(length=length), patch.dict('sys.modules', {'vllm': vllm}), \
                 patch('sys.argv', ['judge_vllm.py', '/unused', '--max-model-len', length]), \
                 redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    judge_vllm.main()
                self.assertEqual(exc.exception.code, 2)
        vllm.LLM.assert_not_called()


if __name__ == '__main__':
    unittest.main()
