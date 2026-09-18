import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import control_bench as cb
from control_bench import paths
from control_bench.cli import judge_vllm
from control_bench.domains.instruction_following import judge_core as core, scoring, tasks
from control_bench.runtime import search


class JudgeAndOutputsTest(unittest.TestCase):
    def test_rules_for_all_five_targets(self):
        cases=[('lowercase','hello world','Hello world'),('fixed_ending','ok. Hope this helps!','ok. hope this helps!'),
               ('question_ending','could you try?','please try.'),('one_sentence','hello world.','hello. goodbye.'),
               ('no_comma','red and blue','red, blue')]
        for target,good,bad in cases:
            with self.subTest(target=target):
                verifier=tasks.TASKS[target]['verifier']
                self.assertTrue(tasks.score(good,verifier)['rule_pass'])
                self.assertFalse(tasks.score(bad,verifier)['rule_pass'])
                self.assertTrue(tasks.score(good,verifier,truncated=True)['rule_pass'])
                self.assertFalse(tasks.score(bad,verifier,truncated=True)['rule_pass'])
                self.assertTrue(tasks.score(good,verifier,truncated=True)['truncated'])
                self.assertFalse(tasks.score('',verifier)['rule_pass'])
                self.assertFalse(tasks.score('',verifier,truncated=True)['rule_pass'])

    def test_lowercase_allows_uncased_answers_and_rejects_case_changes(self):
        for answer in ['42', ' 42\n', '-3.14', '４２', 'hello 42!', 'straße',
                       '中文', '!!!', '🙂']:
            with self.subTest(answer=answer):
                self.assertTrue(tasks.verify_lowercase(answer))
                self.assertTrue(tasks.score(answer, 'verify_lowercase')['rule_pass'])
        for answer in ['', '  \n\t', 'Hello 42', 'É', 'ǅ', '𝑨', 'ℂ', 'Ⓐ']:
            with self.subTest(answer=answer):
                self.assertFalse(tasks.verify_lowercase(answer))
                self.assertFalse(tasks.score(answer, 'verify_lowercase')['rule_pass'])

    def test_numeric_lowercase_answers_reach_judge_before_success_is_decided(self):
        answers = ['42', '41', '!!!', 'FORTY-TWO', '']
        items = [{'item_id': str(i), 'prompt_id': str(i), 'base_prompt': 'What is 6 times 7?'}
                 for i in range(len(answers))]
        outputs = [{'output': answer, 'truncated': False, 'n_new_tokens': 2}
                   for answer in answers]
        records = scoring.build_records('lowercase', 'full', items, outputs, 'numeric')
        self.assertEqual([r['rule_pass'] for r in records], [True, True, True, False, False])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(scoring.prepare_shards(records, folder, n_batches=1), 3)
            submitted = [json.loads(line) for line in (folder/'judge_in_0.jsonl').read_text().splitlines()]
            self.assertEqual({r['answer'] for r in submitted}, {'42', '41', '!!!'})
        scoring.merge(records, {})
        with self.assertRaisesRegex(RuntimeError, 'no judge verdict'):
            scoring.summarize(records)
        verdicts = {
            records[0]['uid']: {'coherent': True, 'answer_status': 'adequate'},
            records[1]['uid']: {'coherent': True, 'answer_status': 'partial'},
            records[2]['uid']: {'coherent': False, 'answer_status': 'none'},
        }
        scoring.merge(records, verdicts)
        self.assertEqual([r['success'] for r in records], [True, False, False, False, False])
        self.assertEqual(scoring.summarize(records)['accuracy'], 1/5)

    def test_truncated_answers_reach_judge_and_keep_quality_requirements(self):
        cases = [
            ('complete', 'what is the capital of france?', 'paris. an optional detail is', True, 'adequate'),
            ('partial', 'name the capitals of france and germany.', 'france: paris. germany:', True, 'partial'),
            ('none', 'what is the capital of france?', 'the rain falls on the', True, 'none'),
            ('incoherent', 'what is the capital of france?', 'paris the if because red', False, 'adequate'),
            ('format', 'what is the capital of france?', 'Paris.', True, 'adequate'),
            ('empty', 'what is the capital of france?', '', False, 'none'),
        ]
        items = [{'item_id': uid, 'prompt_id': uid, 'base_prompt': question}
                 for uid, question, *_ in cases]
        outputs = [{'output': answer, 'truncated': True, 'n_new_tokens': 3072}
                   for _, _, answer, *_ in cases]
        records = scoring.build_records('lowercase', 'full', items, outputs, 'cutoff')
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(scoring.prepare_shards(records, folder, n_batches=1), 4)
            submitted = [json.loads(line) for line in (folder/'judge_in_0.jsonl').read_text().splitlines()]
            self.assertEqual({row['uid'] for row in submitted}, {row['uid'] for row in records[:4]})
            self.assertEqual({row['answer'] for row in submitted}, {row['output'] for row in records[:4]})
        scoring.merge(records, {})
        with self.assertRaisesRegex(RuntimeError, 'no judge verdict'):
            scoring.summarize(records)
        verdicts = {record['uid']: {'coherent': case[3], 'answer_status': case[4]}
                    for record, case in zip(records[:4], cases[:4])}
        scoring.merge(records, verdicts)
        self.assertEqual([r['success'] for r in records], [True, False, False, False, False, False])
        self.assertTrue(all(r['truncated'] for r in records))
        summary = scoring.summarize(records)
        self.assertEqual(summary['n_total'], 6)
        self.assertEqual(summary['n_truncated'], 6)
        self.assertEqual(summary['accuracy'], 1/6)
        self.assertEqual(summary['truncation_rate'], 1.0)

    def test_original_request_and_candidate_followup_stay_separate(self):
        from unittest.mock import Mock
        item = {'uid': 'test', 'question': 'What is 2 + 3?',
                'answer': '2 + 3 = 5. Would you like another example?', 'n_words': 12}
        tokenizer = Mock()
        core.render(tokenizer, core.build_prompt(item))
        chat = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual([m['role'] for m in chat], ['system', 'user'])
        data = json.loads(chat[1]['content'])
        self.assertEqual(data['original_user_request'], item['question'])
        self.assertEqual(data['candidate_response'], item['answer'])
        self.assertEqual(data['id'], core.blind_label(item['uid']))
        self.assertFalse(tokenizer.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_judge_parser_complete_truncated_and_mismatched_ids(self):
        uid='test'; label=core.blind_label(uid)
        for text in [json.dumps({'id':label,'coherent':True,'answer_status':'adequate'}),
                     f'{{"id":"{label}","coherent":true,"answer_status":"adequate","note":"cut']:
            self.assertEqual(core.parse(text,uid)['answer_status'],'adequate')
        self.assertTrue(core.parse('{"id":"wrong","coherent":true,"answer_status":"partial"}',uid)['id_mismatch'])
        self.assertIsNone(core.parse('{"coherent":"false","answer_status":"adequate"}',uid))
        prefix = json.dumps({'id': label, 'required_content': 'Give the sum.',
                             'response_evidence': 'The answer gives four.'})[:-1]
        for tail in [', "coh', ', "coherent": true, "answer_status": "ade']:
            with self.subTest(tail=tail):
                self.assertIsNone(core.parse(prefix + tail, uid))
        verdict = core.parse(prefix + ', "coherent": true, "answer_status": "adequate", "note": "cut', uid)
        self.assertTrue(verdict['coherent'])
        self.assertEqual(verdict['answer_status'], 'adequate')

    def test_preparing_same_shards_preserves_valid_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)
            records=[{'uid':'x','question':'q','output':'a','rule_pass':True}]
            scoring.prepare_shards(records,dest,1)
            result=dest/'judge_out_0.jsonl'; result.write_text('completed verdict')
            (dest/'judge_run.json').write_text(json.dumps(core.run_spec(['judge_in_0.jsonl'],core.JUDGE_MODEL,'vllm',{})))
            scoring.prepare_shards(records,dest,1)
            self.assertEqual(result.read_text(),'completed verdict')
            records[0]['output']='changed answer'
            scoring.prepare_shards(records,dest,1)
            self.assertFalse(result.exists())

    def test_changed_judge_instructions_invalidate_cached_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            records = [{'uid': 'x', 'question': 'What is 6 times 7?',
                        'output': '42', 'rule_pass': True}]
            with patch.object(core, 'JUDGE_INSTRUCTIONS', 'previous judge rubric'):
                scoring.prepare_shards(records, folder, n_batches=1)
                (folder/'judge_run.json').write_text(json.dumps(
                    core.run_spec(['judge_in_0.jsonl'], core.JUDGE_MODEL, 'vllm', {})))
                (folder/'judge_out_0.jsonl').write_text(
                    json.dumps({'uid': 'x', 'coherent': True, 'answer_status': 'adequate'}) + '\n')
                self.assertTrue(scoring.judging_complete(folder))
            self.assertFalse(scoring.judging_complete(folder))
            scoring.prepare_shards(records, folder, n_batches=1)
            self.assertFalse((folder/'judge_run.json').exists())
            self.assertFalse((folder/'judge_out_0.jsonl').exists())

    def test_judge_retries_only_missing_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'judge_in_0.jsonl'
            items=[{'uid':uid,'question':'q','answer':'a','n_words':1} for uid in ['one','two']]
            source.write_text(''.join(json.dumps(item)+'\n' for item in items))
            out=source.with_name('judge_out_0.jsonl')
            out.write_text('{"uid":"one","coherent":true,"answer_status":"adequate","note":"cached"}\n')
            calls=[]
            def generate(prompts,*args,**kwargs):
                calls.append(prompts)
                text='invalid' if len(calls)==1 else json.dumps({'id':core.blind_label('two'),'coherent':True,'answer_status':'adequate'})
                return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]
            with patch.object(core,'render',side_effect=lambda tok,prompt:prompt):
                judge_vllm.judge_shard(SimpleNamespace(generate=generate),None,core,None,source)
            rows=[json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([len(x) for x in calls],[1,1])
            self.assertEqual([r['uid'] for r in rows],['one','two'])
            self.assertEqual(rows[0]['note'],'cached')

    def test_unparsed_judge_fails_instead_of_reporting_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'judge_in_0.jsonl'
            source.write_text('{"uid":"one","question":"q","answer":"a","n_words":1}\n')
            llm=SimpleNamespace(generate=lambda *a,**kw:[SimpleNamespace(outputs=[SimpleNamespace(text='invalid')])])
            with patch.object(core,'render',side_effect=lambda tok,prompt:prompt):
                with self.assertRaisesRegex(RuntimeError,'missing after 2'):
                    judge_vllm.judge_shard(llm,None,core,None,source,max_attempts=2)
            self.assertEqual(source.with_name('judge_out_0.jsonl').read_text(),'')

    def test_bounded_retry_keeps_full_answer_and_cached_verdict(self):
        for resume in (False, True):
            with self.subTest(resume=resume), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'judge_in_0.jsonl'
                answer = 'background ' * 2500 + 'The requested result is 42.'
                items = [{'uid': uid, 'question': 'What is 6 times 7?', 'answer': answer}
                         for uid in ('cached', 'failed')]
                source.write_text(''.join(json.dumps(item) + '\n' for item in items))
                cached = {'uid': 'cached', 'coherent': True, 'answer_status': 'adequate', 'note': 'cached'}
                out = source.with_name('judge_out_0.jsonl')
                out.write_text(json.dumps(cached) + '\n')
                broken = '{"response_evidence":"' + 'a ' * 512
                failures = source.with_name('judge_unparsed_0.jsonl')
                if resume:
                    failures.write_text(json.dumps({'uid': 'failed', 'raw_response': broken}) + '\n')
                primary, bounded = object(), object()
                calls = []
                def generate(prompts, sampling, **kwargs):
                    calls.append(sampling)
                    self.assertEqual(len(prompts), 1)
                    self.assertEqual(json.loads(prompts[0])['candidate_response'], answer)
                    row = {'id': core.blind_label('failed'), 'required_content': 'The product of 6 and 7 is 42.',
                           'response_evidence': 'The answer states 42.', 'coherent': True,
                           'answer_status': 'adequate', 'note': 'The required result is present.'}
                    text = json.dumps(row) if sampling is bounded else broken
                    return [SimpleNamespace(outputs=[SimpleNamespace(text=text, finish_reason='stop')])]
                with patch.object(core, 'render', side_effect=lambda tok, prompt: prompt):
                    judge_vllm.judge_shard(SimpleNamespace(generate=generate), None, core, primary,
                                          source, retry_sampling=bounded)
                self.assertEqual(calls, [bounded] if resume else [primary, bounded])
                rows = [json.loads(line) for line in out.read_text().splitlines()]
                self.assertEqual(rows[0], cached)
                self.assertEqual(rows[1]['uid'], 'failed')
                self.assertEqual(rows[1]['judge_mode'], 'bounded_evidence_v1')
                self.assertEqual(rows[1]['previous_raw_response'], broken)
                self.assertEqual(rows[1]['retry_spec'], core.retry_spec())
                self.assertTrue(rows[1]['coherent'])
                self.assertEqual(failures.read_text(), '')

    def test_bounded_failure_stays_unjudged_and_stops_identical_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            source.write_text('{"uid":"x","question":"q","answer":"a"}\n')
            calls = []
            def generate(prompts, sampling, **kwargs):
                calls.append(sampling)
                return [SimpleNamespace(outputs=[SimpleNamespace(text='broken', finish_reason='length')])]
            with patch.object(core, 'render', side_effect=lambda tok, prompt: prompt):
                with self.assertRaisesRegex(judge_vllm.UnparsedVerdictsError, 'missing after 2 attempts'):
                    judge_vllm.judge_shard(SimpleNamespace(generate=generate), None, core, None,
                                          source, retry_sampling=object())
            self.assertEqual(len(calls), 2)
            self.assertEqual(source.with_name('judge_out_0.jsonl').read_text(), '')
            failure = json.loads(source.with_name('judge_unparsed_0.jsonl').read_text())
            self.assertEqual(failure['judge_mode'], 'bounded_evidence_v1')
            self.assertEqual(failure['finish_reason'], 'length')
            self.assertNotIn('coherent', failure)

    def test_retry_requires_bounded_evidence_before_verdict(self):
        row = {'id': core.blind_label('input-id'), 'required_content': 'Calculate 6 times 7.',
               'response_evidence': 'The answer gives 42.', 'coherent': True,
               'answer_status': 'adequate', 'note': 'Correct result.'}
        verdict = core.parse_retry(json.dumps(row), 'input-id')
        self.assertEqual(verdict['required_content'], row['required_content'])
        self.assertEqual(verdict['response_evidence'], row['response_evidence'])
        self.assertEqual(verdict['uid'], 'input-id')
        self.assertIsNotNone(core._fields(verdict))
        for text in ['{"coherent":false', '{"coherent":false,"answer_status":"none"}',
                     json.dumps({k: row[k] for k in reversed(row)}),
                     json.dumps({**row, 'response_evidence': 'a' * 121}),
                     json.dumps({**row, 'required_content': ''}),
                     json.dumps({**row, 'coherent': 'false'})]:
            self.assertIsNone(core.parse_retry(text, 'input-id'))
        for evidence in ['a' * 120, '"' * 120, chr(92) * 120]:
            with self.subTest(evidence=evidence[:1]):
                text = json.dumps({**row, 'response_evidence': evidence})
                self.assertEqual(core.parse_retry(text, 'input-id')['response_evidence'], evidence)
        self.assertEqual(core.retry_spec()['prompt_sha256'],
                         core.run_spec([], core.JUDGE_MODEL, 'vllm', {})['prompt_sha256'])

    def test_retired_retry_verdicts_are_rejudged_with_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'judge_in_0.jsonl'
            source.write_text('{"uid":"x","question":"What is 6 times 7?","answer":"42"}\n')
            out = source.with_name('judge_out_0.jsonl')
            out.write_text(json.dumps({'uid': 'x', 'coherent': False, 'answer_status': 'none',
                                       'judge_mode': 'retired_retry', 'retry_spec': {}}) + '\n')
            (Path(tmp) / 'judge_run.json').write_text(json.dumps(
                core.run_spec([source.name], core.JUDGE_MODEL, 'vllm', {})))
            self.assertFalse(scoring.judging_complete(Path(tmp)))
            with self.assertRaisesRegex(ValueError, 'outdated retry settings'):
                scoring.load_verdicts(Path(tmp))
            row = {'id': core.blind_label('x'), 'required_content': 'Compute 6 times 7.',
                   'response_evidence': 'The answer states 42.', 'coherent': True,
                   'answer_status': 'adequate', 'note': 'Correct product.'}
            llm = SimpleNamespace(generate=lambda *a, **kw: [SimpleNamespace(outputs=[SimpleNamespace(text=json.dumps(row))])])
            with patch.object(core, 'render', side_effect=lambda tok, prompt: prompt):
                judge_vllm.judge_shard(llm, None, core, None, source, retry_sampling=object())
            self.assertTrue(json.loads(out.read_text())['coherent'])
            self.assertTrue(scoring.judging_complete(Path(tmp)))

    def test_retry_sampling_supports_both_vllm_apis(self):
        from types import ModuleType
        from unittest.mock import Mock
        for field, cls in [('guided_decoding', 'GuidedDecodingParams'),
                           ('structured_outputs', 'StructuredOutputsParams')]:
            with self.subTest(api=field):
                module = ModuleType('vllm')
                module.SamplingParams = Mock(side_effect=lambda **kwargs: kwargs)
                module.sampling_params = SimpleNamespace(**{cls: Mock(side_effect=lambda **kwargs: kwargs)})
                with patch.dict('sys.modules', {'vllm': module}):
                    sampling = judge_vllm.make_retry_sampling(core)
                self.assertEqual(sampling['max_tokens'], 512)
                self.assertEqual(sampling['max_tokens'], core.MAX_NEW)
                self.assertEqual(sampling['temperature'], 0.0)
                self.assertEqual(sampling[field]['regex'], core.retry_pattern())
        self.assertIsNone(judge_vllm.make_retry_sampling(SimpleNamespace()))

    def test_judge_cli_continues_other_shards_after_unparsed_failure(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / 'first', Path(tmp) / 'second'
            first.mkdir()
            second.mkdir()
            sources = [first / 'judge_in_0.jsonl', first / 'judge_in_1.jsonl', second / 'judge_in_0.jsonl']
            for source in sources:
                source.write_text('')
            legacy = core.run_spec([p.name for p in sources[:2]], core.JUDGE_MODEL, 'vllm', {})
            (first / 'judge_run.json').write_text(json.dumps(legacy))
            cached = first / 'judge_out_1.jsonl'
            cached.write_text('keep existing verdict')
            vllm = SimpleNamespace(LLM=Mock(return_value=SimpleNamespace(get_tokenizer=lambda: None, llm_engine=None)),
                                   SamplingParams=Mock(), __version__='test')
            seen = []
            def judge_shard(*args, **kwargs):
                seen.append(args[4])
                if args[4] == sources[0]:
                    raise judge_vllm.UnparsedVerdictsError('one missing verdict')
            with patch.dict('sys.modules', {'vllm': vllm}), \
                 patch('sys.argv', ['judge_vllm.py', str(first), str(second)]), \
                 patch.object(judge_vllm, 'load_core', return_value=core), \
                 patch.object(judge_vllm, 'make_retry_sampling', return_value=object()), \
                 patch.object(judge_vllm, 'judge_shard', side_effect=judge_shard):
                with self.assertRaisesRegex(judge_vllm.UnparsedVerdictsError, 'one missing verdict'):
                    judge_vllm.main()
            self.assertEqual(seen, sources)
            self.assertEqual(cached.read_text(), 'keep existing verdict')
            manifest = json.loads((first / 'judge_run.json').read_text())
            self.assertEqual(manifest['unparsed_retry'], core.retry_spec())
            self.assertEqual({k: manifest[k] for k in legacy}, legacy)

    def test_empty_judge_shard_needs_no_model_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'judge_in_0.jsonl'; source.write_text('')
            judge_vllm.judge_shard(None,None,core,None,source)
            self.assertEqual(source.with_name('judge_out_0.jsonl').read_text(),'')

    def test_incomplete_or_changed_generation_request_is_not_reused(self):
        wrapper=SimpleNamespace(tokenizer=None,revision='test')
        domain=SimpleNamespace(NAME='test',render=lambda tok,item:item['text'])
        items=[{'item_id':'a','text':'first'},{'item_id':'b','text':'second'}]
        calls=[]
        def generate(wrapper,texts,*args,**kwargs):
            calls.append(texts)
            return [{'output':'ok','truncated':False,'n_new_tokens':1} for _ in texts]
        with tempfile.TemporaryDirectory() as tmp,patch.object(paths,'DATA_ROOT',Path(tmp)),patch.object(search,'generate_texts',side_effect=generate):
            kwargs=dict(model='test/model',experiment='calibration',method='reference',target='target',
                        cand=search.candidate('resid_post_0',0),artifact=None,batch_size=2,max_new_tokens=2,log=lambda _:None)
            out=search.generate_candidate(wrapper,domain,items=items[:1],**kwargs)
            search.generate_candidate(wrapper,domain,items=items,**kwargs)
            self.assertEqual(len(search.read_jsonl(out/'generations.jsonl')),2)
            search.generate_candidate(wrapper,domain,items=items,**kwargs)
            self.assertEqual(len(calls),2)
            (out/'summary.json').write_text('{}')
            (out.parent/'selection.json').write_text('{}')
            judge=out.parent/'judging'; judge.mkdir(); (judge/'judge_out_0.jsonl').write_text('cached')
            kwargs['max_new_tokens']=3
            search.generate_candidate(wrapper,domain,items=items,**kwargs)
            self.assertEqual(len(calls),3)
            self.assertFalse((out/'summary.json').exists())
            self.assertFalse((out.parent/'selection.json').exists())
            self.assertFalse(judge.exists())
            items[0]['text']='same id, changed input'
            search.generate_candidate(wrapper,domain,items=items,**kwargs)
            self.assertEqual(len(calls),4)


if __name__=='__main__': unittest.main()
