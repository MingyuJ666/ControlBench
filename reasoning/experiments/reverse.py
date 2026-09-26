"""Fixed DiffMean L16 alpha=-1 evaluation with an independent sign calculation."""
import argparse
import json
from reasoning.common import SPLIT, PAIRS, prepare_run, ROOT, digest, read_jsonl, write_json, code_hashes, state
from pathlib import Path
from reasoning.datasets import problem_id
from reasoning.evaluation import evaluate


def run(args):
    import torch
    import transformers
    from reasoning.steering import RepSteerBackend, learner, rs
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output=args.output
    source=ROOT/'outputs/repsteer_compare/artifacts/diffmean'
    pair_path=PAIRS
    split_path=SPLIT
    split=json.loads(split_path.read_text())
    lookup={problem_id(r['problem']):dict(r,id=problem_id(r['problem']))
            for r in read_jsonl(ROOT/'data/original.jsonl')}
    rows=[lookup[i] for i in split['eval']]
    assert len(rows)==280
    pairs=read_jsonl(pair_path)
    teacher={r['id']:r for r in read_jsonl(ROOT/'data/positive.jsonl')}
    llama={r['id']:r for r in read_jsonl(ROOT/'data/negative_llama.jsonl')}
    for pair in pairs:
        assert pair['id'] in split['extraction'] and pair['id'] not in split['eval']
        assert teacher[pair['id']]['correct'] and not llama[pair['id']]['correct']
        assert pair['teacher']==teacher[pair['id']]['solution']
        assert pair['llama']==llama[pair['id']]['completion']
    config=dict(method='diffmean',layer=16,alpha=-1,phase='decode',max_tokens=8192,
        batch_size=4,device=args.device,eval_ids=split['eval'],seed=42,
        normalization='l2',selection='fixed by user after 11/20 vs 10/20 dev result',
        torch=torch.__version__,transformers=transformers.__version__,
        files={str(p.relative_to(ROOT)):digest(p) for p in
               [pair_path,split_path,ROOT/'data/original.jsonl',
                source/'manifest.json',source/'tensors.safetensors']},
        code=code_hashes())
    prepare_run(output, config, args.device)
    backend=RepSteerBackend(args.device,16)
    artifact=rs.load_artifact(source)
    artifact.bind(backend.model)
    data,ids=backend.contrastive_data(pair_path)
    state(output,'auditing_direction',pairs=len(ids))
    positive,negative=learner('diffmean',16)._capture_pair(backend.model,data,backend.store)
    def matrix(batch):
        x=batch.activations.detach().float().cpu()
        return x[:,0] if x.ndim==3 and x.shape[1]==1 else x
    pos,neg=matrix(positive),matrix(negative)
    raw=pos.mean(0)-neg.mean(0)
    expected=raw/raw.norm()
    saved=artifact.direction.float().cpu()
    cosine=float(torch.nn.functional.cosine_similarity(expected,saved,dim=0))
    torch.testing.assert_close(saved,expected,rtol=2e-4,atol=2e-5)
    from repsteer.operators.base import direction_tensor
    actual_delta=rs.operators.Add().apply(torch.zeros(1,1,saved.numel()),artifact,-1,None)
    torch.testing.assert_close(actual_delta[0,0],-saved)
    audit=dict(pairs=len(ids),positive_source='correct GPT-5.1',negative_source='incorrect Llama',
        saved_vector='normalize(mean(positive)-mean(negative))',cosine_with_recomputed=cosine,
        max_abs_error=float((saved-expected).abs().max()),raw_difference_norm=float(raw.norm()),
        alpha=-1,effective_delta='-saved_vector: toward negative group along this learned axis',
        labels_swapped=False,add_operator_sign_check_passed=True)
    write_json(output/'direction_audit.json',audit)
    state(output,'direction_audit_passed',**audit)
    candidate=dict(layer=16,alpha=-1,phase='decode')
    baseline=evaluate(backend,None,None,rows,8192,'eval',config,output)
    steered=evaluate(backend,artifact,candidate,rows,8192,'eval',config,output)
    b={r['id']:r for r in read_jsonl(output/'eval_baseline.jsonl')}
    s=read_jsonl(output/'eval_L16_a-1_decode.jsonl')
    gains=sum(not b[r['id']]['correct'] and r['correct'] for r in s)
    losses=sum(b[r['id']]['correct'] and not r['correct'] for r in s)
    report=dict(scope='Fixed-setting repeated within-MATH500 280-question evaluation',
        setting=candidate,max_tokens=8192,baseline=baseline,steered=steered,gains=gains,losses=losses,
        delta_correct=gains-losses,delta_accuracy=(gains-losses)/len(rows),direction_audit=audit,
        note='Both conditions rerun with the same repsteer wrapper and batch size. '
             'These 280 questions were evaluated previously; this is a repeated pilot, not a fresh external test.')
    write_json(output/'report.json',report)
    (output/'report.md').write_text('# DiffMean L16 alpha=-1\n\n```json\n'+json.dumps(report,indent=2)+'\n```\n')
    state(output,'completed',delta_correct=gains-losses,baseline_correct=baseline['correct'],
          steered_correct=steered['correct'],report=str(output/'report.md'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','run','status'])
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v2/repsteer_reverse_eval')
    args=parser.parse_args()
    from reasoning.common import dispatch
    dispatch(args, run, 'reasoning.experiments.reverse')


if __name__ == '__main__':
    main()
