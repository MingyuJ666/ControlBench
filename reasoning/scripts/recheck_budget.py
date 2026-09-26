"""Re-evaluate a frozen tuning selection and identical eval IDs at a new budget."""
import argparse
import json
from pathlib import Path
from reasoning.common import ROOT, open_weight_model, prepare_run, digest, read_jsonl, write_json, code_hashes, dispatch, state
from reasoning.datasets import problem_id
from reasoning.evaluation import evaluate


def run(args):
    import torch
    import transformers
    from reasoning.steering import RepSteerBackend, rs
    source=args.source.resolve()
    old=json.loads((source/'config.json').read_text())
    selected=json.loads((source/'selected_steering.json').read_text())['candidate']
    local=open_weight_model()
    if any(old[k]!=local[k] for k in ('model','revision')):
        raise ValueError('Configured model differs from the original evaluation')
    if selected not in old['grid']:
        raise ValueError('Selection was not in original grid')
    data_path=ROOT/'data/original.jsonl'
    if digest(data_path)!=old['files']['data/original.jsonl']:
        raise ValueError('Original MATH data changed')
    lookup={problem_id(r['problem']):dict(r,id=problem_id(r['problem'])) for r in read_jsonl(data_path)}
    ids=old['eval_ids']
    if not ids or len(ids)!=len(set(ids)):
        raise ValueError('Missing or duplicate evaluation IDs')
    rows=[lookup[i] for i in ids]
    path=source/'artifacts'/f'L{selected["layer"]}'
    files=[source/'config.json',source/'selected_steering.json',path/'manifest.json',path/'tensors.safetensors',Path(__file__),data_path]
    config=dict(source=str(source),candidate=selected,eval_ids=ids,max_tokens=args.max_tokens,
                batch_size=old['batch_size'],seed=old['seed'],model=local['model'],
                revision=local['revision'],device=args.device,code=code_hashes(),
                files={str(p.resolve()):digest(p) for p in files},
                torch=torch.__version__,transformers=transformers.__version__)
    torch.set_num_threads(4)
    torch.manual_seed(old['seed'])
    prepare_run(args.output,config,args.device)
    backend=RepSteerBackend(args.device)
    artifact=rs.load_artifact(path)
    artifact.bind(backend.model)
    results=[]
    for candidate in [None,selected]:
        results.append(evaluate(backend,artifact if candidate else None,candidate,rows,
                       args.max_tokens,'eval',config,args.output,old['batch_size']))
        write_json(args.output/'results.json',results)
    for r in results:
        r['delta_correct']=r['correct']-results[0]['correct']
        r['delta_accuracy']=r['accuracy']-results[0]['accuracy']
    write_json(args.output/'report.json',dict(evaluation=results,source=str(source),max_tokens=args.max_tokens))
    text=f'# MATH budget recheck ({args.max_tokens} tokens)\n\nSame evaluation IDs and fixed steering selection as the source run.\n\n| Condition | Correct | Accuracy | Truncated |\n|---|---:|---:|---:|\n'
    for r in results:
        text+=f'| {r["label"]} | {r["correct"]}/{r["total"]} | {r["accuracy"]:.1%} | {r["truncated"]} |\n'
    (args.output/'report.md').write_text(text)
    state(args.output,'completed',evaluation=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','run','status'])
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--max-tokens',type=int,default=4096)
    args=parser.parse_args()
    if args.max_tokens<1:
        parser.error('Budget must be positive')
    dispatch(args,run,'reasoning.scripts.recheck_budget')
