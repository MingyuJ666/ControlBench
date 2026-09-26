"""Two-stage small-dev hyperparameter search, isolated from the ongoing method run."""
from reasoning.datasets import (add_extraction_argument, EXTRACTION_POSITIONS,
                                extraction_positions, expand_extraction_grid)
import argparse
import random
import json
import subprocess
from reasoning.common import SPLIT, PAIRS, MAX_TOKENS, steering_grid, open_weight_model, prepare_run, ROOT, digest, read_jsonl, write_json, code_hashes, state
from pathlib import Path
from reasoning.datasets import problem_id
from reasoning.datasets.contrastive import prepare_pairs
from reasoning.evaluation import evaluate
from reasoning.common import steering_artifact_path


def candidate_grid(smoke=False, layers=None, strengths=None, model=None,
                   extraction_position='before_answer'):
    if layers is None and strengths is None:
        layers, strengths = steering_grid(model or open_weight_model()['model'])
    elif not layers or not strengths:
        raise ValueError('Both layers and strengths are required')
    grid = [dict(layer=layer, alpha=alpha, phase='decode')
            for layer in layers for alpha in strengths]
    return expand_extraction_grid(grid[:2] if smoke else grid, extraction_position)


def shortlist(scores, k=3):
    return sorted([s for s in scores if s['candidate'] is not None],
                  key=lambda s:(-s['correct'],abs(s['candidate']['alpha']),s['label']))[:k]


def run(args):
    import torch
    import transformers
    from reasoning.steering import RepSteerBackend
    torch.set_num_threads(4)
    torch.manual_seed(42)
    output=args.output
    split_path=SPLIT
    pair_path = prepare_pairs(open_weight_model()['model'], output)
    data_path=ROOT/'data/original.jsonl'
    split=json.loads(split_path.read_text())
    lookup={problem_id(r['problem']):dict(r,id=problem_id(r['problem'])) for r in read_jsonl(data_path)}
    screen_ids=split['dev'][:2 if args.smoke else 8]
    confirm_ids=split['dev'][2:4] if args.smoke else split['dev'][8:]
    assert not set(screen_ids)&set(confirm_ids)
    pairs=read_jsonl(pair_path)
    assert {r['id'] for r in pairs}<=set(split['extraction'])
    assert not {r['id'] for r in pairs}&set(screen_ids+confirm_ids)
    screen_budget=confirm_budget=32 if args.smoke else args.max_tokens
    grid=candidate_grid(args.smoke, args.layers, args.strengths,
                        extraction_position=args.extraction_position)
    positions=extraction_positions(args.extraction_position)
    eval_ids=random.Random(42).sample(split['eval'], args.eval_limit)
    if set(eval_ids) & (set(screen_ids+confirm_ids) | {r['id'] for r in pairs}):
        raise ValueError('Evaluation data overlaps extraction/development data')
    local=open_weight_model()
    config=dict(method=args.method,grid=grid,screen_ids=screen_ids,confirm_ids=confirm_ids,
        screen_budget=screen_budget,confirmation_budget=confirm_budget,
        eval_ids=eval_ids,eval_budget=confirm_budget,
        normalization='l2',position={p: EXTRACTION_POSITIONS[p] for p in positions}, extraction_position=args.extraction_position,
        shortlist_rule='top 3 screen correct, tie smaller absolute alpha then label',
        batch_size=args.batch_size,seed=42,model=local['model'],revision=local['revision'],device=args.device,
        torch=torch.__version__,transformers=transformers.__version__,
        repsteer_commit=subprocess.check_output(['git','-C',str(ROOT/'vendor/repsteer'),
                                                'rev-parse','HEAD'],text=True).strip(),
        files={str(p.relative_to(ROOT)):digest(p) for p in [split_path,pair_path,data_path]},
        code=code_hashes())
    prepare_run(output, config, args.device)
    backend=RepSteerBackend(args.device, extraction_position=positions[0])
    artifacts={}
    for position in positions:
        backend.extraction_position=position
        data,valid=backend.contrastive_data(pair_path)
        audit_name=f'extraction_{position}.json' if len(positions)>1 else 'extraction.json'
        write_json(output/audit_name,dict(valid_ids=valid,pairs=len(valid),extraction_position=position))
        for layer in sorted({c['layer'] for c in grid}):
            state(output,'learning',layer=layer,pairs=len(valid),extraction_position=position)
            backend.layer=layer
            candidate=dict(layer=layer)
            if len(positions)>1:
                candidate['extraction_position']=position
            path=steering_artifact_path(output/'artifacts',candidate)
            artifacts[path]=backend.fit(args.method,data,path)
    screen_rows=[lookup[i] for i in screen_ids]
    screen=[]
    for candidate in [None]+grid:
        artifact=artifacts[steering_artifact_path(output/'artifacts',candidate)] if candidate else None
        result=evaluate(backend,artifact,candidate,screen_rows,screen_budget,'screen',config,output,args.batch_size)
        screen.append(result)
        write_json(output/'screen_leaderboard.json',screen)
        state(output,'screen_condition_completed',conditions=len(screen),total_conditions=len(grid)+1,**result)
    chosen=shortlist(screen)
    write_json(output/'shortlist.json',chosen)
    confirm_rows=[lookup[i] for i in confirm_ids]
    confirm=[]
    for candidate in [None]+[r['candidate'] for r in chosen]:
        artifact=artifacts[steering_artifact_path(output/'artifacts',candidate)] if candidate else None
        result=evaluate(backend,artifact,candidate,confirm_rows,confirm_budget,'confirm',config,output,args.batch_size)
        confirm.append(result)
        write_json(output/'confirmation_leaderboard.json',confirm)
    baseline=confirm[0]
    for result in confirm:
        result['delta_correct']=result['correct']-baseline['correct']
        result['delta_accuracy']=result['accuracy']-baseline['accuracy']
    best=max(confirm,key=lambda r:(r['correct'],r['candidate'] is None,
                                  -abs(r['candidate']['alpha']) if r['candidate'] else 0))
    report=dict(scope='Exploratory subdivision of an already reused 20-question dev set',
                screen=screen,confirmation=confirm,best=best,
                improvement_on_confirmation=best['correct']>baseline['correct'],
                note=f'Screening and confirmation both use {confirm_budget} tokens with matched controls. '
                     'Do not compare scores across stages directly or claim independent validation.')
    write_json(output/'report.json',report)
    write_json(output/'selected.json',best)
    steering_best=shortlist(confirm, k=1)[0]
    write_json(output/'selected_steering.json',steering_best)
    if eval_ids:
        # Select using development results before touching evaluation data.
        eval_rows=[lookup[i] for i in eval_ids]
        results=[]
        for candidate in [None, steering_best['candidate']]:
            artifact=artifacts[steering_artifact_path(output/'artifacts',candidate)] if candidate else None
            results.append(evaluate(backend,artifact,candidate,eval_rows,confirm_budget,
                                    'eval',config,output,args.batch_size))
        for result in results:
            result['delta_correct']=result['correct']-results[0]['correct']
            result['delta_accuracy']=result['accuracy']-results[0]['accuracy']
        report['evaluation']=results
        report['evaluation_note']='Fixed seed-42 subset of historical MATH500 eval; no eval-based selection. This split has been evaluated in previous experiments.'
        write_json(output/'report.json',report)
    text='# Small-dev hyperparameter search\n\n'+report['scope']+'\n\n'+report['note']+'\n'
    for title,rows in [(f'Screen ({len(screen_rows)} questions, {screen_budget} tokens)',screen),
                       (f'Confirmation ({len(confirm_rows)} questions, {confirm_budget} tokens)',confirm)]:
        text+='\n## '+title+'\n\n| Configuration | Correct | Accuracy | Truncated |\n|---|---:|---:|---:|\n'
        for row in rows:
            text+=f'| {row["label"]} | {row["correct"]}/{row["total"]} | {row["accuracy"]:.1%} | {row["truncated"]} |\n'
    text+='\nBest confirmation configuration: '+best['label']+'\n'
    if eval_ids:
        text+='\n## MATH evaluation\n\n'+report['evaluation_note']+'\n\n'
        text+='| Configuration | Correct | Accuracy | Truncated |\n|---|---:|---:|---:|\n'
        for row in report['evaluation']:
            text+=f'| {row["label"]} | {row["correct"]}/{row["total"]} | {row["accuracy"]:.1%} | {row["truncated"]} |\n'
    (output/'report.md').write_text(text)
    state(output,'completed',best=best,improvement=report['improvement_on_confirmation'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['start','run','status','dry-run'])
    parser.add_argument('--method',choices=['diffmean','lat','probe'],default='diffmean')
    parser.add_argument('--device',default='cuda:3')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--layers',nargs='+',type=int,help='Override the model-specific layer grid; also provide --strengths')
    parser.add_argument('--strengths',nargs='+',type=float,help='Override the default 0.5,1,1.5,2,2.5,3 strengths')
    parser.add_argument('--eval-limit',type=int,default=0)
    parser.add_argument('--max-tokens',type=int,default=MAX_TOKENS,help='Budget for screening, confirmation and --eval-limit evaluation')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/v2/repsteer_tuning')
    add_extraction_argument(parser, allow_both=True)
    args=parser.parse_args()
    import math
    if args.batch_size < 1 or args.max_tokens < 1:
        parser.error('--batch-size and --max-tokens must be positive')
    if not 0 <= args.eval_limit <= 280:
        parser.error('--eval-limit must be between 0 and 280')
    if (args.layers is None) != (args.strengths is None):
        parser.error('--layers and --strengths must be supplied together')
    if args.layers is not None and (any(x < 0 for x in args.layers) or
            len(set(args.layers)) != len(args.layers) or
            any(not math.isfinite(x) or x <= 0 for x in args.strengths) or
            len(set(args.strengths)) != len(args.strengths)):
        parser.error('Require distinct nonnegative layers and distinct positive finite strengths')
    if args.action == 'dry-run':
        grid = candidate_grid(args.smoke,args.layers,args.strengths,
                              extraction_position=args.extraction_position)
        print(json.dumps(dict(method=args.method,model=open_weight_model()['model'],
                             candidates=len(grid),grid=grid, extraction_position=args.extraction_position),indent=2))
        return
    from reasoning.common import dispatch
    dispatch(args, run, 'reasoning.experiments.tune')


if __name__ == '__main__':
    main()
