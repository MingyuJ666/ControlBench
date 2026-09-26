"""One fixed evaluation question through the reusable repsteer backend."""
from reasoning.datasets import add_extraction_argument
import argparse
from reasoning.common import SPLIT, NEW_OUTPUTS, MAX_TOKENS, write_json, code_hashes, open_weight_model, digest
from pathlib import Path
from reasoning.datasets import load
from reasoning.datasets.contrastive import prepare_pairs
from reasoning.evaluation import score


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=['diffmean','lat','probe'],default='diffmean')
    parser.add_argument('--alpha',type=float,default=1.5)
    parser.add_argument('--layer',type=int,default=16)
    parser.add_argument('--device',default='cuda:2')
    parser.add_argument('--max-tokens',type=int,default=MAX_TOKENS)
    parser.add_argument('--output',type=Path,default=NEW_OUTPUTS/'repsteer_sample')
    add_extraction_argument(parser)
    args=parser.parse_args()
    import json
    import torch
    from reasoning.steering import RepSteerBackend
    torch.set_num_threads(4)
    torch.manual_seed(42)
    if args.max_tokens<1:
        parser.error('max-tokens must be positive')
    split=json.loads((SPLIT).read_text())
    question=next(row for row in load() if row['id']==split['eval'][0])
    config=dict(method=args.method,alpha=args.alpha,layer=args.layer,device=args.device,
                extraction_position=args.extraction_position,
                max_tokens=args.max_tokens,sample_id=question['id'],code=code_hashes())
    if args.output.exists():
        parser.error('Use a fresh --output directory for a sample run')
    args.output.mkdir(parents=True)
    pair_path=prepare_pairs(open_weight_model()['model'], args.output)
    config['pairs_sha256']=digest(pair_path)
    write_json(args.output/'config.json',config)
    backend=RepSteerBackend(args.device,args.layer, extraction_position=args.extraction_position)
    data,ids=backend.contrastive_data(pair_path)
    artifact=backend.fit(args.method,data,args.output/'artifact')
    results={}
    for label,alpha in [('baseline',0),('steered',args.alpha)]:
        value=backend.generate([question['problem']],artifact,alpha,args.max_tokens)[0]
        results[label]={**value,**score(value['completion'],question['answer'])}
        write_json(args.output/f'{label}.json',results[label])
    report=dict(config=config,question=question,results=results,pairs=len(ids))
    write_json(args.output/'report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
