"""Evaluate existing fixed steering vectors on MATH500, AMC 2023 and AIME 2024."""
import argparse
import json
import subprocess
from reasoning.common import SPLIT, PAIRS, MAX_TOKENS, SYSTEM, SUFFIX, COT_SUFFIX, open_weight_model, prepare_run, digest, read_jsonl, write_json, ROOT, code_hashes, dispatch, state
from pathlib import Path
from reasoning.datasets.benchmarks import DATA, load_suite
from reasoning.evaluation import evaluate
from reasoning.common import steering_artifact_path


# Fixed from existing MATH500 dev results; never selected on external tests.
CONDITIONS = {
    'diffmean': (ROOT / 'outputs/repsteer_tuning/artifacts/L24',
                 dict(layer=24, alpha=0.5, phase='decode')),
    'lat': (ROOT / 'outputs/repsteer_compare/artifacts/lat',
            dict(layer=16, alpha=0.5, phase='decode')),
}


def conditions(args):
    if not args.tuning_dirs:
        return CONDITIONS
    result = {}
    for directory in args.tuning_dirs:
        config = json.loads((directory/'config.json').read_text())
        selected = json.loads((directory/'selected_steering.json').read_text())
        candidate = selected['candidate']
        if candidate not in config['grid']:
            raise ValueError('Selected setting is outside the recorded search grid')
        method = config['method']
        tuned = config.get('confirmation_budget')
        if tuned is not None and tuned != args.max_tokens:
            raise ValueError(f'{directory} selected its setting at {tuned} tokens; '
                             f'evaluate with --max-tokens {tuned} or retune at {args.max_tokens}')
        if method in result:
            raise ValueError(f'Duplicate tuning directory for {method}')
        result[method] = (steering_artifact_path(directory/'artifacts', candidate), candidate)
    return result


def condition_names(args):
    if args.cot_only:
        return ['cot']
    return ['baseline', *(['cot'] if args.cot_baseline else []), *args.methods]


def selected_model(args):
    if args.model_path is None:
        return open_weight_model()
    return dict(model=args.model_id, revision=args.model_revision,
                local_path=str(args.model_path.resolve()), device=args.device)


def inputs(args):
    settings = conditions(args) if args.methods else {}
    datasets, audit = load_suite(args.data_dir)
    datasets = {name: datasets[name] for name in args.datasets}
    local = selected_model(args)
    files = [SPLIT, PAIRS, ROOT / 'data/original.jsonl',
             ROOT / 'outputs/repsteer_tuning/config.json',
             args.data_dir / 'manifest.json']
    files += [args.data_dir / f'{name}.jsonl' for name in ('amc2023', 'aime2024')]
    for method in args.methods:
        path, candidate = settings[method]
        manifest = json.loads((path / 'manifest.json').read_text())
        if (manifest['site']['layer'] != candidate['layer'] or
                manifest['model']['id'] != local['model'] or
                manifest['model']['revision'] != local['revision'] or
                manifest['method'] != {'diffmean': 'diff_mean', 'lat': 'lat', 'probe': 'linear_probe'}[method]):
            raise ValueError(f'Artifact does not match configured model/method/layer: {method}')
        files += [path / 'manifest.json', path / 'tensors.safetensors']
    for directory in args.tuning_dirs or []:
        files += [directory/'config.json',directory/'selected_steering.json']
    model_path = Path(local['local_path'])
    for name in ('config.json','generation_config.json','tokenizer_config.json',
                 'tokenizer.json','chat_template.jinja','model.safetensors.index.json'):
        if (model_path/name).exists():
            files.append(model_path/name)
    files += sorted((model_path/'.cache/huggingface/download').glob('*.metadata'))
    config = dict(datasets={name: [r['id'] for r in rows] for name, rows in datasets.items()},
                  conditions={m: settings[m][1] for m in args.methods},
                  max_tokens=args.max_tokens, batch_size=args.batch_size, do_sample=False, seed=42,
                  cot_baseline=args.cot_baseline, cot_only=args.cot_only,
                  evaluation_conditions=condition_names(args),
                  prompts=dict(system=SYSTEM,baseline=SUFFIX,
                               **({"cot":COT_SUFFIX} if args.cot_baseline else {})),
                  model=local['model'], revision=local['revision'], device=args.device,
                  local_path=local['local_path'],
                  enable_thinking={'default':None,'enabled':True,'disabled':False}[args.thinking],
                  files={str(p.resolve()): digest(p) for p in files}, code=code_hashes(),
                  repsteer_commit=subprocess.check_output(
                      ['git', '-C', str(ROOT / 'vendor/repsteer'), 'rev-parse', 'HEAD'], text=True).strip(),
                  split_audit=audit)
    return datasets, config


def report(output, results):
    write_json(output / 'report.json', results)
    text = '# Fixed-vector evaluation\n\nMATH500 is a repeated 280-question test; AMC/AIME are test-only. No test-set selection.\n\n'
    text += 'Gains/Losses are unavailable when only CoT is run; no ordinary baseline is generated.\n\n'
    text += '| Dataset | Condition | Correct | Accuracy | Gains | Losses | Truncated | Tokens | Reused |\n'
    text += '|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for dataset, conditions in results.items():
        for name, s in conditions.items():
            text += (f'| {dataset} | {name} | {s["correct"]}/{s["total"]} | {s["accuracy"]:.2%} '
                     f'| {s["gains"] if s["gains"] is not None else "—"} | {s["losses"] if s["losses"] is not None else "—"} | {s["truncated"]} | {s["tokens"]} | {s.get("reused", 0)} |\n')
    (output / 'report.md').write_text(text)


def run(args):
    datasets, config = inputs(args)
    settings = conditions(args) if args.methods else {}
    import torch
    import transformers
    from reasoning.steering import RepSteerBackend, rs
    config.update(torch=torch.__version__, transformers=transformers.__version__)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    existing = args.output/'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError('Config changed; use a new output directory')
    write_json(existing, config)
    class LazyBackend:
        instance = None
        def __getattr__(self, name):
            if self.instance is None:
                prepare_run(args.output, config, args.device)
                self.instance = RepSteerBackend(args.device, model_config=config,
                    enable_thinking=config['enable_thinking'])
            return getattr(self.instance, name)
    backend = LazyBackend()
    artifacts = {}
    for method in args.methods:
        artifacts[method] = rs.load_artifact(settings[method][0])
        artifacts[method].bind(backend.model)
    results = {}
    for dataset, rows in datasets.items():
        results[dataset] = {}
        baseline = None
        for method in condition_names(args):
            candidate = None if method in ('baseline','cot') else settings[method][1]
            stage = f'{dataset}_{method}'
            summary = evaluate(backend, artifacts.get(method), candidate, rows,
                               args.max_tokens, stage, config, args.output, args.batch_size,
                               prompt_suffix=COT_SUFFIX if method == "cot" else None)
            answers = read_jsonl(args.output / f'{stage}_{summary["label"]}.jsonl')
            if args.cot_only:
                gains = losses = delta_correct = delta_accuracy = None
            else:
                if baseline is None:
                    baseline = {r['id']: r for r in answers}
                gains = sum(not baseline[r['id']]['correct'] and r['correct'] for r in answers)
                losses = sum(baseline[r['id']]['correct'] and not r['correct'] for r in answers)
                delta_correct = gains-losses
                delta_accuracy = delta_correct/len(rows)
            summary['reused'] = sum('baseline_source' in r for r in answers)
            tokens = sum(r['tokens'] for r in answers)
            summary.update(gains=gains, losses=losses, delta_correct=delta_correct,
                           delta_accuracy=delta_accuracy, tokens=tokens,
                           mean_tokens=tokens/len(rows), truncated_rate=summary['truncated']/len(rows))
            results[dataset][method] = summary
            report(args.output, results)
    state(args.output, 'completed', report=str(args.output / 'report.md'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'run', 'status', 'dry-run'])
    parser.add_argument('--datasets', nargs='+', choices=['math500', 'amc2023', 'aime2024'],
                        default=['math500', 'amc2023', 'aime2024'])
    parser.add_argument('--methods', nargs='+', choices=['diffmean','lat','probe'])
    parser.add_argument('--cot-baseline', action='store_true',
                        help="Add an unsteered Let's think step by step. prompt condition")
    parser.add_argument('--cot-only', action='store_true',
                        help='Run only the unsteered step-by-step condition; no ordinary baseline')
    parser.add_argument('--baseline-only', action='store_true',
                        help='Evaluate prompt baselines without loading steering vectors')
    parser.add_argument('--tuning-dirs', nargs='+', type=Path, help='Read selected steering vectors from completed tune output directories')
    parser.add_argument('--max-tokens', type=int, default=MAX_TOKENS)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--model-path', type=Path, help='Override the local model for this run')
    parser.add_argument('--model-id', help='Model identity for --model-path')
    parser.add_argument('--model-revision', help='Pinned revision for --model-path')
    parser.add_argument('--thinking', choices=['default','enabled','disabled'], default='default',
                        help='Explicit chat-template thinking mode, shared by all conditions')
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/v2/benchmark_eval')
    args = parser.parse_args()
    if any((args.model_path, args.model_id, args.model_revision)) and not all((args.model_path, args.model_id, args.model_revision)):
        parser.error('Provide --model-path, --model-id and --model-revision together')
    if args.model_path is not None and not (args.model_path/'config.json').is_file():
        parser.error('--model-path must contain config.json')
    if args.cot_only:
        args.baseline_only = True
        args.cot_baseline = True
    if args.baseline_only and (args.methods or args.tuning_dirs):
        parser.error('--baseline-only cannot be combined with --methods or --tuning-dirs')
    args.methods = [] if args.baseline_only else (args.methods or list(conditions(args)))
    if any(m not in conditions(args) for m in args.methods):
        parser.error('Provide a --tuning-dirs selection for every requested method')
    if args.max_tokens < 1 or args.batch_size < 1 or len(set(args.datasets)) != len(args.datasets) or len(set(args.methods)) != len(args.methods):
        parser.error('Invalid budget or duplicate datasets/methods')
    if args.action == 'dry-run':
        datasets, config = inputs(args)
        print(json.dumps(dict(counts={k: len(v) for k, v in datasets.items()},
                              conditions=config['conditions'], max_tokens=args.max_tokens,
                              prompts=config['prompts'], cot_baseline=args.cot_baseline,
                              evaluation_conditions=config['evaluation_conditions'],
                              model=config['model'], revision=config['revision'],
                              enable_thinking=config['enable_thinking'],
                              split_audit=config['split_audit']), indent=2))
        return
    dispatch(args, run, 'reasoning.experiments.evaluate')


if __name__ == '__main__':
    main()
