"""Paper figure and LaTeX table: best dev-selected steering config per model and method.

Reads frozen run outputs only (no model loading). Needs matplotlib, which is not in
the experiment environment:  python -m reasoning.scripts.best_config_figure --out DIR
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'outputs/v2'
DATASETS = [('math500', 'MATH500'), ('amc2023', 'AMC23'), ('aime2024', 'AIME24')]
METHODS = [('caa', 'CAA'), ('pca', 'PCA'), ('lat', 'LAT'), ('probe', 'Probe'),
           ('iti', 'ITI')]
COLORS = ['#2a78d6', '#eb6834', '#1baf7a']


def load(path):
    return json.loads((V2 / path).read_text())


def suite(run, methods, position):
    results = load(f'{run}/results.json')
    rows = {}
    for method in methods:
        selected = load(f'{run}/{method}_selected.json')
        rows[method] = dict(candidate=selected['candidate'], position=position,
                            dev=(selected['correct'], selected['total']),
                            test={d: results[d][method] for d, _ in DATASETS})
    baseline = {d: results[d]['baseline'] for d, _ in DATASETS}
    return rows, baseline


def fixed(tune_run, eval_run, method, position):
    selected = load(f'{tune_run}/selected_steering.json')
    report = load(f'{eval_run}/report.json')
    return (dict(candidate=selected['candidate'], position=position,
                 dev=(selected['correct'], selected['total']),
                 test={d: report[d][method] for d, _ in DATASETS}),
            {d: report[d]['baseline'] for d, _ in DATASETS})


def collect():
    """One run per model and method; Llama uses the fixed-vector evaluations of its tune runs."""
    llama, llama_base = {}, None
    for method in ('lat', 'probe'):
        llama[method], base = fixed(f'{method}_grid_20260913', f'{method}_grid_eval_20260913',
                                    method, 'before')
        if llama_base is not None and base != llama_base:
            raise ValueError('Llama baselines differ between evaluation runs')
        llama_base = base
    qwen, qwen_base = suite('paper_qwen_dev60_gpu3_20260916',
                            ['caa', 'pca', 'lat', 'probe', 'iti'], 'after')
    return [('Llama-3.1-8B-Instruct', llama, llama_base),
            ('Qwen3-8B', qwen, qwen_base)]


def setting(method, candidate):
    alpha = f'{candidate["alpha"]:g}'
    return f'48 heads, α={alpha}' if method == 'iti' else f'L{candidate["layer"]}, α={alpha}'


def pct(row):
    return 100 * row['correct'] / row['total']


def figure(models, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.linewidth': 0.6,
                         'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    ink, muted, grid = '#0b0b0b', '#52514e', '#e4e3df'
    deltas = [pct(r['test'][d]) - pct(base[d]) for _, rows, base in models
              for r in rows.values() for d, _ in DATASETS]
    limit = max(abs(min(deltas)), abs(max(deltas))) + 2
    ratios = [len(rows) + 0.6 for _, rows, _ in models]
    fig, axes = plt.subplots(1, len(models), figsize=(5.2, 2.6), sharey=True,
                             gridspec_kw=dict(width_ratios=ratios))
    width = 0.26
    for ax, (name, rows, base) in zip(axes, models):
        present = [(m, label) for m, label in METHODS if m in rows]
        for i, (method, _) in enumerate(present):
            for j, (dataset, _) in enumerate(DATASETS):
                delta = pct(rows[method]['test'][dataset]) - pct(base[dataset])
                ax.bar(i + (j - 1) * width, delta, width * 0.88, color=COLORS[j],
                       edgecolor='white', linewidth=0.6, zorder=3)
        ax.axhline(0, color=ink, linewidth=0.7, zorder=4)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([f'{label}{"†" if rows[m]["position"] == "before" else ""}\n'
                            + setting(m, rows[m]['candidate']).replace(', ', '\n')
                            for m, label in present], fontsize=6.5, color=ink)
        ax.tick_params(axis='x', length=0, pad=3)
        ax.tick_params(axis='y', colors=muted, labelsize=7)
        ax.set_ylim(-limit, limit)
        ax.set_xlim(-0.8, len(present) - 0.2)
        ax.yaxis.grid(True, color=grid, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.spines['left'].set_color(muted)
        ax.spines['bottom'].set_visible(False)
        baseline = ' / '.join(f'{pct(base[d]):.1f}' for d, _ in DATASETS)
        ax.set_title(f'{name}\nbaseline {baseline}', fontsize=7.5, color=ink, pad=4)
    axes[0].set_ylabel('Δ accuracy vs. baseline (pp)', color=ink)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(handles, [label for _, label in DATASETS], loc='upper center', ncol=3,
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02), handlelength=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=0.8)
    for suffix in ('pdf', 'png'):
        fig.savefig(out / f'best_configs.{suffix}', dpi=300, bbox_inches='tight')


def table(models, out):
    lines = [r'\begin{tabular}{llllrrrr}', r'\toprule',
             r'Model & Method & Setting & Ext. & Dev & MATH500 & AMC23 & AIME24 \\', r'\midrule']
    for index, (name, rows, base) in enumerate(models):
        if index:
            lines.append(r'\midrule')
        cells = ' & '.join(f'{pct(base[d]):.1f}' for d, _ in DATASETS)
        lines.append(rf'{name} & Baseline & -- & -- & -- & {cells} \\')
        for method, label in METHODS:
            if method not in rows:
                continue
            r = rows[method]
            values = []
            for d, _ in DATASETS:
                delta = pct(r['test'][d]) - pct(base[d])
                values.append(f'{pct(r["test"][d]):.1f} ({delta:+.1f})')
            setting_text = setting(method, r['candidate']).replace('α', r'$\alpha$')
            lines.append(rf' & {label} & {setting_text} & {r["position"]} & '
                         rf'{r["dev"][0]}/{r["dev"][1]} & ' + ' & '.join(values) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'best_configs.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    models = collect()
    figure(models, args.out)
    table(models, args.out)
    print(args.out)


if __name__ == '__main__':
    main()
