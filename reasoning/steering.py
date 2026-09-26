"""Learn steering vectors and apply interventions during generation."""
import os
import sys
import torch
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from reasoning.common import ROOT, MAX_TOKENS, SUFFIX, open_weight_model, read_jsonl
from reasoning.models import TransformersLLM
from reasoning.datasets import extraction_text, EXTRACTION_POSITIONS

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
sys.path.insert(0, str(ROOT / 'vendor/repsteer/src'))
import repsteer as rs

METHODS = ('diffmean', 'lat', 'probe')

def learner(method, layer=16):
    common = dict(site=rs.sites.resid_post(layer=layer),
                  positions=rs.positions.LastNonPaddingToken(),
                  normalize='l2', batch_size=1, apply_chat_template=False,
                  special_tokens=False, capture_dtype=torch.float32)
    if method == 'diffmean':
        return rs.learners.DiffMean(**common)
    if method == 'lat':
        return rs.learners.LAT(n_components=1, center=True, **common)
    if method == 'probe':
        return rs.learners.LinearProbe(l2=0.1, max_iter=25, **common)
    raise ValueError(f'Unknown method: {method}')

def vector(artifact):
    return artifact.direction if hasattr(artifact, 'direction') else artifact.weight


def strip_generation(tokens, stop_ids, budget):
    """Trim batch padding at the first generated EOS; count actual generated tokens."""
    tokens = list(tokens)
    for i, token in enumerate(tokens):
        if token in stop_ids:
            return tokens[:i + 1], False
    return tokens, len(tokens) >= budget

class RepSteerBackend:
    def __init__(self, device=None, layer=16, model_config=None, enable_thinking=None,
                 extraction_position='before_answer'):
        if extraction_position not in EXTRACTION_POSITIONS:
            raise ValueError(f'Unknown extraction position: {extraction_position}')
        self.extraction_position = extraction_position
        config = open_weight_model() if model_config is None else model_config
        self.device = device or config['device']
        self.layer = layer
        self.base = TransformersLLM(config['local_path'], self.device, enable_thinking=enable_thinking)
        self.model = rs.models.from_model(self.base.model, self.base.tokenizer,
                        model_id=config['model'], revision=config['revision'])
        self.base.tokenizer.pad_token = self.base.tokenizer.eos_token
        self.base.tokenizer.padding_side = 'left'
        self.store = rs.capture.MemoryStore()

    def contrastive_data(self, pair_path):
        records = []
        for row in read_jsonl(pair_path):
            prefix = self.base.tokenizer.decode(self.base.prompt(row['problem']),
                                               skip_special_tokens=False)
            pos = extraction_text(row['teacher'], self.extraction_position)
            neg = extraction_text(row['negative'] if 'negative' in row else row['llama'],
                                  self.extraction_position)
            if not pos or not neg:
                continue
            positive, negative = prefix + pos, prefix + neg
            if max(len(self.base.tokenizer.encode(t, add_special_tokens=False))
                   for t in (positive, negative)) > min(16384, self.base.model.config.max_position_embeddings):
                continue
            records.append(dict(id=row['id'], positive=positive, negative=negative))
        if len(records) < 8:
            raise ValueError('Need at least 8 valid pairs')
        return rs.data.ContrastivePairs.from_records(records), [r['id'] for r in records]

    def fit(self, method, data, output):
        output = Path(output)
        if output.exists():
            artifact = rs.load_artifact(output)
            self.check_extraction_position(artifact)
            artifact.bind(self.model)
        else:
            artifact = learner(method, self.layer).fit(self.model, data, store=self.store)
            direction = vector(artifact)
            if not torch.isfinite(direction).all() or not torch.isclose(direction.norm(), torch.tensor(1.), atol=1e-4):
                raise ValueError('Expected finite unit-norm direction')
            artifact = self.tag_extraction_position(artifact)
            artifact.save(output)
        return artifact

    def tag_extraction_position(self, artifact):
        return replace(artifact, metadata=artifact.metadata.with_updates(config={
            **artifact.metadata.config, 'extraction_position': self.extraction_position,
            'extraction': EXTRACTION_POSITIONS[self.extraction_position]}))

    def check_extraction_position(self, artifact):
        if artifact.metadata.config.get('extraction_position') != self.extraction_position:
            raise ValueError('Artifact extraction position differs or is unspecified; '
                             'use a new output directory to fit a new artifact')

    def generate(self, questions, artifact=None, alpha=0, max_tokens=MAX_TOKENS, prompt_suffix=SUFFIX):
        prompts = [self.base.prompt(q, suffix=prompt_suffix) for q in questions]
        width = max(map(len, prompts))
        requested_budget = max_tokens
        if getattr(self.base, 'system_in_user', False):
            max_tokens = min(max_tokens, self.base.model.config.max_position_embeddings - width)
            if max_tokens <= 0:
                raise ValueError('Prompt exhausts model context')
        ids = torch.full((len(prompts), width), self.base.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, -len(prompt):] = torch.tensor(prompt, device=self.device)
            mask[i, -len(prompt):] = 1
        if alpha and artifact is None:
            raise ValueError('Nonzero alpha requires an artifact')
        control = rs.Intervention(artifact=artifact, operator=rs.operators.Add(),
            positions=rs.positions.GeneratedTokens(), strength=rs.schedules.Constant(alpha),
            phase='decode') if alpha else None
        with self.model.steer(control) if control else nullcontext():
            result = self.model.generate(input_ids=ids, attention_mask=mask,
                max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                eos_token_id=self.base.stop_ids, pad_token_id=self.base.tokenizer.pad_token_id)
        raw = result.raw.sequences if hasattr(result.raw, 'sequences') else result.raw
        values = []
        for row in raw[:, width:].tolist():
            tokens, truncated = strip_generation(row, self.base.stop_ids, max_tokens)
            values.append(dict(completion=self.base.tokenizer.decode(tokens, skip_special_tokens=True),
                               tokens=len(tokens), truncated=truncated,
                               max_new_tokens=max_tokens, requested_max_new_tokens=requested_budget))
        return values


def generate(backend, questions, artifact, candidate, budget):
    if candidate is None:
        return backend.generate(questions,max_tokens=budget)
    controls = [rs.Intervention(artifact=artifact,operator=rs.operators.Add(),
                positions=rs.positions.GeneratedTokens(),
                strength=rs.schedules.Constant(candidate['alpha']),phase='decode')]
    if candidate['phase'] == 'both':
        controls.append(rs.Intervention(artifact=artifact,operator=rs.operators.Add(),
            positions=rs.positions.LastPromptToken(),
            strength=rs.schedules.Constant(candidate['alpha']),phase='prefill'))
    # alpha=0 below suppresses only the backend's own optional hook; this outer
    # repsteer plan remains active during its model.generate call.
    with backend.model.steer(rs.SteeringPlan(controls)):
        return backend.generate(questions,max_tokens=budget)


