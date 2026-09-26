# repsteer

[![PyPI](https://img.shields.io/pypi/v/repsteer.svg)](https://pypi.org/project/repsteer/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/binhu02/repsteer.svg)](LICENSE)

> **Documentation in progress:** The Markdown documentation and full docs site are currently under development. The Markdown available here is an LLM-generated draft and may contain inaccuracies or incomplete information.

`repsteer` is a Python toolkit for representation engineering and activation steering. It lets you learn, save, and apply interventions at semantic locations—such as “the residual-stream output of layer 20”—without writing PyTorch module paths.

It targets eager Hugging Face text models and static-image VLMs. It is intended for research, prototyping, and reproducible experiments; a steering direction is not guaranteed to work for every model or task.

```text
contrastive text -> capture activations -> Learner -> Artifact
                                                   |
prompt -> Model + Intervention -> steer() -> generated result
```

## Installation

Python 3.10+ is required. Install a PyTorch build appropriate to your CPU/CUDA environment first, then choose the extras you need:

```bash
# Core APIs and Hugging Face text models
python -m pip install repsteer

# SAELens SAE support
python -m pip install "repsteer[sae]"

# Static-image support for Qwen2.5-VL and InternVL
python -m pip install "repsteer[vlm]"
```

From a source checkout:

```bash
python -m pip install -e ".[dev]"
python -c "import repsteer; print(repsteer.__version__)"
```

## Quickstart: learn and apply a direction

This complete example learns a `DiffMean` direction from positive/negative contrastive text, then applies it during generation.

```python
import repsteer as rs

model = rs.models.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    dtype="bfloat16",
)

data = rs.data.ContrastivePairs.from_records([
    {"positive": "I help people when they struggle.", "negative": "I hurt people when they struggle."},
    {"positive": "Kindness and patience are valuable.", "negative": "Cruelty and impatience are valuable."},
    {"positive": "Respond with empathy and care.", "negative": "Respond with contempt and hostility."},
    {"positive": "I want others to feel safe.", "negative": "I want others to feel afraid."},
])

# Read the final non-padding-token activation from each text and compute
# positive mean - negative mean.
artifact = rs.learners.DiffMean(
    site=rs.sites.resid_post(layer=20),
    positions=rs.positions.LastNonPaddingToken(),
).fit(model, data)

control = rs.Intervention(
    artifact=artifact,
    operator=rs.operators.Add(),
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.Constant(1.5),
    phase="decode",
)

with model.steer(control):
    result = model.generate(
        "A customer says: I have waited two weeks and I am furious. Reply briefly:",
        max_new_tokens=64,
        do_sample=False,
        decode_kwargs={"skip_special_tokens": True},
    )

print(result.text)
artifact.save("artifacts/kindness")
```

This example downloads a model and therefore needs appropriate model access, network access, and memory. `resid_post(20)` is only an example layer; check the model depth and the site you want to study for each model. `result` is a `GenerationResult` with `text`, `token_ids`, and the underlying Transformers result in `raw`. For decoder-only models, `text` normally contains both the input prompt and generated text.

### The four key objects

| Object | Purpose | Value in the quickstart |
| --- | --- | --- |
| `Site` | A semantic model location to read or write activations | `resid_post(20)` |
| `Artifact` | A learned direction, subspace, probe, or SAE feature with provenance | Result of `DiffMean(...).fit(...)` |
| `Intervention` | How, where, when, and how strongly to apply an artifact | `Add + GeneratedTokens + 1.5` |
| `SteeringPlan` | An ordered collection of interventions | Optional when using one intervention |

## Common workflows

### 1. Reuse a saved artifact

An artifact directory contains a JSON manifest, `safetensors` tensors, and SHA-256 checksums. Loading verifies integrity by default.

```python
import repsteer as rs

artifact = rs.load_artifact("artifacts/kindness")
artifact.bind(model)  # Optional: validate model, revision, site, and hidden size early.

control = rs.Intervention(
    artifact=artifact,
    operator=rs.operators.Add(),
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.NormRelative(ratio=0.2),
    phase="decode",
)

with model.steer(control):
    result = model.generate(
        "A customer says: I have waited two weeks and I am furious. Reply briefly:",
        max_new_tokens=64,
    )
```

The default compatibility level, `"exact"`, checks model ID, an explicit revision, site, and hidden size. VLM artifacts also check processor/preprocessing information. Use `compatibility="architecture"` or `"dimension"` only for explicit research-reuse cases where weaker checking is intended.

### 2. Combine interventions and inspect the plan first

`compile()` resolves and validates a plan without installing hooks. `explain()` and `diagnostics()` help identify incorrect sites, execution order, and direction conflicts.

```python
prefill = (
    control.with_phase("prefill")
    .with_positions(rs.positions.LastPromptToken())
    .with_strength(0.5)
    .with_priority(-10)
)

plan = rs.SteeringPlan([prefill, control])
compiled = model.compile(plan)

print(compiled.explain())
print(compiled.diagnostics().effective_rank)

with model.steer(compiled):
    result = model.generate(
        "A customer says: I have waited two weeks and I am furious. Reply briefly:",
        max_new_tokens=48,
    )
```

Interventions run first by ascending `priority`, then by declaration order. Hooks exist only inside `with model.steer(...):` and are cleaned up even when generation raises an exception.

### 3. Capture activations directly

Use `capture_activations()` to inspect activations yourself or reuse a capture result:

```python
batch = rs.capture.capture_activations(
    model,
    inputs=["Kindness matters.", "I enjoy helping."],
    site=rs.sites.resid_post(20),
    positions=rs.positions.PromptTokens(),
    pooling="mean",
    batch_size=2,
)

print(batch.shape)  # [samples, hidden_size]
```

Pass `store=rs.capture.MemoryStore()` to cache results by capture request. Capture is rejected during an active steering session or `generate()` call so that intervened activations are not accidentally reused.

## Recipes: common intervention plans

`rs.recipes` contains small, artifact-first builders for common behaviors. Each
recipe returns an ordinary `SteeringPlan`; it does not compile a model, install
hooks, or start generation. The resulting plan remains inspectable and
composable with other interventions.

Remove the component of an activation in a learned direction or subspace:

```python
plan = rs.recipes.directional_ablation(
    artifact=direction_or_subspace,
    positions=rs.positions.AllTokens(),
    phase="both",
)

with model.steer(plan):
    result = model.generate("Write one sentence.", max_new_tokens=32, do_sample=False)
```

Apply a direction only when an existing gate enables it:

```python
gate = rs.gates.ProbeGate(
    probe_artifact,
    threshold=0.5,
    evaluate_at=rs.sites.resid_post(16),
)
plan = rs.recipes.gated_direction(
    steering_artifact=direction_artifact,
    gate=gate,
    positions=rs.positions.GeneratedTokens(),
    strength=0.4,
    phase="decode",
)
```

`gated_direction()` is a general conditional direction-addition API, not a
complete CAST implementation. It accepts artifacts rather than bare tensors;
the normal compiler still performs model, revision, site, and hidden-dimension
compatibility checks.

## Core API

### Models and generation

| API | Purpose |
| --- | --- |
| `rs.models.from_pretrained(model_id, *, revision=None, dtype=None, tokenizer=None, processor=None, **model_kwargs)` | Lazily load a Hugging Face text model or VLM; remaining arguments are passed to Transformers. |
| `rs.models.from_model(raw_model, tokenizer=None, processor=None, adapter=None, model_id=None, revision=None)` | Wrap an already-constructed PyTorch/Transformers model. Provide stable `model_id` and `revision` values for local models. |
| `model.generate(prompt_or_inputs, **kwargs)` | Hugging Face-style generation. It accepts strings, structured chat messages, encoded mappings, or tensors; extra options include `seed`, `decode_kwargs`, and chat-template controls. VLMs accept `image=` or `images=`. |
| `model.capture(request)` | Perform one raw activation capture; normally prefer `rs.capture.capture_activations()`. |
| `model.compile(plan, compatibility="exact")` | Validate sites, artifacts, gates, and composition order; returns a `CompiledPlan`. |
| `model.steer(plan, compatibility="exact")` | Return a context manager for calling `generate()` or `forward()` with steering active. |

### Chat-template instruction models

Pass Hugging Face-style messages directly to `generate()` for instruction
checkpoints. A message mapping, one conversation, or a batch of conversations
automatically uses `apply_chat_template`; raw strings retain the existing
untamplated behavior unless you explicitly request a template.

```python
messages = [
    {"role": "system", "content": "Answer concisely and accurately."},
    {"role": "user", "content": "What is activation steering?"},
]

with model.steer(control):
    result = model.generate(
        messages=messages,  # Positional conversations work too.
        max_new_tokens=96,
        do_sample=False,
        decode_kwargs={"skip_special_tokens": True},
    )
```

Generation adds the assistant generation prompt by default. To template a raw
string as one `user` turn, or to pass template-specific inputs, use:

```python
result = model.generate(
    "Summarize the result.",
    apply_chat_template=True,
    system_prompt="Use one sentence.",
    chat_template_kwargs={"tools": tools},  # Optional; passed to the template.
    add_generation_prompt=True,
    max_new_tokens=64,
)
```

The rendered template owns its special/control tokens, so repsteer defaults to
`add_special_tokens=False` while re-tokenizing it. Pass
`tokenizer_kwargs={"add_special_tokens": True}` only when a custom template
explicitly requires that behavior. For VLM generation with `image=` or
`images=`, the processor's template renderer is preferred and the normal image
placeholder/cardinality checks still apply.

Activation capture and contrastive learners use the same rendering rules. Their
`add_generation_prompt` default is `False`, which is usually appropriate when
learning from complete chat examples:

```python
chat_data = rs.data.ContrastivePairs.from_records(
    [
        {
            "positive_messages": [{"role": "user", "content": "Be helpful."}],
            "negative_messages": [{"role": "user", "content": "Be cruel."}],
        },
    ]
)
artifact = rs.learners.DiffMean(
    site=rs.sites.resid_post(20),
    positions=rs.positions.LastNonPaddingToken(),
    apply_chat_template=True,  # Optional for structured messages; recorded explicitly.
    system_prompt="Follow the user's request.",
).fit(model, chat_data)
```

`CaptureRequest` and learners also accept `chat_template_kwargs`. Template
settings, the effective capture rendering state, and the tokenizer's template
hash are recorded in cache/artifact provenance; exact artifact compatibility
checks a learned template hash when present. Do not mix raw prompts and
structured conversations in one automatic chat batch; split it or render the
inputs yourself. Already-tokenized `input_ids` bypass rendering and cannot be
combined with chat-template options.

### Sites: choose a model location

Built-in text-model adapters support these language-stream sites:

```python
rs.sites.resid_pre(layer)
rs.sites.attn_out(layer)
rs.sites.resid_mid(layer)
rs.sites.mlp_out(layer)
rs.sites.resid_post(layer)
```

Built-in text adapters support Gemma/Gemma2, Llama, Mistral, and Qwen2. Qwen2.5-VL and InternVL additionally support:

```python
rs.sites.vision_resid(layer)
rs.sites.projector_in()
rs.sites.projector_out()
```

`head_out()`, `logits()`, and `fusion_out()` are public site constructors, but current built-in adapters do not resolve them. This support statement applies to adapters, not a claim that every public checkpoint has been tested on hardware.

### Data, capture, and learners

```python
# Common contrastive-data constructors
rs.data.ContrastivePairs.from_records(records)
rs.data.ContrastivePairs.from_pairs(positive, negative)
rs.data.ContrastivePairs.from_groups(positive, negative)
```

Every built-in learner requires `site` and `positions`. Common options are `pooling`, `normalize`, `batch_size`, and `seed`; all use `.fit(model, data)` to produce an artifact.

| Learner | Suitable input | Output |
| --- | --- | --- |
| `rs.learners.DiffMean` | Paired or independent positive/negative samples | `DirectionArtifact` |
| `rs.learners.ActAdd` / `CAA` | Strictly aligned positive/negative pairs | `DirectionArtifact` |
| `rs.learners.PCA` | Positive/negative activation sets | `SubspaceArtifact` |
| `rs.learners.LAT` | Paired activation differences | Direction with one component; otherwise a subspace |
| `rs.learners.LinearProbe` / `LogisticProbe` | Binary positive/negative samples | `ProbeArtifact` |

`LastNonPaddingToken()` selects one activation per sample, which suits identity pooling. For multiple selected tokens, set `pooling="mean"`, `"last"`, or `"max"`:

```python
artifact = rs.learners.DiffMean(
    site=rs.sites.resid_post(20),
    positions=rs.positions.PromptTokens(),
    pooling="mean",
    batch_size=8,
).fit(model, data)
```

### Interventions, positions, and phases

```python
rs.Intervention(
    artifact=artifact,  # Required
    operator=rs.operators.Add(),  # Required
    positions=rs.positions.GeneratedTokens(),  # Required
    strength=rs.schedules.Constant(1),  # Required
    site=None,  # Defaults to artifact metadata's site
    gate=rs.gates.Always(),
    phase="both",  # "prefill" | "decode" | "both"
    priority=0,
)
```

`phase` controls when a hook may run; the position selector decides which tokens or patches are changed in that phase:

| Selector | Prefill | Decode |
| --- | --- | --- |
| `PromptTokens()` | Every non-padding prompt token | Empty |
| `LastPromptToken()` | Last non-padding prompt token | Empty |
| `GeneratedTokens()` | Empty | Token in the current decode forward |
| `AllTokens()` | Every non-padding prompt token | Current token |
| `LastNonPaddingToken()` | Last valid token in the current forward | Current token |
| `TokenIndices(...)`, `TextSpan(...)`, `SpecialToken(...)` | Their respective selected positions | Depends on the current forward |
| `ImageTokens(...)`, `ImagePatches(...)`, `ObjectPatches(...)` | Corresponding VLM positions only | Empty |

Important: in ordinary cached generation, a decode intervention using `GeneratedTokens()` does not change the first generated token, because its logits come from prefill. To affect it, operate on `LastPromptToken()` (or another appropriate selector) during `prefill`. Decode steering does not rewrite the existing KV cache.

### Operators and strength schedules

| API | Behavior |
| --- | --- |
| `rs.operators.Add()` / `Subtract()` | Add/subtract a direction; accept a direction, probe weight, or rank-1 subspace. |
| `rs.operators.RemoveProjection()` | Remove the projection onto a direction or full subspace. |
| `rs.operators.Replace()` | Interpolate toward a target vector by the specified strength. |
| `rs.operators.Clamp()` / `Ablate()` | Edit an already encoded SAE latent feature. |
| `rs.operators.SAEClamp()` / `SAEAblate()` | Encode, edit, and decode residual activations through an SAE while preserving the reconstruction residual. |
| `rs.schedules.Constant(alpha)` | Use fixed strength. |
| `rs.schedules.NormRelative(ratio, p=2)` | Scale strength by the current activation norm. |

`Constant(0)` and a statically zero `NormRelative` are strict no-ops: they do not call the selector, gate, or operator, and do not install the corresponding hook. This makes them useful baselines in sweeps.

### Gates: conditional steering

| API | Purpose |
| --- | --- |
| `rs.gates.Always()` | Default gate; always enabled. |
| `rs.gates.ProbeGate(...)` | Threshold a `ProbeArtifact` prediction. |
| `rs.gates.SAEActivationGate(...)` | Threshold an SAE feature activation. |
| `rs.gates.CosineGate(...)`, `CallableGate(...)` | Custom cosine- or Python-callable-based conditions. |
| `rs.gates.AndGate(...)`, `OrGate(...)`, `NotGate(...)` | Compose gates logically. |

When an activation gate has `evaluate_at=site`, it reads a language-stream
activation once in prefill and caches exactly one decision per original batch
sample for decode. That cache is scoped to one `model.generate()` invocation
and is cleared on both successful and exceptional exits. A scalar decision is
rejected for batch sizes greater than one rather than broadcast across samples.
Such sequence gates cannot start from a pre-populated KV cache, cannot use a
vision/projector condition site, and currently control language-stream targets
only; the runtime has no reliable mapping from language samples to arbitrary
vision/projector rows. A different language condition site is supported when
it executes no later than the steering site during prefill; use `phase="decode"`
for the stable staged form. Gates without `evaluate_at` remain dynamic gates of
the controlled activation.

### Artifacts and safe I/O

| API | Purpose |
| --- | --- |
| `artifact.save(path)` / `rs.artifacts.save_artifact(artifact, path)` | Save a portable artifact bundle. |
| `rs.load_artifact(path, verify_checksums=True, device="cpu")` | Verify SHA-256 checksums by default, then load. |
| `artifact.bind(model, compatibility="exact")` | Validate target-model compatibility without changing the artifact. |
| `rs.ArtifactMetadata` | Record model, site, data, and method provenance when manually constructing artifacts or selecting SAE features. |

The principal artifact types are `DirectionArtifact`, `SubspaceArtifact`, `ProbeArtifact`, and `SAEFeatureArtifact`. Artifacts store tensors and JSON metadata only; they do not load pickle or other executable payloads.

## Advanced: SAE feature steering

Install `repsteer[sae]` to select features from a SAELens SAE or your own `SAEAdapter`. This is the shortest path; replace `release` and `sae_id` with real SAE identifiers.

```python
sae = rs.sae.load(
    provider="saelens",
    release="<release>",
    sae_id="<sae-id>",
    device="cuda",
)

feature = rs.sae.select_feature(
    sae=sae,
    model=model,
    data=data,
    criterion="supervised",
    site=rs.sites.resid_post(20),
)

control = rs.Intervention(
    artifact=feature,
    operator=rs.operators.Add(),  # Add the feature's decoder direction.
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.Constant(0.5),
    phase="decode",
)
```

Use `rs.sae.select_supervised_features(...)` to get multiple features ranked by activation separation, or `rs.sae.select_causal_features(...)` to run a caller-provided causal evaluator over candidates.

Bare `Clamp` and `Ablate` apply only when the hook already supplies SAE latents. For normal residual-stream activations, pass the same runtime SAE to `SAEClamp` or `SAEAblate`:

```python
ablate = control.with_operator(rs.operators.SAEAblate(sae=sae))
clamp = control.with_operator(rs.operators.SAEClamp(value=5.0, sae=sae))
```

## Advanced: static-image VLM steering

Install `repsteer[vlm]`. Built-in VLM adapters support static images in Qwen2.5-VL and InternVL, loading an `AutoProcessor` when appropriate.

```python
from PIL import Image
import repsteer as rs

model = rs.models.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    revision="<immutable-model-commit>",
    dtype="bfloat16",
    device_map={"": "cuda:0"},
)

vision_artifact = rs.load_artifact("artifacts/vision_direction")
language_artifact = rs.load_artifact("artifacts/language_direction")

vision = rs.Intervention(
    artifact=vision_artifact,
    site=rs.sites.vision_resid(12),
    operator=rs.operators.Add(),
    positions=rs.positions.ObjectPatches(
        boxes=[(0.10, 0.10, 0.90, 0.90)],
        image_index=0,
    ),
    strength=rs.schedules.Constant(0.8),
    phase="prefill",
)
language = rs.Intervention(
    artifact=language_artifact,
    site=rs.sites.resid_post(20),
    operator=rs.operators.Add(),
    positions=rs.positions.GeneratedTokens(),
    strength=rs.schedules.Constant(-0.3),
    phase="decode",
)

image = Image.open("image.jpg")
with model.steer(rs.SteeringPlan([vision, language])):
    result = model.generate(
        "Describe the main object in this image.",
        image=image,
        max_new_tokens=64,
        do_sample=False,
    )

print(result.text)
```

VLM position APIs:

| API | Selects |
| --- | --- |
| `ImageTokens(image_index=0)` | Image-placeholder tokens for that image in the language stream. |
| `ImagePatches(mask, image_index=0)` | A 2D boolean mask on the processor's original, pre-merge patch grid. |
| `ObjectPatches(boxes=[(x1, y1, x2, y2)], image_index=0)` | Patches whose centers fall in normalized `XYXY` boxes. |

For one prompt and one image, repsteer can insert the model processor's image placeholder automatically. Multi-image or prompt-batch calls must provide explicit placeholders matching image cardinality; `image=` and `images=` are mutually exclusive. Video is unsupported. For tiled InternVL inputs without original-image crop transforms, `ObjectPatches` is rejected rather than using an unreliable position mapping.

## Evaluation and search

Use `Grid` and `sweep()` for small, recorded searches over layers, strengths, and selectors:

```python
search = rs.evaluation.Grid(
    sites=[rs.sites.resid_post(layer) for layer in (18, 20)],
    strengths=[0.0, 0.5, 1.0],
    positions=[rs.positions.GeneratedTokens()],
)

report = rs.evaluation.sweep(
    model=model,
    artifact=artifact,
    data=["Reply to a customer whose order is late."],
    search=search,
    metrics=[rs.evaluation.ContainsText("sorry")],
    generate_kwargs={"max_new_tokens": 32, "do_sample": False},
)

best = report.select(maximize="contains")
report.to_json("reports/sweep.json")
```

Related APIs include `rs.evaluation.CallableMetric`, `ContainsText`, `MeanOutputLength`, and `SweepReport`. VLM experiments can use `EvaluationMetricGroups`, `MultimodalEvaluationRecord`, and `MultimodalEvaluationReport` to record representation, causal-behavior, and capability/quality measurements separately.

### Deterministic held-out selection

`rs.selection` is a small, model-free configuration substrate: it splits stable
record IDs, evaluates already-defined candidates, and records a replayable
selection decision. It does not load a model, run generation, capture
activations, or train a learner.

```python
split = rs.selection.make_holdout_split(
    ["example-0", "example-1", "example-2", "example-3"],
    validation_ratio=0.5,
    seed=17,
    source_fingerprint="sha256:dataset-records",
)
candidates = [
    rs.selection.Candidate(
        "greater_0.25", {"threshold": 0.25, "comparator": "greater"}
    ),
    rs.selection.Candidate("less_0.25", {"threshold": 0.25, "comparator": "less"}),
]

report = rs.selection.grid_search(
    candidates,
    evaluator=lambda candidate: {
        "f1": evaluate_threshold(candidate, split.validation_ids)
    },
    objective="f1",
    split=split,
    seed=17,
)
print(report.selected_id)
report.to_json("reports/selection.json")
```

Candidate parameters and metric values are finite JSON data only. Selection
uses the declared objective, optional declared secondary metrics, and a final
canonical candidate-ID tie-break; all candidate metrics remain in the
checksummed `SelectionReport`. An auditable report always retains either the
explicit split or its fingerprint, plus a source fingerprint; use `select_best`
directly for an in-memory comparison that does not need a report.

### Role-keyed artifact bundles

`ArtifactBundle` groups independently valid artifacts by a stable key and
semantic role. Each component still uses the normal artifact manifest,
`safetensors`, checksum, and compatibility contract; loading validates both the
component and bundle digests.

```python
bundle = rs.artifacts.ArtifactBundle(
    components=(
        rs.artifacts.ArtifactBundleComponent(
            "condition/layer_12", "condition", condition_artifact
        ),
        rs.artifacts.ArtifactBundleComponent(
            "behavior/layer_16", "behavior", behavior_artifact
        ),
    ),
    selection_report=report,
)
bundle.save("artifacts/condition_behavior")
loaded = rs.artifacts.load_artifact_bundle("artifacts/condition_behavior")
loaded.bind(model)  # Checks every component at the requested compatibility level.
```

This is generic storage and provenance infrastructure, not a reproduction of a
method. In particular, a condition/behavior bundle does not provide a CAST
learner, recipe, or generation behavior.

### Adapter surface capabilities

Adapters expose a versioned, read-only declaration of the representation
surfaces they actually resolve and test:

```python
capabilities = model.adapter.capabilities
print(capabilities.explain())
capabilities.require("residual_write")
```

Built-in text adapters declare their residual read/write and language
sample-mapping contracts. Built-in VLM adapters separately declare their
static-image modality-map contract; this does not widen language target-row
mapping to vision or projector targets. `head_result` and `attention_bias` are
explicitly unsupported for every built-in adapter.

## Compatibility evidence

Adapter resolution and a successful public-checkpoint run are different claims.
The table lists only local evidence in this repository; none of these rows
claims a public checkpoint, CUDA, or multi-GPU validation.

| Model family | Adapter capability | Checkpoint/revision | Validation level | Notes |
| --- | --- | --- | --- | --- |
| Gemma2, Llama, Mistral, Qwen2 | Language residual/attention/MLP semantic sites and residual capability declaration | Local tiny Transformers configs | adapter contract tested | CPU unit contracts validate site read/rebuild behavior. |
| Qwen2.5-VL, InternVL | Language, vision-residual, projector input/output sites and static-image modality-map declaration | Local fake structural contracts | adapter contract tested | Static-image mapping contracts only; no public VLM checkpoint smoke test. |

## Runtime boundaries

- Local tests cover normal forwards with dynamic gates plus greedy and sampled
  batch generation with language sequence gates. Beam search, beam reordering,
  `num_return_sequences`, assisted generation, speculative decoding, custom
  generation loops, and model-parallel generation are not validated or
  supported by a state-remapping contract.
- Only eager Hugging Face runtime is supported.
- Quantized/offloaded target modules, model-parallel steering, video/audio, unlisted VLM architectures, and the VLM fusion stream are unsupported.
- Current built-in adapters do not support `head_out`, `logits`, or unit/head-level interventions.
- `head_result` and `attention_bias` capability surfaces are explicitly unsupported; no head hook, attention hook, or generation mode was added in 0.4.0.
- Dynamic recomputation of sequence gates during decode and re-entrant/checkpointed activation capture are unsupported.
- CAST, ITI, and PASTA have no stable learner, recipe, or runtime implementation.
  The 0.4.0 selection and bundle infrastructure is not a method reproduction.
- Adapter contract tests are not public-checkpoint validation. No new public
  checkpoint, CUDA, remote API, or benchmark dependency was added.
- Validate artifact compatibility, direction sign, layer choice, strength, processor mapping, and task metrics for every experiment. This library does not replace safety, bias, or effectiveness evaluation.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
mypy src/repsteer
ruff check .
ruff format --check .
```

The test suite uses local tiny-model configurations; CUDA tests are skipped when CUDA is unavailable.

## License

Apache-2.0. See [LICENSE](LICENSE).
