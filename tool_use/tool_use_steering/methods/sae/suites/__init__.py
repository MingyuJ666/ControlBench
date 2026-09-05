"""Public SAE releases, one module per subject model.

Each suite module exposes load(sae_layer) -> {"repo", "W_dec", "encode"}:
`W_dec` holds decoder directions as columns ([d_model, d_sae]) and `encode`
maps last-token residuals [n, d_model] to feature activations [n, d_sae],
following the release's official inference code exactly. The registry maps
a model-id substring to its suite -- loud on unknowns, because encoding
with the wrong dictionary is a silent scoring bug.
"""

from . import gemma_scope, llama_scope, qwen_scope

SUITES = (("qwen", qwen_scope), ("llama", llama_scope), ("gemma", gemma_scope))


def sae_for(model: str, sae_layer: int) -> dict:
    lowered = model.lower()
    for key, suite in SUITES:
        if key in lowered:
            return suite.load(sae_layer)
    raise KeyError(f"no SAE suite for model {model!r}; add a module under "
                   "methods/sae/suites and register it in SUITES")
