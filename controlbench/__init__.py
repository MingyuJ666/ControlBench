"""ControlBench: a benchmark for inference-time control / steering methods.

The first method implemented is DenseSteer (Ouyang et al., ICML 2026),
a training-free activation-steering method that pushes small language models
toward "dense" math reasoning (fewer steps, higher info per step).
"""

__all__ = ["data", "models", "eval", "rewrite", "methods"]
