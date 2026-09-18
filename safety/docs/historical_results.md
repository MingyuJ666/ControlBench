# Completed historical experiment: v2_0916

These are previously verified results, not outputs of the reorganized code.
The source delivery has 22 completed cells and two not-applicable SAE cells.
Fifteen cells were refit and evaluated using the v2 grids; seven were carried
forward. The `origin` column in [results_v2_0916.csv](results_v2_0916.csv) retains
that distinction. ACC and unsafe-F1 in CSV are fractions.

Original sealed archive: `v2_0916.tar.gz` (491 files).
The archive contains predictions, tensors, frozen configs, job records and the
original executable sources. It is a separate experiment deliverable; this code
release does not bundle or rewrite it.

| Model | Method | Origin | Layer | Alpha | ACC | unsafe-F1 |
|---|---|---|---:|---:|---:|---:|
| qwen3_8b | no_steering | v1_carried_forward | -1 | 0 | 37.15% | 52.74% |
| qwen3_8b | prompted | v1_carried_forward | -1 | 0 | 46.88% | 56.90% |
| qwen3_8b | caa | v2_0916_rerun | 27 | 2.5 | 41.32% | 54.45% |
| qwen3_8b | pca | v2_0916_rerun | 27 | 2.5 | 47.57% | 57.22% |
| qwen3_8b | lat | v2_0916_rerun | 9 | 1.5 | 48.26% | 57.55% |
| qwen3_8b | linear_probe | v2_0916_rerun | 9 | 0.5 | 37.15% | 52.74% |
| qwen3_8b | iti | v2_0916_rerun | -1 | 1 | 37.15% | 52.74% |
| gemma2_9b | no_steering | v1_carried_forward | -1 | 0 | 33.83% | 30.16% |
| gemma2_9b | prompted | v1_carried_forward | -1 | 0 | 48.87% | 35.85% |
| gemma2_9b | caa | v2_0916_rerun | 32 | 2.5 | 38.35% | 31.67% |
| gemma2_9b | pca | v2_0916_rerun | 32 | 2 | 35.34% | 30.65% |
| gemma2_9b | lat | v2_0916_rerun | 21 | 2.5 | 51.13% | 36.89% |
| gemma2_9b | linear_probe | v2_0916_rerun | 21 | 2 | 33.83% | 30.16% |
| gemma2_9b | sae | v1_carried_forward | 20 | 3 | 33.83% | 30.16% |
| gemma2_9b | iti | v2_0916_rerun | -1 | 1 | 29.32% | 28.79% |
| llama31_8b | no_steering | v1_carried_forward | -1 | 0 | 34.67% | 43.31% |
| llama31_8b | prompted | v1_carried_forward | -1 | 0 | 51.05% | 49.71% |
| llama31_8b | caa | v2_0916_rerun | 16 | 2.5 | 43.62% | 46.18% |
| llama31_8b | pca | v2_0916_rerun | 24 | 2.5 | 46.10% | 47.46% |
| llama31_8b | lat | v2_0916_rerun | 24 | 2.5 | 46.10% | 47.46% |
| llama31_8b | linear_probe | v2_0916_rerun | 8 | 1 | 34.86% | 43.38% |
| llama31_8b | iti | v2_0916_rerun | -1 | 1 | 32.95% | 42.67% |

Layer -1 means no residual-layer choice (reference conditions or global ITI).
Gemma SAE selected block20, alpha3 from its unchanged historical grid. Llama
PCA and LAT have equal aggregate metrics but two distinct sample predictions;
the methods and tensors are distinct. Four Llama PCA/LAT Test outputs were
invalid and were included in the metric denominators.

The independent archive verifier allows a 1e-6 relative / 1e-8 absolute tolerance
only when reconstructing tensor norms on another CPU. Selection retains the
original recorded norms exactly; no tie is re-decided from recomputed norms.
