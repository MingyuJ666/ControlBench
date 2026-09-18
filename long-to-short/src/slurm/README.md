# Slurm templates

These scripts submit the non-ITI or ITI track as resumable jobs. They deliberately contain no account, partition, or user-home assumptions. Before a real submission, adjust the `#SBATCH` resources and any site-specific module commands for the local cluster.

`L2S_SCRATCH_ROOT` selects shared scratch storage (default: `PROJECT/.cache/l2s`), `L2S_GPU_VENV` selects the GPU environment, and `L2S_RUN_ROOT` selects the track output directory. Run a preview first:

```bash
bash src/slurm/submit_non_iti.sh --dry-run
bash src/slurm/submit_iti.sh --dry-run
```

Set `L2S_SUBMIT_CONFIRM=I_UNDERSTAND_FULL_GRID` only after reviewing the generated matrix and the storage requirements. `--through prepare` submits environment setup, fit, and baselines; `--through grid` additionally submits all grid shards; `--through report` also submits answer judging and reporting.
