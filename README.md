# ControlBench

A benchmark for steering language models: whether an internal intervention
can induce a target behavior precisely, without degrading the model's
ability to answer. It is organized as domains; each domain is self-contained
under its own directory, with its own README, data and pipeline.

## Domains

- `instruction_following/` — hidden instruction following. A steering
  vector, not the prompt, has to make the model obey a formatting rule it
  was never told. Five rule-checkable tasks on OpenAssistant questions,
  seven steering methods under one search budget and one blinded judge.
  Its README carries the protocol, the current results, and the runbook.

`scripts/01_generate_traces.py` predates this layout: it belongs to the
DenseSteer reproduction whose package was removed from the repository, and
it does not run as is.
