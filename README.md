# ControlBench

A benchmark for steering language models: whether an internal intervention
can induce a target behavior precisely, without degrading the model's
ability to answer. It is organized as domains; each domain is self-contained
under its own directory, with its own README, data and pipeline.

## Domains

- `cot/` — MATH500 reasoning with normal/CoT prompts and repsteer DiffMean, LAT,
  LinearProbe. See [cot/README.md](cot/README.md) for setup and commands.

- `instruction_following/` — hidden instruction following. A steering
  vector, not the prompt, has to make the model obey a formatting rule it
  was never told. Five rule-checkable tasks on OpenAssistant questions,
  seven steering methods under one search budget and one blinded judge.
  Its README carries the protocol, the current results, and the runbook.

- `tool_use/` — tool-use decision-making. A steering vector, not the prompt,
  has to move the model between calling a tool, asking for a missing
  parameter, and abstaining when no tool fits. Three decision tasks over
  Qwen 3 8B, Llama 3.1 8B and Gemma 2 9B, played both single-turn and
  multi-turn, scored by a per-model call parser and an LLM judge. Its README
  carries the protocol and the runbook.
