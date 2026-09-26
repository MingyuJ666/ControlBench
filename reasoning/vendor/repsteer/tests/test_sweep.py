from contextlib import contextmanager

import torch

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.evaluation import Grid, MeanOutputLength, sweep
from repsteer.models.outputs import GenerationResult
from repsteer.positions import GeneratedTokens, LastPromptToken
from repsteer.sites import resid_post


class _SweepModel:
    model_id = "tiny/sweep"
    revision = "r1"

    def __init__(self):
        self.active = None

    @contextmanager
    def steer(self, intervention):
        self.active = intervention
        try:
            yield
        finally:
            self.active = None

    def generate(self, prompt, **_kwargs):
        length = len(prompt.split()) + 1
        tokens = torch.arange(length).unsqueeze(0)
        return GenerationResult(text=prompt, token_ids=tokens, raw=tokens)


class _ChatSweepModel(_SweepModel):
    def __init__(self):
        super().__init__()
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        tokens = torch.arange(2).unsqueeze(0)
        return GenerationResult(text="chat", token_ids=tokens, raw=tokens)


def test_layer_strength_token_policy_sweep_is_machine_readable(tmp_path):
    artifact = DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny/sweep",
            model_revision="r1",
            site=resid_post(0),
            method="unit",
        ),
        torch.tensor([1.0, 0.0]),
    )
    report = sweep(
        model=_SweepModel(),
        artifact=artifact,
        data=["one", "two words"],
        search=Grid(
            sites=[resid_post(0), resid_post(1)],
            strengths=[0, 1],
            positions=[LastPromptToken(), GeneratedTokens()],
        ),
        metrics=[MeanOutputLength()],
    )

    assert len(report.records) == 8
    assert {record.phase for record in report.records} == {"prefill", "decode"}
    assert all(record.samples == 2 for record in report.records)
    assert report.select(maximize="output_length").metrics["output_length"] == 2.5
    report.to_json(tmp_path / "sweep.json")
    report.to_markdown(tmp_path / "sweep.md")
    assert (tmp_path / "sweep.json").is_file()
    assert (tmp_path / "sweep.md").is_file()


def test_sweep_passes_a_single_chat_message_as_a_positional_prompt():
    model = _ChatSweepModel()
    artifact = DirectionArtifact(
        ArtifactMetadata(
            model_id="tiny/sweep",
            model_revision="r1",
            site=resid_post(0),
            method="unit",
        ),
        torch.tensor([1.0, 0.0]),
    )
    message = {"role": "user", "content": "say hello"}

    report = sweep(
        model=model,
        artifact=artifact,
        data=[message],
        search=Grid(
            sites=[resid_post(0)],
            strengths=[0],
            positions=[LastPromptToken()],
        ),
        metrics=[MeanOutputLength()],
    )

    assert model.prompts == [(message, {})]
    assert report.records[0].samples == 1
