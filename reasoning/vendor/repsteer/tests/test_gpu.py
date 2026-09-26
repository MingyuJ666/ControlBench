import pytest
import torch
import transformers

from repsteer.artifacts import ArtifactMetadata, DirectionArtifact
from repsteer.core import Intervention
from repsteer.models import from_model
from repsteer.operators import Add
from repsteer.positions import GeneratedTokens
from repsteer.schedules import Constant
from repsteer.sites import resid_post


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_cached_generation_zero_strength_matches_baseline():
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    wrapper = from_model(
        transformers.LlamaForCausalLM(config).eval().cuda(),
        model_id="tiny/cuda-llama",
        revision="gpu-contract-1",
    )
    site = resid_post(0)
    artifact = DirectionArtifact(
        ArtifactMetadata(
            model_id=wrapper.model_id,
            model_revision=wrapper.revision,
            architecture=wrapper.architecture,
            site=site,
            hidden_size=wrapper.hidden_size,
            method="gpu_contract",
        ),
        torch.linspace(-1, 1, wrapper.hidden_size),
    )
    control = Intervention(
        artifact=artifact,
        operator=Add(),
        positions=GeneratedTokens(),
        strength=Constant(0),
        phase="decode",
    )
    input_ids = torch.tensor([[1, 5, 6]])

    baseline = wrapper.generate(input_ids, max_new_tokens=3, do_sample=False, seed=42)
    with wrapper.steer(control):
        steered = wrapper.generate(
            input_ids, max_new_tokens=3, do_sample=False, seed=42
        )

    assert torch.equal(steered.token_ids, baseline.token_ids)
