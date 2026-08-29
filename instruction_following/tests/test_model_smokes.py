"""The tests that load the real 8B model. Opt-in with RUN_MODEL_SMOKE=1: they
are slow, and they are the only place the fake-model arithmetic is checked
against the thing it stands in for.
"""
import os
import sys
from pathlib import Path
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines.caa import producer  # noqa: E402
from baselines import caa  # noqa: E402
from kernel.config import slug  # noqa: E402
from baselines.shared import capture  # noqa: E402
from kernel.splits import extraction_pairs  # noqa: E402
from kernel.config import TARGET_MODEL, slug  # noqa: E402
from kernel.generate import render_chat  # noqa: E402
from kernel.generate import GenConfig, Generator  # noqa: E402
from kernel.tasks import make_prompted_prompt  # noqa: E402
import json
from kernel.reference_conditions import (  # noqa: E402
    CONDITIONS,
    generate_reference_conditions,
    outputs_path,
)
from kernel.tasks import TASKS  # noqa: E402


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="set RUN_MODEL_SMOKE=1 to load Qwen3-8B",
)
def test_qwen3_real_hidden_state_extraction_smoke(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real model smoke test")

    pairs = extraction_pairs("no_comma")[:4]
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    monkeypatch.setattr(capture, "extraction_pairs", lambda task: pairs)

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    destination = producer.produce_task(
        model,
        tokenizer,
        slug(TARGET_MODEL),
        "no_comma",
        batch=2,
    )
    record = torch.load(destination, weights_only=True)

    assert destination.is_file()
    assert record["n_pairs"] == 4
    assert record["vector"].shape == (
        model.config.num_hidden_layers + 1,
        model.config.hidden_size,
    )
    assert record["vector"].dtype == torch.float32
    assert torch.isfinite(record["vector"]).all()
    assert len(record["split_half_cosine"]) == (
        model.config.num_hidden_layers + 1
    )
    assert all(
        float("-inf") < value < float("inf")
        for value in record["split_half_cosine"]
    )


def _caa_intervention(model, vector, layer, alpha, positions="all"):
    """The context CAA installs, reached through its public interface.

    The smoke checks the arithmetic of the addition, so it asks for every
    position; which positions the benchmark steers is decided in
    baselines.shared.positions and tested there."""
    from baselines.shared.residual_add import add_direction

    caa.make_intervention(
        model, {"vector": vector}, {"layer": layer, "alpha": alpha}
    )
    return add_direction(
        model, alpha * vector[layer], layer, positions=positions
    )


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="set RUN_MODEL_SMOKE=1 to load Qwen3-8B",
)
def test_qwen3_real_injection_changes_then_restores_logits(
    tmp_path, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real model smoke test")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer = max(1, n_layers // 2)
    vector = torch.zeros(n_layers + 1, hidden_size)
    vector[layer] = torch.linspace(-1.0, 1.0, hidden_size)

    model_slug = slug(TARGET_MODEL)
    path = tmp_path / model_slug / "caa" / "no_comma.pt"
    path.parent.mkdir(parents=True)
    torch.save({
        "vector": vector,
        "extraction_position": "last_prompt_token",
        "method": "caa_diff_in_means",
        "task": "no_comma",
        "model": model.config._name_or_path,
    }, path)
    monkeypatch.setattr(
        caa, "artifact_path",
        lambda model, task: tmp_path / model / "caa" / f"{task}.pt",
    )
    loaded = caa.load_artifact(model_slug, "no_comma")["vector"]

    prompt = render_chat(tokenizer, "Why is the sky blue?")
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        plain = model(**encoded, use_cache=False).logits
        with _caa_intervention(model, loaded, layer=layer, alpha=2.0):
            steered = model(**encoded, use_cache=False).logits
        restored = model(**encoded, use_cache=False).logits

    assert torch.isfinite(steered).all()
    assert not torch.equal(plain, steered)
    assert torch.equal(plain, restored)


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="set RUN_MODEL_SMOKE=1 to load Qwen3-8B",
)
def test_qwen3_real_generation_smoke():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real model smoke test")

    base = "Why is the sky blue?"
    prompts = [base, make_prompted_prompt(base, "no_comma")]
    gen = Generator(GenConfig(max_new_tokens=32, batch_size=2))
    results = gen.generate(prompts, progress=False)

    assert len(results) == len(prompts)
    assert all(set(row) == {"output", "truncated", "n_new_tokens"}
               for row in results)
    assert all(isinstance(row["output"], str) and row["output"] for row in results)
    assert all("<think>" not in row["output"] for row in results)


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE") != "1",
    reason="set RUN_MODEL_SMOKE=1 to load Qwen3-8B",
)
def test_qwen3_real_reference_generation_smoke(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the real model smoke test")

    generator = Generator(GenConfig(
        model=TARGET_MODEL,
        max_new_tokens=32,
        batch_size=6,
    ))
    run = generate_reference_conditions(
        generator,
        TARGET_MODEL,
        slug(TARGET_MODEL),
        "dev",
        limit=1,
        root=tmp_path,
        progress=False,
    )

    assert run["n_prompts"] == 1
    assert run["n_unique_generations"] == 6
    assert run["generation_calls"] == 2
    for task in TASKS:
        for condition in CONDITIONS:
            path = outputs_path(
                slug(TARGET_MODEL), "dev", task, condition, tmp_path
            )
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(records) == 1
            assert records[0]["output"]
            assert "<think>" not in records[0]["output"]
