"""Domain configuration for native repsteer ITI profiles."""

import repsteer as rs
from repsteer.selection import Candidate

from .learn import CAPTURE_DTYPE, DEFAULT_BATCH

DEFAULT_TOP_K = 48
DEFAULT_SEED = 42
SUPPORTED_DOMAINS = ("instruction_following", "tool_use", "tool_use_multi_turn")


def profile_key(top_k: int, seed: int) -> str:
    return f"heads_k{top_k}_seed{seed}"


def learn(wrapper, domain, target, sites=(), log=print, *, batch_size=DEFAULT_BATCH,
          pairs=None, top_k=DEFAULT_TOP_K, seed=DEFAULT_SEED):
    if domain.NAME not in SUPPORTED_DOMAINS:
        raise ValueError(f"ITI is not supported for {domain.NAME}")
    if sites:
        raise ValueError("ITI selects heads across all layers; do not supply layer sites")
    pairs = domain.pairs(target) if pairs is None else pairs
    log(f"  ITI: learning from {len(pairs)} pairs across all layers, top_k={top_k}, seed={seed}")
    profile = rs.learners.ITI(
        top_k=top_k, seed=seed, batch_size=batch_size, capture_dtype=CAPTURE_DTYPE,
        **domain.rendering().learner_kwargs(),
    ).fit(wrapper, pairs)
    log(f"  ITI: selected {len(profile.heads)} heads across {len(profile.layers)} layers")
    return {profile_key(top_k, seed): profile}


def candidates(top_k: int, seed: int, strengths) -> list[Candidate]:
    key = profile_key(top_k, seed)
    return [Candidate(id=f"{key}_alpha{strength:g}", parameters={
        "site": key, "layer": None, "strength": float(strength),
        "strength_unit": "iti_alpha", "top_k": top_k, "iti_seed": seed,
    }) for strength in strengths]


def plan(profile, params):
    return rs.recipes.iti(profile, strength=params["strength"])
