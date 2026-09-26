"""Sparse-autoencoder integrations and feature selection."""

from .adapter import SAEAdapter, validate_adapter, validate_feature_id
from .custom import CustomSAEAdapter, FunctionalSAEAdapter
from .feature_selection import (
    CausalEvaluator,
    feature_artifact,
    select_causal_features,
    select_feature,
    select_supervised_features,
    supervised_feature_scores,
)
from .registry import available_providers, load, register_provider, unregister_provider
from .saelens import SAELensAdapter

__all__ = [
    "CausalEvaluator",
    "CustomSAEAdapter",
    "FunctionalSAEAdapter",
    "SAEAdapter",
    "SAELensAdapter",
    "available_providers",
    "feature_artifact",
    "load",
    "register_provider",
    "select_causal_features",
    "select_feature",
    "select_supervised_features",
    "supervised_feature_scores",
    "unregister_provider",
    "validate_adapter",
    "validate_feature_id",
]
