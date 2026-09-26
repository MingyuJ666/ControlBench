"""Pure-PyTorch activation steering learners."""

from .actadd import (
    CAA,
    ActAdd,
    ContrastiveActivationAddition,
    compute_actadd,
    paired_difference_mean,
)
from .base import ContrastiveLearner, Learner
from .diff_mean import DiffMean, compute_diff_mean, diff_mean
from .iti import ITI
from .lat import LAT, compute_lat, lat_components
from .pca import PCA, compute_pca, principal_components
from .probe import LinearProbe, LogisticProbe, fit_logistic_probe

__all__ = [
    "ActAdd",
    "CAA",
    "ContrastiveActivationAddition",
    "ContrastiveLearner",
    "DiffMean",
    "ITI",
    "LAT",
    "Learner",
    "LinearProbe",
    "LogisticProbe",
    "PCA",
    "compute_actadd",
    "compute_diff_mean",
    "compute_lat",
    "compute_pca",
    "diff_mean",
    "fit_logistic_probe",
    "lat_components",
    "paired_difference_mean",
    "principal_components",
]
