"""Native repsteer methods with benchmark data and parameter configuration."""

from dataclasses import dataclass
from typing import Any, Callable

import repsteer as rs
from repsteer.selection import Candidate

from . import iti as iti_mod
from .learn import DEFAULT_BATCH, learn_artifacts, load_artifact
from .plans import add_plan
from ..runtime.search import candidates as site_candidates

@dataclass(frozen=True)
class Method:
    name: str
    learn: Callable[..., dict]
    load: Callable[..., Any]
    plan: Callable[..., rs.SteeringPlan]
    strength_key: str
    domains: tuple[str, ...] = ()

    def supports(self, domain: str) -> bool:
        return not self.domains or domain in self.domains

    def candidates(self, sites: list[str], grid) -> list[Candidate]:
        return site_candidates(sites, grid, self.strength_key)


def _native_method(name: str, learner_cls, **learner_kwargs) -> Method:
    def learn(wrapper, domain, target, sites, log=print, *, batch_size=DEFAULT_BATCH, pairs=None):
        pairs = domain.pairs(target) if pairs is None else pairs
        return learn_artifacts(wrapper, learner_cls, pairs, sites,
                               domain.rendering(), batch_size=batch_size,
                               learner_kwargs=learner_kwargs)

    return Method(
        name=name, learn=learn,
        load=lambda domain, model, target, site, *, training=None: load_artifact(
            domain.NAME, model, name, target, site, training=training),
        plan=lambda artifact, params: add_plan(artifact, params["strength"]),
        strength_key="r",
    )


METHODS = {
    "caa": _native_method("caa", rs.learners.CAA, normalize="l2"),
    "linear_probe": _native_method("linear_probe", rs.learners.LinearProbe, normalize="l2"),
    "lat": _native_method("lat", rs.learners.LAT, normalize="l2"),
    "pca": _native_method("pca", rs.learners.PCA, n_components=1, center=True),
    "iti": Method(
        name="iti", learn=iti_mod.learn,
        load=lambda domain, model, target, site, *, training=None: load_artifact(
            domain.NAME, model, "iti", target, site, training=training),
        plan=iti_mod.plan, strength_key="alpha", domains=iti_mod.SUPPORTED_DOMAINS,
    ),
}
