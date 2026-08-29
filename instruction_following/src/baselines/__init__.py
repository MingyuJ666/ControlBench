"""Registry mapping method keys to their implementation modules.

Resolution is lazy on purpose: a method's optional dependencies (a probe
library, an SAE checkpoint, training code) must never break a run of an
unrelated method, so a module is imported only when its key is selected.
``availability`` reports what each method would need without importing it into
the caller's process.
"""

import importlib
from types import ModuleType

REGISTRY = {
    "caa": "baselines.caa",
    "probe": "baselines.probe",
    "lat_pca": "baselines.lat_pca",
    "iti": "baselines.iti",
    "sae": "baselines.sae",
    # Same dictionary, same heads, same direction and dose as the two above;
    # they differ only in which unit is selected, which is the measurement.
    "iti_disc": "baselines.iti_disc",
    "sae_disc": "baselines.sae_disc",
}

# Declared here so status can describe a method before its module exists.
PLANNED = {
    "reft": "trained rank-1 intervention (not implemented)",
}

DEFAULT_BASELINE = "caa"


def keys() -> list[str]:
    """Registered method keys, in registration order."""
    return list(REGISTRY)


def get(key: str) -> ModuleType:
    """Import and return one method module."""
    if key not in REGISTRY:
        known = ", ".join(REGISTRY)
        planned = ", ".join(PLANNED)
        raise KeyError(
            f"unknown method {key!r}; registered: {known}"
            + (f"; planned: {planned}" if planned else "")
        )
    module = importlib.import_module(REGISTRY[key])
    if getattr(module, "KEY", None) != key:
        raise RuntimeError(
            f"module {REGISTRY[key]} declares KEY "
            f"{getattr(module, 'KEY', None)!r}, expected {key!r}"
        )
    return module


def availability() -> dict[str, str]:
    """Report per method whether it can run, importing each in turn."""
    report = {}
    for key in REGISTRY:
        try:
            reason = get(key).check_available()
        except ImportError as error:
            reason = f"import failed: {error}"
        report[key] = "available" if reason is None else reason
    for key, note in PLANNED.items():
        report[key] = note
    return report
