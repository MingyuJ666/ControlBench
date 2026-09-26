"""Versioned, fail-closed declarations of adapter representation surfaces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

_CAPABILITY_SCHEMA_VERSION = 1
_KNOWN_REQUIREMENTS = frozenset(
    {
        "residual_read",
        "residual_write",
        "sample_mapping",
        "modality_mapping",
        "head_result",
        "attention_bias",
    }
)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _require_schema_version(value: Any, *, name: str) -> int:
    """Require the exact integer schema version; bool and float are invalid."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(
            f"{name} must be an integer schema version, got {type(value).__name__}"
        )
    if value != _CAPABILITY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {name} {value!r}; supported: {_CAPABILITY_SCHEMA_VERSION}"
        )
    return value


def _normalise_names(
    value: tuple[str, ...] | list[str], *, name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, tuple | list):
        raise TypeError(f"{name} must be a list or tuple of names")
    names = tuple(value)
    if not names and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    for index, item in enumerate(names):
        _require_nonempty_string(item, name=f"{name}[{index}]")
    if len(set(names)) != len(names):
        raise ValueError(f"{name} cannot contain duplicate names")
    return names


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True, slots=True)
class SampleMappingCapability:
    """Declared target-row-to-sample mapping supported by an adapter.

    This record is intentionally narrower than general generation behavior.  It
    says only which semantic streams have an explicit sample mapping contract
    for adapter-owned representation surfaces.
    """

    condition_streams: tuple[str, ...] | list[str] = ("language",)
    target_streams: tuple[str, ...] | list[str] = ("language",)
    contract: str = "language_batch_rows"
    schema_version: int = _CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _require_schema_version(
            self.schema_version, name="SampleMappingCapability schema"
        )
        object.__setattr__(
            self,
            "condition_streams",
            _normalise_names(self.condition_streams, name="condition_streams"),
        )
        object.__setattr__(
            self,
            "target_streams",
            _normalise_names(self.target_streams, name="target_streams"),
        )
        object.__setattr__(
            self,
            "contract",
            _require_nonempty_string(
                self.contract, name="SampleMappingCapability.contract"
            ),
        )
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe declaration."""

        return {
            "schema_version": self.schema_version,
            "condition_streams": list(self.condition_streams),
            "target_streams": list(self.target_streams),
            "contract": self.contract,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SampleMappingCapability:
        """Load one strict mapping-capability declaration."""

        if not isinstance(value, dict):
            raise TypeError("SampleMappingCapability must be a JSON object")
        allowed = {"schema_version", "condition_streams", "target_streams", "contract"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "SampleMappingCapability has unsupported fields: " + ", ".join(unknown)
            )
        missing = sorted(allowed - set(value))
        if missing:
            raise ValueError(
                "SampleMappingCapability is missing required fields: "
                + ", ".join(missing)
            )
        if not isinstance(value["condition_streams"], list):
            raise TypeError("condition_streams must be a JSON list")
        if not isinstance(value["target_streams"], list):
            raise TypeError("target_streams must be a JSON list")
        return cls(
            schema_version=value["schema_version"],
            condition_streams=tuple(value["condition_streams"]),
            target_streams=tuple(value["target_streams"]),
            contract=value["contract"],
        )


@dataclass(frozen=True, slots=True)
class ModalityMappingCapability:
    """A tested adapter-owned image/patch mapping surface for a VLM adapter."""

    streams: tuple[str, ...] | list[str] = ("vision", "projector", "language")
    contract: str = "adapter_modality_map"
    schema_version: int = _CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _require_schema_version(
            self.schema_version, name="ModalityMappingCapability schema"
        )
        object.__setattr__(
            self, "streams", _normalise_names(self.streams, name="streams")
        )
        object.__setattr__(
            self,
            "contract",
            _require_nonempty_string(
                self.contract, name="ModalityMappingCapability.contract"
            ),
        )
        object.__setattr__(self, "schema_version", schema_version)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe declaration."""

        return {
            "schema_version": self.schema_version,
            "streams": list(self.streams),
            "contract": self.contract,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModalityMappingCapability:
        """Load one strict VLM modality-mapping declaration."""

        if not isinstance(value, dict):
            raise TypeError("ModalityMappingCapability must be a JSON object")
        allowed = {"schema_version", "streams", "contract"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "ModalityMappingCapability has unsupported fields: "
                + ", ".join(unknown)
            )
        missing = sorted(allowed - set(value))
        if missing:
            raise ValueError(
                "ModalityMappingCapability is missing required fields: "
                + ", ".join(missing)
            )
        if not isinstance(value["streams"], list):
            raise TypeError("streams must be a JSON list")
        return cls(
            schema_version=value["schema_version"],
            streams=tuple(value["streams"]),
            contract=value["contract"],
        )


@dataclass(frozen=True, slots=True)
class HeadResultCapability:
    """A tested pre-output-projection query-head representation surface."""

    component: str = "language.head_result"
    contract: str = "self_attn.o_proj.input_flattened_query_heads"
    schema_version: int = _CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "schema_version",
            _require_schema_version(
                self.schema_version, name="HeadResultCapability schema"
            ),
        )
        object.__setattr__(
            self,
            "component",
            _require_nonempty_string(
                self.component, name="HeadResultCapability.component"
            ),
        )
        object.__setattr__(
            self,
            "contract",
            _require_nonempty_string(
                self.contract, name="HeadResultCapability.contract"
            ),
        )
        if self.component != "language.head_result":
            raise ValueError(
                "HeadResultCapability.component must be 'language.head_result'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "component": self.component,
            "contract": self.contract,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> HeadResultCapability:
        if not isinstance(value, dict):
            raise TypeError("HeadResultCapability must be a JSON object")
        allowed = {"schema_version", "component", "contract"}
        unknown = sorted(set(value) - allowed)
        missing = sorted(allowed - set(value))
        if unknown or missing:
            details: list[str] = []
            if unknown:
                details.append("unsupported fields: " + ", ".join(unknown))
            if missing:
                details.append("missing required fields: " + ", ".join(missing))
            raise ValueError("HeadResultCapability has " + "; ".join(details))
        return cls(
            schema_version=value["schema_version"],
            component=value["component"],
            contract=value["contract"],
        )


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Immutable representation-surface capabilities for one adapter family.

    ``None`` for ``head_result`` and ``attention_bias`` explicitly means that
    the adapter does *not* declare those surfaces.  This declaration never
    implies support for beam handling, cache reordering, or other generation
    behavior outside the adapter's representation contract.
    """

    adapter_name: str
    residual_sites: tuple[str, ...] | list[str] = ()
    supports_residual_read: bool = False
    supports_residual_write: bool = False
    sample_mapping: SampleMappingCapability | None = None
    modality_mapping: ModalityMappingCapability | None = None
    head_result: HeadResultCapability | None = None
    attention_bias: None = None
    schema_version: int = _CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        schema_version = _require_schema_version(
            self.schema_version, name="AdapterCapabilities schema"
        )
        object.__setattr__(
            self,
            "adapter_name",
            _require_nonempty_string(self.adapter_name, name="adapter_name"),
        )
        if not isinstance(self.supports_residual_read, bool):
            raise TypeError("supports_residual_read must be a bool")
        if not isinstance(self.supports_residual_write, bool):
            raise TypeError("supports_residual_write must be a bool")
        sites = _normalise_names(
            self.residual_sites, name="residual_sites", allow_empty=True
        )
        if (self.supports_residual_read or self.supports_residual_write) and not sites:
            raise ValueError(
                "a residual read/write capability must declare at least one "
                "residual site"
            )
        if self.sample_mapping is not None and not isinstance(
            self.sample_mapping, SampleMappingCapability
        ):
            raise TypeError("sample_mapping must be SampleMappingCapability or None")
        if self.modality_mapping is not None and not isinstance(
            self.modality_mapping, ModalityMappingCapability
        ):
            raise TypeError(
                "modality_mapping must be ModalityMappingCapability or None"
            )
        if self.head_result is not None and not isinstance(
            self.head_result, HeadResultCapability
        ):
            raise TypeError("head_result must be HeadResultCapability or None")
        if self.attention_bias is not None:
            raise ValueError(
                "attention_bias is unsupported by this release and must be None"
            )
        object.__setattr__(self, "residual_sites", sites)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def canonical(self) -> str:
        """Return deterministic JSON suitable for diagnostics or snapshots."""

        return _canonical_json(self.to_dict())

    @property
    def fingerprint(self) -> str:
        """Return a stable declaration digest."""

        return "sha256:" + hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the full versioned declaration without executable payloads."""

        return {
            "schema_version": self.schema_version,
            "adapter_name": self.adapter_name,
            "residual_sites": list(self.residual_sites),
            "supports_residual_read": self.supports_residual_read,
            "supports_residual_write": self.supports_residual_write,
            "sample_mapping": (
                None if self.sample_mapping is None else self.sample_mapping.to_dict()
            ),
            "modality_mapping": (
                None
                if self.modality_mapping is None
                else self.modality_mapping.to_dict()
            ),
            "head_result": (
                None if self.head_result is None else self.head_result.to_dict()
            ),
            "attention_bias": None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AdapterCapabilities:
        """Load a strict capability declaration and reject unknown schemas."""

        if not isinstance(value, dict):
            raise TypeError("AdapterCapabilities must be a JSON object")
        allowed = {
            "schema_version",
            "adapter_name",
            "residual_sites",
            "supports_residual_read",
            "supports_residual_write",
            "sample_mapping",
            "modality_mapping",
            "head_result",
            "attention_bias",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(
                "AdapterCapabilities has unsupported fields: " + ", ".join(unknown)
            )
        missing = sorted(allowed - set(value))
        if missing:
            raise ValueError(
                "AdapterCapabilities is missing required fields: " + ", ".join(missing)
            )
        if not isinstance(value["residual_sites"], list):
            raise TypeError("residual_sites must be a JSON list")
        sample_mapping = value["sample_mapping"]
        modality_mapping = value["modality_mapping"]
        head_result = value["head_result"]
        if sample_mapping is not None and not isinstance(sample_mapping, dict):
            raise TypeError("sample_mapping must be a JSON object or null")
        if modality_mapping is not None and not isinstance(modality_mapping, dict):
            raise TypeError("modality_mapping must be a JSON object or null")
        if head_result is not None and not isinstance(head_result, dict):
            raise TypeError("head_result must be a JSON object or null")
        return cls(
            schema_version=value["schema_version"],
            adapter_name=value["adapter_name"],
            residual_sites=tuple(value["residual_sites"]),
            supports_residual_read=value["supports_residual_read"],
            supports_residual_write=value["supports_residual_write"],
            sample_mapping=(
                None
                if sample_mapping is None
                else SampleMappingCapability.from_dict(sample_mapping)
            ),
            modality_mapping=(
                None
                if modality_mapping is None
                else ModalityMappingCapability.from_dict(modality_mapping)
            ),
            head_result=(
                None
                if head_result is None
                else HeadResultCapability.from_dict(head_result)
            ),
            attention_bias=value["attention_bias"],
        )

    def supports(self, requirement: str) -> bool:
        """Return whether one known surface is declared; unknown names are false."""

        if requirement == "residual_read":
            return self.supports_residual_read
        if requirement == "residual_write":
            return self.supports_residual_write
        if requirement == "sample_mapping":
            return self.sample_mapping is not None
        if requirement == "modality_mapping":
            return self.modality_mapping is not None
        if requirement == "head_result":
            return self.head_result is not None
        if requirement == "attention_bias":
            return False
        return False

    def supports_requirement(self, requirement: str) -> bool:
        """Alias for :meth:`supports` used by future compiler callers."""

        return self.supports(requirement)

    def require(self, requirement: str) -> None:
        """Fail closed with a deterministic diagnostic for an unmet requirement."""

        if requirement not in _KNOWN_REQUIREMENTS:
            known = ", ".join(sorted(_KNOWN_REQUIREMENTS))
            raise ValueError(
                "unknown adapter capability requirement "
                f"{requirement!r}; known: {known}"
            )
        if not self.supports(requirement):
            raise ValueError(self.explain(requirement))

    def explain(self, requirement: str | None = None) -> str:
        """Return a stable, concise capability diagnostic."""

        if requirement is None:
            return (
                f"AdapterCapabilities(adapter={self.adapter_name}, "
                f"residual_read={self.supports_residual_read}, "
                f"residual_write={self.supports_residual_write}, "
                f"sample_mapping={self.sample_mapping is not None}, "
                f"modality_mapping={self.modality_mapping is not None}, "
                f"head_result={self.head_result is not None}, attention_bias=False)"
            )
        if requirement not in _KNOWN_REQUIREMENTS:
            return (
                f"Adapter {self.adapter_name!r} does not recognize capability "
                f"requirement {requirement!r}; fail closed"
            )
        if self.supports(requirement):
            return f"Adapter {self.adapter_name!r} declares {requirement!r} support"
        if requirement in {"head_result", "attention_bias"}:
            return (
                f"Adapter {self.adapter_name!r} does not declare {requirement!r}; "
                "this representation surface is unsupported"
            )
        return (
            f"Adapter {self.adapter_name!r} does not declare {requirement!r}; "
            "fail closed rather than approximating the surface"
        )


__all__ = [
    "AdapterCapabilities",
    "HeadResultCapability",
    "ModalityMappingCapability",
    "SampleMappingCapability",
]
