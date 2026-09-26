from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, Sequence
from copy import copy
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Literal, Protocol, cast

from torch import Tensor, nn

from .capabilities import (
    AdapterCapabilities,
    HeadResultCapability,
    SampleMappingCapability,
)


def _site_error(message: str) -> Exception:
    try:
        from repsteer.core.errors import SiteResolutionError

        return SiteResolutionError(message)
    except (ImportError, AttributeError):
        return ValueError(message)


class TensorAccessor(Protocol):
    """Read and immutably rebuild one explicitly identified tensor."""

    @property
    def description(self) -> str: ...

    def read(self, container: Any) -> Tensor: ...

    def rebuild(self, container: Any, tensor: Tensor) -> Any: ...


def _child(container: Any, key: int | str) -> Any:
    if isinstance(key, int):
        if not isinstance(container, tuple | list):
            raise _site_error(
                f"tensor path expected tuple/list before index {key}, got "
                f"{type(container).__name__}"
            )
        try:
            return container[key]
        except IndexError as exc:
            raise _site_error(f"tensor path index {key} is out of range") from exc
    if isinstance(container, Mapping):
        if key not in container:
            raise _site_error(f"tensor path key {key!r} is absent")
        return container[key]
    if not hasattr(container, key):
        raise _site_error(
            f"tensor path attribute {key!r} is absent on {type(container).__name__}"
        )
    return getattr(container, key)


def _replace_child(container: Any, key: int | str, value: Any) -> Any:
    if isinstance(key, int):
        if isinstance(container, tuple):
            values = list(container)
            values[key] = value
            if hasattr(container, "_fields"):
                return type(container)(*values)
            return tuple(values)
        if isinstance(container, list):
            values = list(container)
            values[key] = value
            return values
        raise _site_error(
            f"cannot rebuild index {key} on {type(container).__name__}; "
            "adapter tensor paths must be explicit"
        )

    if isinstance(container, Mapping):
        # Transformers ModelOutput objects are Mapping + dataclass.  Prefer
        # dataclass replacement because it keeps attribute and mapping views in
        # sync; plain mappings are rebuilt without mutating user-owned values.
        if is_dataclass(container):
            try:
                return replace(cast(Any, container), **{key: value})
            except (TypeError, ValueError):
                pass
        try:
            rebuilt = copy(container)
            if isinstance(rebuilt, MutableMapping):
                rebuilt[key] = value
                return rebuilt
        except (TypeError, AttributeError):
            pass
        mapping_values: dict[Any, Any] = dict(container)
        mapping_values[key] = value
        return mapping_values

    if is_dataclass(container):
        names = {field.name for field in fields(container)}
        if key not in names:
            raise _site_error(
                f"cannot rebuild unknown dataclass field {key!r} on "
                f"{type(container).__name__}"
            )
        return replace(cast(Any, container), **{key: value})

    # Arbitrary output objects are copied before assignment.  Refusing to
    # mutate the original is important for cache-bearing ModelOutput objects.
    if hasattr(container, key):
        try:
            rebuilt = copy(container)
            setattr(rebuilt, key, value)
            return rebuilt
        except (TypeError, AttributeError) as exc:
            raise _site_error(
                f"attribute {key!r} on {type(container).__name__} cannot be rebuilt"
            ) from exc
    raise _site_error(f"cannot rebuild tensor path key {key!r}")


@dataclass(frozen=True)
class PathTensorAccessor:
    path: tuple[int | str, ...] = ()

    @property
    def description(self) -> str:
        return "root" if not self.path else ".".join(map(str, self.path))

    def read(self, container: Any) -> Tensor:
        value = container
        for key in self.path:
            value = _child(value, key)
        if not isinstance(value, Tensor):
            raise _site_error(
                f"resolved tensor path {self.description!r} produced "
                f"{type(value).__name__}, not torch.Tensor"
            )
        return value

    def rebuild(self, container: Any, tensor: Tensor) -> Any:
        if not isinstance(tensor, Tensor):
            raise TypeError("replacement must be a torch.Tensor")
        if not self.path:
            if not isinstance(container, Tensor):
                raise _site_error(
                    f"resolved root is {type(container).__name__}, not torch.Tensor"
                )
            return tensor
        ancestors: list[tuple[Any, int | str]] = []
        value = container
        for key in self.path:
            ancestors.append((value, key))
            value = _child(value, key)
        if not isinstance(value, Tensor):
            raise _site_error(
                f"resolved tensor path {self.description!r} produced "
                f"{type(value).__name__}, not torch.Tensor"
            )
        rebuilt: Any = tensor
        for parent, key in reversed(ancestors):
            rebuilt = _replace_child(parent, key, rebuilt)
        return rebuilt


@dataclass(frozen=True)
class RootOrFirstTensorAccessor:
    """Explicit adapter contract for APIs that versioned Tensor -> tuple[0].

    This is deliberately *not* a recursive "find the first tensor" helper.  It
    accepts exactly a tensor or a tuple/list whose first item is a tensor, and
    safely preserves every other output item when rebuilding.
    """

    extra_path: tuple[int | str, ...] = ()

    @property
    def description(self) -> str:
        suffix = ".".join(map(str, self.extra_path))
        return "tensor-or-tuple[0]" + (f".{suffix}" if suffix else "")

    def _path(self, container: Any) -> tuple[int | str, ...]:
        if isinstance(container, Tensor):
            return self.extra_path
        if isinstance(container, tuple | list) and container:
            if not isinstance(container[0], Tensor) and not self.extra_path:
                raise _site_error(
                    f"adapter expected tensor at tuple/list item 0, got "
                    f"{type(container[0]).__name__}"
                )
            return (0, *self.extra_path)
        raise _site_error(
            "adapter expected a Tensor or a non-empty tuple/list with its "
            f"hidden state at item 0, got {type(container).__name__}"
        )

    def read(self, container: Any) -> Tensor:
        return PathTensorAccessor(self._path(container)).read(container)

    def rebuild(self, container: Any, tensor: Tensor) -> Any:
        return PathTensorAccessor(self._path(container)).rebuild(container, tensor)


@dataclass(frozen=True)
class ResolvedSite:
    site: Any
    module: nn.Module
    module_path: str
    hook_kind: Literal["forward_pre", "forward"]
    tensor_accessor: TensorAccessor
    hidden_dim: int
    architecture_name: str

    @property
    def read_tensor(self) -> TensorAccessor:
        return self.tensor_accessor

    @property
    def write_tensor(self) -> TensorAccessor:
        return self.tensor_accessor

    def read(self, container: Any) -> Tensor:
        return self.tensor_accessor.read(container)

    def rebuild(self, container: Any, tensor: Tensor) -> Any:
        return self.tensor_accessor.rebuild(container, tensor)

    @property
    def conflict_key(self) -> tuple[int, str, str]:
        return (id(self.module), self.hook_kind, self.tensor_accessor.description)


class ArchitectureAdapter(ABC):
    architecture_name = "unknown"

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Return a fail-closed declaration for adapter-owned surfaces.

        Custom adapters inherit an intentionally empty declaration until they
        override it with a tested representation contract.
        """

        return AdapterCapabilities(adapter_name=self.architecture_name)

    @abstractmethod
    def supports(self, model: nn.Module) -> bool:
        raise NotImplementedError

    @abstractmethod
    def resolve(self, model: nn.Module, site: Any) -> ResolvedSite:
        raise NotImplementedError

    @abstractmethod
    def hidden_size(self, model: nn.Module, site: Any | None = None) -> int:
        raise NotImplementedError

    def build_modality_map(
        self,
        batch: Mapping[str, Tensor],
        *,
        model: nn.Module | None = None,
    ) -> Any | None:
        del batch, model
        return None

    def generation_phase(
        self, kwargs: Mapping[str, Any]
    ) -> Literal["prefill", "decode"]:
        past = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        return "decode" if _cache_has_content(past) else "prefill"


def _cache_has_content(cache: Any) -> bool:
    if cache is None:
        return False
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length()) > 0
        except (TypeError, ValueError, RuntimeError):
            return True
    if isinstance(cache, Sequence) and not isinstance(cache, str | bytes):
        if len(cache) == 0:
            return False
        first = cache[0]
        if isinstance(first, Sequence) and len(first) > 0:
            first = first[0]
        if isinstance(first, Tensor):
            return first.numel() > 0 and (first.ndim < 3 or first.shape[-2] > 0)
        return True
    return True


class DecoderOnlyAdapter(ArchitectureAdapter):
    """Shared semantic mapping for modern HF decoder-only transformer blocks."""

    model_types: frozenset[str] = frozenset()
    class_prefixes: tuple[str, ...] = ()
    supported_components = frozenset(
        {
            "resid_pre",
            "attn_out",
            "head_result",
            "resid_mid",
            "mlp_out",
            "resid_post",
        }
    )
    component_io = {
        "resid_pre": "output",
        "attn_out": "output",
        "head_result": "input",
        "resid_mid": "output",
        "mlp_out": "output",
        "resid_post": "output",
    }

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declare the tested residual and language sample-mapping contracts."""

        ordered_components = (
            "resid_pre",
            "attn_out",
            "resid_mid",
            "mlp_out",
            "resid_post",
        )
        return AdapterCapabilities(
            adapter_name=self.architecture_name,
            residual_sites=tuple(
                f"language.{component}"
                for component in ordered_components
                if component in self.supported_components
            ),
            supports_residual_read=True,
            supports_residual_write=True,
            sample_mapping=SampleMappingCapability(
                condition_streams=("language",),
                target_streams=("language",),
                contract="language_batch_rows",
            ),
            head_result=HeadResultCapability(),
        )

    def supports(self, model: nn.Module) -> bool:
        model_type = str(
            getattr(getattr(model, "config", None), "model_type", "")
        ).lower()
        if model_type in self.model_types:
            return True
        name = type(model).__name__.lower()
        return any(name.startswith(prefix.lower()) for prefix in self.class_prefixes)

    def _layers(self, model: nn.Module) -> tuple[Any, str]:
        candidates = (("model.layers", ("model", "layers")), ("layers", ("layers",)))
        for display, path in candidates:
            value: Any = model
            try:
                for part in path:
                    value = getattr(value, part)
            except AttributeError:
                continue
            if isinstance(value, nn.ModuleList | list | tuple):
                return value, display
        raise _site_error(
            f"{self.architecture_name} adapter could not find decoder layers at "
            "model.layers or layers"
        )

    def hidden_size(self, model: nn.Module, site: Any | None = None) -> int:
        config = getattr(model, "config", None)
        for owner in (config, getattr(config, "text_config", None)):
            size = getattr(owner, "hidden_size", None)
            if size is not None:
                return int(size)
        raise _site_error(
            f"{self.architecture_name} adapter could not determine config.hidden_size"
        )

    def resolve(self, model: nn.Module, site: Any) -> ResolvedSite:
        stream = getattr(site, "stream", "language")
        component = getattr(site, "component", None)
        layer_index = getattr(site, "layer", None)
        unit = getattr(site, "unit", None)
        if stream != "language":
            raise _site_error(
                f"{self.architecture_name} adapter only supports the language stream, "
                f"got {stream!r}"
            )
        if component not in self.supported_components:
            supported = ", ".join(sorted(self.supported_components))
            raise _site_error(
                f"component {component!r} is not supported by the "
                f"{self.architecture_name} adapter; supported: {supported}"
            )
        expected_io = self.component_io[component]
        actual_io = getattr(site, "io", expected_io)
        if actual_io != expected_io:
            raise _site_error(
                f"semantic site {component!r} requires io={expected_io!r}, got "
                f"io={actual_io!r}; the decoder-only semantic-site contract does not "
                "support module-I/O overrides"
            )
        if not isinstance(layer_index, int):
            raise _site_error(
                f"site {component!r} requires an integer layer, got {layer_index!r}"
            )
        if unit is not None:
            raise _site_error(
                f"unit-level addressing is not supported for {component!r}"
            )

        layers, layers_path = self._layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise _site_error(
                f"layer {layer_index} is out of range for {self.architecture_name}; "
                f"valid range is [0, {len(layers) - 1}]"
            )
        layer = layers[layer_index]
        prefix = f"{layers_path}.{layer_index}"
        explicit_path = tuple(getattr(site, "tensor_path", ()) or ())

        if component == "resid_pre":
            module, path, kind = layer, prefix, "forward_pre"
            accessor: TensorAccessor = PathTensorAccessor(explicit_path or (0,))
        elif component == "attn_out":
            module = getattr(layer, "self_attn", None)
            if not isinstance(module, nn.Module):
                raise _site_error(
                    f"decoder layer {layer_index} has no self_attn module"
                )
            path, kind = f"{prefix}.self_attn", "forward"
            accessor = cast(
                TensorAccessor,
                (
                    PathTensorAccessor(explicit_path)
                    if explicit_path
                    else RootOrFirstTensorAccessor()
                ),
            )
        elif component == "head_result":
            attention = getattr(layer, "self_attn", None)
            if not isinstance(attention, nn.Module):
                raise _site_error(
                    f"decoder layer {layer_index} has no self_attn module"
                )
            module = getattr(attention, "o_proj", None)
            if not isinstance(module, nn.Module):
                raise _site_error(
                    f"decoder layer {layer_index} self_attn has no o_proj module"
                )
            # nn.Module's own __getattr__ is typed Tensor | Module, not int;
            # o_proj's concrete type (nn.Linear and friends) isn't statically
            # known here, so read defensively and let int() reject whatever
            # comes back instead of trusting an inferred attribute type.
            raw_in_features = getattr(module, "in_features", None)
            try:
                input_width = int(cast(Any, raw_in_features))
            except (TypeError, ValueError) as exc:
                raise _site_error(
                    f"{self.architecture_name} o_proj must expose integer in_features"
                ) from exc
            query_heads = _query_head_count(model)
            if input_width <= 0 or input_width % query_heads:
                raise _site_error(
                    "o_proj input width must be divisible by "
                    "config.num_attention_heads "
                    f"for head_result; got {input_width} and {query_heads}"
                )
            path, kind = f"{prefix}.self_attn.o_proj", "forward_pre"
            accessor = PathTensorAccessor(explicit_path or (0,))
            head_dim = input_width // query_heads
        elif component == "resid_mid":
            module = getattr(layer, "post_attention_layernorm", None)
            if not isinstance(module, nn.Module):
                raise _site_error(
                    f"decoder layer {layer_index} has no "
                    "post_attention_layernorm module"
                )
            path, kind = f"{prefix}.post_attention_layernorm", "forward_pre"
            accessor = PathTensorAccessor(explicit_path or (0,))
        elif component == "mlp_out":
            module = getattr(layer, "mlp", None)
            if not isinstance(module, nn.Module):
                raise _site_error(f"decoder layer {layer_index} has no mlp module")
            path, kind = f"{prefix}.mlp", "forward"
            accessor = cast(
                TensorAccessor,
                (
                    PathTensorAccessor(explicit_path)
                    if explicit_path
                    else RootOrFirstTensorAccessor()
                ),
            )
        else:  # resid_post
            module, path, kind = layer, prefix, "forward"
            accessor = cast(
                TensorAccessor,
                (
                    PathTensorAccessor(explicit_path)
                    if explicit_path
                    else RootOrFirstTensorAccessor()
                ),
            )

        return ResolvedSite(
            site=site,
            module=module,
            module_path=path,
            hook_kind=kind,  # type: ignore[arg-type]
            tensor_accessor=accessor,
            hidden_dim=(
                head_dim
                if component == "head_result"
                else self.hidden_size(model, site)
            ),
            architecture_name=self.architecture_name,
        )


def _query_head_count(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for owner in (
        config,
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
    ):
        value = getattr(owner, "num_attention_heads", None)
        if value is None:
            continue
        if isinstance(value, bool):
            break
        count = int(value)
        if count > 0:
            return count
    raise _site_error(
        "adapter could not determine a positive config.num_attention_heads"
    )
