"""Shared semantic-site resolution for vision-language architectures."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from torch import nn

from ..base import (
    DecoderOnlyAdapter,
    PathTensorAccessor,
    ResolvedSite,
    RootOrFirstTensorAccessor,
    TensorAccessor,
    _site_error,
)
from ..capabilities import AdapterCapabilities, ModalityMappingCapability


def resolve_attribute_path(
    model: nn.Module,
    candidates: Sequence[tuple[str, tuple[str, ...]]],
    *,
    description: str,
) -> tuple[Any, str]:
    """Resolve the first explicit architecture path without recursive guessing."""

    for display, path in candidates:
        value: Any = model
        try:
            for part in path:
                value = getattr(value, part)
        except AttributeError:
            continue
        return value, display
    choices = ", ".join(display for display, _ in candidates)
    raise _site_error(f"could not find {description}; checked {choices}")


class VisionLanguageAdapter(DecoderOnlyAdapter):
    """Decoder-only language sites plus vision-block and projector sites."""

    language_layer_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    vision_layer_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()
    projector_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def capabilities(self) -> AdapterCapabilities:
        """Declare tested VLM residual and adapter modality-map contracts.

        The mapping contract remains language-target-only for cached sequence
        decisions.  Vision/projector ``ModalityMap`` support is declared
        separately and must not be mistaken for target-row-to-sample mapping.
        """

        text_capabilities = super().capabilities
        return AdapterCapabilities(
            adapter_name=self.architecture_name,
            residual_sites=(
                *text_capabilities.residual_sites,
                "vision.vision_resid",
                "projector.projector_in",
                "projector.projector_out",
            ),
            supports_residual_read=True,
            supports_residual_write=True,
            sample_mapping=text_capabilities.sample_mapping,
            head_result=text_capabilities.head_result,
            modality_mapping=ModalityMappingCapability(
                streams=("vision", "projector", "language"),
                contract=f"{self.architecture_name}_modality_map_static_images",
            ),
        )

    def supports(self, model: nn.Module) -> bool:
        config = getattr(model, "config", None)
        model_type = str(getattr(config, "model_type", "")).lower()
        has_subconfigs = getattr(config, "vision_config", None) is not None and (
            getattr(config, "text_config", None) is not None
            or getattr(config, "llm_config", None) is not None
        )
        if model_type in self.model_types:
            return has_subconfigs
        name = type(model).__name__.lower()
        class_match = any(
            name.startswith(prefix.lower()) for prefix in self.class_prefixes
        )
        # Vision-only tower classes commonly share the family prefix/model_type.
        # Requiring both sub-configs keeps registry selection on full VLMs.
        return class_match and has_subconfigs

    def _layers(self, model: nn.Module) -> tuple[Any, str]:
        layers, path = resolve_attribute_path(
            model,
            self.language_layer_paths,
            description=f"{self.architecture_name} language layers",
        )
        if not isinstance(layers, nn.ModuleList | list | tuple):
            raise _site_error(
                f"{self.architecture_name} language layers at {path} are "
                f"{type(layers).__name__}, not a module sequence"
            )
        return layers, path

    def _vision_layers(self, model: nn.Module) -> tuple[Any, str]:
        layers, path = resolve_attribute_path(
            model,
            self.vision_layer_paths,
            description=f"{self.architecture_name} vision layers",
        )
        if not isinstance(layers, nn.ModuleList | list | tuple):
            raise _site_error(
                f"{self.architecture_name} vision layers at {path} are "
                f"{type(layers).__name__}, not a module sequence"
            )
        return layers, path

    def _projector(self, model: nn.Module) -> tuple[nn.Module, str]:
        projector, path = resolve_attribute_path(
            model,
            self.projector_paths,
            description=f"{self.architecture_name} multimodal projector",
        )
        if not isinstance(projector, nn.Module):
            raise _site_error(
                f"{self.architecture_name} projector at {path} is "
                f"{type(projector).__name__}, not torch.nn.Module"
            )
        return projector, path

    def _vision_hidden_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        vision_config = getattr(config, "vision_config", None)
        value = getattr(vision_config, "hidden_size", None)
        if value is None:
            raise _site_error(
                f"{self.architecture_name} config.vision_config.hidden_size is absent"
            )
        return int(value)

    def _projector_input_size(self, model: nn.Module) -> int:
        return self._vision_hidden_size(model)

    def _projector_output_size(self, model: nn.Module) -> int:
        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None) or getattr(
            config, "llm_config", None
        )
        value = getattr(text_config, "hidden_size", None)
        if value is None:
            raise _site_error(
                f"{self.architecture_name} config.text_config.hidden_size is absent"
            )
        return int(value)

    def hidden_size(self, model: nn.Module, site: Any | None = None) -> int:
        stream = getattr(site, "stream", "language") if site is not None else "language"
        component = getattr(site, "component", None)
        if stream == "vision":
            return self._vision_hidden_size(model)
        if stream == "projector":
            return (
                self._projector_input_size(model)
                if component == "projector_in"
                else self._projector_output_size(model)
            )
        if stream != "language":
            raise _site_error(
                f"{self.architecture_name} does not expose hidden size for "
                f"stream {stream!r}"
            )
        return self._projector_output_size(model)

    def resolve(self, model: nn.Module, site: Any) -> ResolvedSite:
        stream = getattr(site, "stream", "language")
        if stream == "language":
            return super().resolve(model, site)
        if stream == "vision":
            return self._resolve_vision(model, site)
        if stream == "projector":
            return self._resolve_projector(model, site)
        raise _site_error(
            f"{self.architecture_name} supports language, vision, and projector "
            f"streams, got {stream!r}"
        )

    def _resolve_vision(self, model: nn.Module, site: Any) -> ResolvedSite:
        component = getattr(site, "component", None)
        layer_index = getattr(site, "layer", None)
        if component != "vision_resid":
            raise _site_error(
                f"vision stream requires component 'vision_resid', got {component!r}"
            )
        if getattr(site, "io", "output") != "output":
            raise _site_error("vision_resid requires io='output'")
        if not isinstance(layer_index, int):
            raise _site_error("vision_resid requires an integer layer")
        if getattr(site, "unit", None) is not None:
            raise _site_error("unit-level vision_resid addressing is unsupported")
        layers, layers_path = self._vision_layers(model)
        if layer_index < 0 or layer_index >= len(layers):
            raise _site_error(
                f"vision layer {layer_index} is out of range for "
                f"{self.architecture_name}; valid range is [0, {len(layers) - 1}]"
            )
        module = layers[layer_index]
        if not isinstance(module, nn.Module):
            raise _site_error(
                f"vision layer {layer_index} is {type(module).__name__}, "
                "not torch.nn.Module"
            )
        explicit_path = tuple(getattr(site, "tensor_path", ()) or ())
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
            module_path=f"{layers_path}.{layer_index}",
            hook_kind="forward",
            tensor_accessor=accessor,
            hidden_dim=self.hidden_size(model, site),
            architecture_name=self.architecture_name,
        )

    def _resolve_projector(self, model: nn.Module, site: Any) -> ResolvedSite:
        component = getattr(site, "component", None)
        if component not in ("projector_in", "projector_out"):
            raise _site_error(
                "projector stream supports components 'projector_in' and "
                f"'projector_out', got {component!r}"
            )
        if getattr(site, "layer", None) is not None:
            raise _site_error(f"{component} is not layer-indexed")
        if getattr(site, "unit", None) is not None:
            raise _site_error(f"unit-level {component} addressing is unsupported")
        expected_io = "input" if component == "projector_in" else "output"
        if getattr(site, "io", expected_io) != expected_io:
            raise _site_error(f"{component} requires io={expected_io!r}")
        module, module_path = self._projector(model)
        explicit_path = tuple(getattr(site, "tensor_path", ()) or ())
        if component == "projector_in":
            accessor: TensorAccessor = PathTensorAccessor(explicit_path or (0,))
            hook_kind = "forward_pre"
        else:
            accessor = cast(
                TensorAccessor,
                (
                    PathTensorAccessor(explicit_path)
                    if explicit_path
                    else RootOrFirstTensorAccessor()
                ),
            )
            hook_kind = "forward"
        return ResolvedSite(
            site=site,
            module=module,
            module_path=module_path,
            hook_kind=hook_kind,  # type: ignore[arg-type]
            tensor_accessor=accessor,
            hidden_dim=self.hidden_size(model, site),
            architecture_name=self.architecture_name,
        )


__all__ = ["VisionLanguageAdapter", "resolve_attribute_path"]
