"""Concise constructors for semantic model sites."""

from __future__ import annotations

from typing import Any

from repsteer.core.site import Site, SiteIO, SiteStream, SiteUnit


def _site(
    component: str,
    layer: int | None = None,
    *,
    stream: SiteStream | str = "language",
    unit: SiteUnit = None,
    io: SiteIO = "output",
    tensor_path: tuple[int | str, ...] = (),
) -> Site:
    return Site(
        stream=stream,
        component=component,
        layer=layer,
        unit=unit,
        io=io,
        tensor_path=tensor_path,
    )


def resid_pre(layer: int, **kwargs: Any) -> Site:
    return _site("resid_pre", layer, **kwargs)


def attn_out(layer: int, **kwargs: Any) -> Site:
    return _site("attn_out", layer, **kwargs)


def head_result(layer: int, **kwargs: Any) -> Site:
    """The flattened query-head result immediately before ``self_attn.o_proj``."""

    kwargs.setdefault("io", "input")
    return _site("head_result", layer, **kwargs)


def resid_mid(layer: int, **kwargs: Any) -> Site:
    return _site("resid_mid", layer, **kwargs)


def mlp_out(layer: int, **kwargs: Any) -> Site:
    return _site("mlp_out", layer, **kwargs)


def resid_post(layer: int, **kwargs: Any) -> Site:
    return _site("resid_post", layer, **kwargs)


def head_out(layer: int, *, unit: SiteUnit = None, **kwargs: Any) -> Site:
    return _site("head_out", layer, unit=unit, **kwargs)


def vision_resid(layer: int, **kwargs: Any) -> Site:
    kwargs.setdefault("stream", "vision")
    return _site("vision_resid", layer, **kwargs)


def projector_in(**kwargs: Any) -> Site:
    kwargs.setdefault("stream", "projector")
    kwargs.setdefault("io", "input")
    return _site("projector_in", layer=None, **kwargs)


def projector_out(**kwargs: Any) -> Site:
    kwargs.setdefault("stream", "projector")
    return _site("projector_out", layer=None, **kwargs)


def fusion_out(**kwargs: Any) -> Site:
    kwargs.setdefault("stream", "fusion")
    return _site("fusion_out", layer=None, **kwargs)


def logits(**kwargs: Any) -> Site:
    return _site("logits", layer=None, **kwargs)


__all__ = [
    "Site",
    "attn_out",
    "fusion_out",
    "head_result",
    "head_out",
    "logits",
    "mlp_out",
    "projector_in",
    "projector_out",
    "resid_mid",
    "resid_post",
    "resid_pre",
    "vision_resid",
]
