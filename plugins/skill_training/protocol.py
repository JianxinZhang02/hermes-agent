"""Public protocol implemented by external skill-training dataset adapters."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Protocol, runtime_checkable

ADAPTER_ENTRY_POINT_GROUP = "hermes.skill_training_adapters"
SUPPORTED_MODES = frozenset({"reference", "binary_enriched"})


class AdapterError(RuntimeError):
    """Raised when an external training adapter cannot be loaded or used."""


@dataclass(frozen=True)
class PublicInputFile:
    source: Path
    relative_path: str


@dataclass(frozen=True)
class TrainingItem:
    id: str
    question: str
    input_files: tuple[PublicInputFile, ...] = ()


@dataclass(frozen=True)
class ReferenceFeedback:
    text: str
    source: str


@dataclass(frozen=True)
class BinaryOutcome:
    accepted: bool
    evaluator: str


@runtime_checkable
class TrainingAdapter(Protocol):
    name: str
    supported_modes: frozenset[str]

    def iter_items(
        self,
        *,
        dataset: Path,
        split: str,
        limit: int,
        options: Mapping[str, str],
    ) -> Iterable[TrainingItem]: ...

    def evaluate(
        self,
        item_id: str,
        blind_answer: str,
    ) -> ReferenceFeedback | BinaryOutcome: ...


def _entry_points():
    points = importlib.metadata.entry_points()
    if hasattr(points, "select"):
        return list(points.select(group=ADAPTER_ENTRY_POINT_GROUP))
    if isinstance(points, dict):
        return list(points.get(ADAPTER_ENTRY_POINT_GROUP, []))
    return [point for point in points if point.group == ADAPTER_ENTRY_POINT_GROUP]


def load_adapter(name: str) -> TrainingAdapter:
    matches = [point for point in _entry_points() if point.name == name]
    if not matches:
        available = ", ".join(sorted({point.name for point in _entry_points()})) or "none"
        raise AdapterError(
            f"skill-training adapter {name!r} is not installed; available: {available}"
        )
    if len(matches) > 1:
        raise AdapterError(f"multiple skill-training adapters are registered as {name!r}")
    try:
        factory = matches[0].load()
        adapter = factory()
    except Exception as exc:
        raise AdapterError(f"failed to load adapter {name!r}: {exc}") from exc
    if not isinstance(adapter, TrainingAdapter):
        raise AdapterError(
            f"adapter {name!r} does not implement the TrainingAdapter protocol"
        )
    modes = frozenset(str(mode) for mode in adapter.supported_modes)
    if not modes or not modes <= SUPPORTED_MODES:
        raise AdapterError(
            f"adapter {name!r} declares unsupported modes: {sorted(modes - SUPPORTED_MODES)}"
        )
    return adapter


