from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from rlflow_api.services.dataset_analysis import (
    OfflineRndAnalysis as OfflineRndAnalysisResult,
)
from rlflow_api.services.dataset_analysis import (
    VisitationAnalysis as VisitationAnalysisResult,
)
from rlflow_api.services.dataset_analysis import offline_rnd_analysis, transition_visitation

router = APIRouter(prefix="/datasets", tags=["datasets"])

TRANSITION_ARRAYS = ("observations", "actions", "rewards", "next_observations", "terminals")


class DatasetInspectRequest(BaseModel):
    path: str
    preview_rows: int = Field(default=25, ge=1, le=500)


class DatasetListItem(BaseModel):
    path: str
    size_bytes: int
    modified_time: float


class DatasetArraySummary(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    min: float | int | bool | None = None
    max: float | int | bool | None = None


class DatasetVisitation(BaseModel):
    height: int
    width: int
    action_count: int
    action_labels: list[str]
    valid_mask: list[list[bool]]
    state_counts: list[list[int]]
    state_action_counts: list[list[list[int]]]
    source: str


class DatasetInspection(BaseModel):
    path: str
    arrays: list[DatasetArraySummary]
    is_transition_dataset: bool
    num_transitions: int | None
    preview: list[dict[str, Any]]
    visitation: DatasetVisitation | None = None


class OfflineRndRequest(BaseModel):
    path: str
    algorithm: Literal["rnd", "cfn", "classifier", "simhash"] = "rnd"
    granularity: Literal["state", "state_action"] = "state"
    epochs: int = Field(default=50, ge=1, le=2000)
    batch_size: int = Field(default=128, ge=1, le=8192)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    hidden_units: list[int] = Field(default_factory=lambda: [128, 128])
    activation: Literal["relu", "tanh", "gelu", "elu", "linear"] = "relu"
    optimizer: Literal["adam", "sgd", "rmsprop"] = "adam"
    action_conditioning: Literal["none", "input", "output", "pair"] = "none"
    update_period: int = Field(default=1, ge=1, le=100000)
    output_dim: int = Field(default=64, ge=1, le=4096)
    intrinsic_reward_scale: float = Field(default=1.0, ge=0.0)
    intrinsic_stats_decay: float = Field(default=0.99, ge=0.0, le=1.0)
    intrinsic_reward_epsilon: float = Field(default=1e-4, gt=0.0)
    intrinsic_reward_clip: float | None = Field(default=10.0, gt=0.0)
    intrinsic_reward_center: bool = False
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    seed: int = Field(default=0, ge=0)
    simhash_mode: Literal["static", "learned", "autoencoder"] = "static"
    simhash_bits: int = Field(default=32, ge=1, le=4096)
    simhash_table_size: int = Field(default=16384, ge=1, le=1_000_000)
    simhash_bonus_exponent: float = Field(default=0.5, gt=0.0)
    simhash_min_count: float = Field(default=1.0, gt=0.0)


class OfflineRndPoint(BaseModel):
    count: int
    learned_bonus: float
    count_bonus: float
    row: int | None = None
    col: int | None = None
    action: int | None = None


class OfflineRndResponse(BaseModel):
    path: str
    algorithm: str
    granularity: str
    epochs: int
    batch_size: int
    unique_items: int
    loss_history: list[float]
    visitation: DatasetVisitation | None = None
    learned_state_bonus: list[list[float | None]] | None = None
    count_state_bonus: list[list[float | None]] | None = None
    learned_state_action_bonus: list[list[list[float | None]]] | None = None
    count_state_action_bonus: list[list[list[float | None]]] | None = None
    scatter: list[OfflineRndPoint]


@router.get("", response_model=list[DatasetListItem])
def list_datasets(request: Request) -> list[DatasetListItem]:
    run_root = _absolute_run_root(request)
    if not run_root.exists():
        return []
    datasets = [
        DatasetListItem(
            path=_display_path(path),
            size_bytes=path.stat().st_size,
            modified_time=path.stat().st_mtime,
        )
        for path in run_root.rglob("*.npz")
        if path.is_file()
    ]
    datasets.sort(key=lambda item: item.modified_time, reverse=True)
    return datasets


@router.post("/inspect", response_model=DatasetInspection)
def inspect_dataset(payload: DatasetInspectRequest, request: Request) -> DatasetInspection:
    path = _resolve_dataset_path(payload.path, request)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset does not exist: {path}")
    if path.suffix != ".npz":
        raise HTTPException(status_code=422, detail="Only compressed .npz datasets are currently supported")

    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not load dataset: {exc}") from exc

    arrays = [_array_summary(name, np.asarray(data[name])) for name in data.files]
    is_transition_dataset = all(name in data.files for name in TRANSITION_ARRAYS)
    num_transitions = None
    preview: list[dict[str, Any]] = []
    if is_transition_dataset:
        lengths = {len(np.asarray(data[name])) for name in TRANSITION_ARRAYS}
        if len(lengths) != 1:
            raise HTTPException(status_code=422, detail="Transition dataset arrays have inconsistent lengths")
        num_transitions = lengths.pop()
        limit = min(payload.preview_rows, num_transitions)
        for idx in range(limit):
            preview.append(
                {
                    "index": idx,
                    "observation": _json_value(np.asarray(data["observations"])[idx]),
                    "action": _json_value(np.asarray(data["actions"])[idx]),
                    "reward": _json_value(np.asarray(data["rewards"])[idx]),
                    "next_observation": _json_value(np.asarray(data["next_observations"])[idx]),
                    "terminal": _json_value(np.asarray(data["terminals"])[idx]),
                }
            )
        visitation = _visitation_model(
            transition_visitation(
                np.asarray(data["observations"]),
                np.asarray(data["actions"]),
            )
        )
    else:
        visitation = None

    return DatasetInspection(
        path=str(path),
        arrays=arrays,
        is_transition_dataset=is_transition_dataset,
        num_transitions=num_transitions,
        preview=preview,
        visitation=visitation,
    )


@router.post("/offline-rnd", response_model=OfflineRndResponse)
def train_offline_rnd(payload: OfflineRndRequest, request: Request) -> OfflineRndResponse:
    path = _resolve_dataset_path(payload.path, request)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset does not exist: {path}")
    if path.suffix != ".npz":
        raise HTTPException(status_code=422, detail="Only compressed .npz datasets are currently supported")
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not load dataset: {exc}") from exc
    missing = sorted(set(TRANSITION_ARRAYS) - set(data.files))
    if missing:
        raise HTTPException(status_code=422, detail=f"Replay dataset is missing arrays: {missing}")

    hidden_units = tuple(int(unit) for unit in payload.hidden_units if int(unit) > 0)
    try:
        analysis = offline_rnd_analysis(
            np.asarray(data["observations"]),
            np.asarray(data["actions"]),
            algorithm=payload.algorithm,
            granularity=payload.granularity,
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            learning_rate=payload.learning_rate,
            hidden_units=hidden_units,
            activation=payload.activation,
            optimizer=payload.optimizer,
            action_conditioning=payload.action_conditioning,
            update_period=payload.update_period,
            output_dim=payload.output_dim,
            intrinsic_reward_scale=payload.intrinsic_reward_scale,
            intrinsic_stats_decay=payload.intrinsic_stats_decay,
            intrinsic_reward_epsilon=payload.intrinsic_reward_epsilon,
            intrinsic_reward_clip=payload.intrinsic_reward_clip,
            intrinsic_reward_center=payload.intrinsic_reward_center,
            max_grad_norm=payload.max_grad_norm,
            seed=payload.seed,
            simhash_mode=payload.simhash_mode,
            simhash_bits=payload.simhash_bits,
            simhash_table_size=payload.simhash_table_size,
            simhash_bonus_exponent=payload.simhash_bonus_exponent,
            simhash_min_count=payload.simhash_min_count,
            cfn_targets=np.asarray(data["cfn_targets"]) if "cfn_targets" in data.files else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return _offline_rnd_model(path, analysis)


def _resolve_dataset_path(path: str, request: Request) -> Path:
    raw_path = Path(path).expanduser()
    candidates: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        resolved = candidate.resolve()
        if resolved not in candidates:
            candidates.append(resolved)
        if candidate.suffix == "":
            suffixed = candidate.with_suffix(".npz").resolve()
            if suffixed not in candidates:
                candidates.append(suffixed)

    if raw_path.is_absolute():
        add_candidate(raw_path)
    else:
        cwd = Path.cwd()
        run_root = _absolute_run_root(request)
        add_candidate(cwd / raw_path)
        if raw_path.parts and raw_path.parts[0] == run_root.name:
            add_candidate(run_root.parent / raw_path)
        else:
            add_candidate(run_root / raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else raw_path.resolve()


def _absolute_run_root(request: Request) -> Path:
    run_root = Path(request.app.state.settings.run_root).expanduser()
    if run_root.is_absolute():
        return run_root.resolve()
    return (Path.cwd() / run_root).resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())


def _array_summary(name: str, array: np.ndarray) -> DatasetArraySummary:
    flattened = array.reshape(-1)
    min_value = _json_value(np.min(flattened)) if flattened.size else None
    max_value = _json_value(np.max(flattened)) if flattened.size else None
    return DatasetArraySummary(
        name=name,
        shape=list(array.shape),
        dtype=str(array.dtype),
        min=min_value,
        max=max_value,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _json_value(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _visitation_model(analysis: VisitationAnalysisResult | None) -> DatasetVisitation | None:
    if analysis is None:
        return None
    return DatasetVisitation(
        height=analysis.height,
        width=analysis.width,
        action_count=analysis.action_count,
        action_labels=list(analysis.action_labels),
        valid_mask=analysis.valid_mask,
        state_counts=analysis.state_counts,
        state_action_counts=analysis.state_action_counts,
        source=analysis.source,
    )


def _offline_rnd_model(
    path: Path,
    analysis: OfflineRndAnalysisResult,
) -> OfflineRndResponse:
    return OfflineRndResponse(
        path=str(path),
        algorithm=analysis.algorithm,
        granularity=analysis.granularity,
        epochs=analysis.epochs,
        batch_size=analysis.batch_size,
        unique_items=analysis.unique_items,
        loss_history=analysis.loss_history,
        visitation=_visitation_model(analysis.visitation),
        learned_state_bonus=analysis.learned_state_bonus,
        count_state_bonus=analysis.count_state_bonus,
        learned_state_action_bonus=analysis.learned_state_action_bonus,
        count_state_action_bonus=analysis.count_state_action_bonus,
        scatter=[OfflineRndPoint(**point) for point in analysis.scatter],
    )
