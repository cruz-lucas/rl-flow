from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import jax
import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from rlflow_builtin.environments.navix import create_navix_environment

router = APIRouter(prefix="/environment-sessions", tags=["environment-sessions"])


class EnvironmentSessionCreate(BaseModel):
    component_id: str = "navix.env.grid"
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0


class EnvironmentActionRequest(BaseModel):
    action: int


class EnvironmentSessionSnapshot(BaseModel):
    session_id: str
    component_id: str
    config: dict[str, Any]
    step: int
    reward: float
    terminated: bool
    truncated: bool
    done: bool
    action_count: int
    action_labels: list[str]
    observation_shape: list[int]
    observation_dtype: str
    observation_preview: Any
    observation_truncated: bool
    svg: str


@dataclass
class EnvironmentSession:
    component_id: str
    config: dict[str, Any]
    seed: int
    env: Any
    key: jax.Array
    timestep: Any
    step: int = 0


@router.post("", response_model=EnvironmentSessionSnapshot)
def create_session(payload: EnvironmentSessionCreate, request: Request) -> EnvironmentSessionSnapshot:
    session = _build_session(payload, request)
    request.app.state.environment_sessions[session_id := uuid4().hex] = session
    return _snapshot(session_id, session)


@router.post("/{session_id}/actions", response_model=EnvironmentSessionSnapshot)
def step_session(session_id: str, payload: EnvironmentActionRequest, request: Request) -> EnvironmentSessionSnapshot:
    session = _get_session(session_id, request)
    action_count = int(session.env.action_space.n)
    if payload.action < 0 or payload.action >= action_count:
        raise HTTPException(status_code=422, detail=f"Action must be between 0 and {action_count - 1}")
    session.timestep = session.env.step(session.timestep, np.asarray(payload.action, dtype=np.int32))
    session.step += 1
    return _snapshot(session_id, session)


@router.post("/{session_id}/reset", response_model=EnvironmentSessionSnapshot)
def reset_session(session_id: str, request: Request) -> EnvironmentSessionSnapshot:
    session = _get_session(session_id, request)
    session.key, reset_key = jax.random.split(session.key)
    session.timestep = session.env.reset(reset_key)
    session.step = 0
    return _snapshot(session_id, session)


@router.get("/{session_id}/export.pdf")
def export_pdf(session_id: str, request: Request) -> Response:
    session = _get_session(session_id, request)
    observation = np.asarray(jax.device_get(session.timestep.observation))
    symbolic = _visible_symbolic_grid(session, observation)
    pdf = _render_pdf(symbolic)
    filename = f"{session.component_id.replace('.', '_')}_step_{session.step}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def ensure_environment_session_store(app: Any) -> None:
    if not hasattr(app.state, "environment_sessions"):
        app.state.environment_sessions = {}


def _build_session(payload: EnvironmentSessionCreate, request: Request) -> EnvironmentSession:
    if payload.component_id != "navix.env.grid":
        raise HTTPException(status_code=422, detail="Environment playground currently supports navix.env.grid")

    component = request.app.state.registry.get(payload.component_id)
    config = {**component.defaults, **payload.config}
    try:
        env = create_navix_environment(**config)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    key = jax.random.PRNGKey(payload.seed)
    key, reset_key = jax.random.split(key)
    timestep = env.reset(reset_key)
    return EnvironmentSession(
        component_id=payload.component_id,
        config=config,
        seed=payload.seed,
        env=env,
        key=key,
        timestep=timestep,
    )


def _get_session(session_id: str, request: Request) -> EnvironmentSession:
    session = request.app.state.environment_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown environment session: {session_id}")
    return session


def _snapshot(session_id: str, session: EnvironmentSession) -> EnvironmentSessionSnapshot:
    timestep = session.timestep
    observation = np.asarray(jax.device_get(timestep.observation))
    observation_preview, observation_truncated = _observation_preview(observation)
    terminated = bool(np.asarray(jax.device_get(timestep.is_termination())))
    truncated = bool(np.asarray(jax.device_get(timestep.is_truncation())))
    return EnvironmentSessionSnapshot(
        session_id=session_id,
        component_id=session.component_id,
        config=session.config,
        step=session.step,
        reward=float(np.asarray(jax.device_get(timestep.reward))),
        terminated=terminated,
        truncated=truncated,
        done=terminated or truncated,
        action_count=int(session.env.action_space.n),
        action_labels=_action_labels(session.config, int(session.env.action_space.n)),
        observation_shape=list(observation.shape),
        observation_dtype=str(observation.dtype),
        observation_preview=observation_preview,
        observation_truncated=observation_truncated,
        svg=_render_svg(_visible_symbolic_grid(session, observation)),
    )


def _action_labels(config: dict[str, Any], action_count: int) -> list[str]:
    if config.get("action_set") == "cardinal":
        return ["Up", "Down", "Left", "Right"]
    labels = ["Turn left", "Turn right", "Forward", "Pick up", "Drop", "Toggle", "Done"]
    return labels[:action_count]


def _observation_preview(observation: np.ndarray, limit: int = 256) -> tuple[Any, bool]:
    if observation.shape == ():
        return observation.item(), False
    if observation.size <= limit:
        return observation.tolist(), False
    flat = observation.reshape(-1)
    return flat[:limit].tolist(), True


def _symbolic_grid(state: Any) -> np.ndarray:
    grid = np.asarray(jax.device_get(state.grid))
    height, width = grid.shape
    symbolic = np.zeros((height, width, 3), dtype=np.uint8)
    symbolic[:, :, 0] = np.where(grid == -1, 2, 1)
    symbolic[:, :, 1] = np.where(grid == -1, 5, 0)

    for entity in state.entities.values():
        positions = np.asarray(jax.device_get(entity.position))
        if positions.ndim == 1:
            positions = positions[None, :]
        tags = np.asarray(jax.device_get(entity.tag)).reshape(-1)
        states = np.asarray(jax.device_get(entity.symbolic_state)).reshape(-1)
        colour = getattr(entity, "colour", np.zeros(tags.shape, dtype=np.uint8))
        colours = np.asarray(jax.device_get(colour)).reshape(-1)
        for idx, position in enumerate(positions):
            row, col = int(position[0]), int(position[1])
            if row < 0 or col < 0 or row >= height or col >= width:
                continue
            symbolic[row, col] = [
                int(tags[min(idx, len(tags) - 1)]),
                int(colours[min(idx, len(colours) - 1)]),
                int(states[min(idx, len(states) - 1)]),
            ]
    return symbolic


def _visible_symbolic_grid(session: EnvironmentSession, observation: np.ndarray) -> np.ndarray:
    if (
        session.config.get("observation_mode") == "symbolic"
        and observation.ndim == 3
        and observation.shape[-1] == 3
    ):
        return np.rint(observation).astype(np.uint8)
    return _symbolic_grid(session.timestep.state)


def _render_svg(symbolic: np.ndarray, cell_size: int = 44) -> str:
    height, width, _ = symbolic.shape
    canvas_width = width * cell_size
    canvas_height = height * cell_size
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_width} {canvas_height}" role="img">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for row in range(height):
        for col in range(width):
            x = col * cell_size
            y = row * cell_size
            parts.extend(_svg_cell(symbolic[row, col], x, y, cell_size))
    for row in range(height + 1):
        y = row * cell_size
        parts.append(f'<line x1="0" y1="{y}" x2="{canvas_width}" y2="{y}" stroke="#9aa8b5" stroke-width="1"/>')
    for col in range(width + 1):
        x = col * cell_size
        parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{canvas_height}" stroke="#9aa8b5" stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def _svg_cell(cell: np.ndarray, x: int, y: int, size: int) -> list[str]:
    entity = int(cell[0])
    colour = int(cell[1])
    state = int(cell[2])
    pad = size * 0.16
    center_x = x + size / 2
    center_y = y + size / 2
    parts = [f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{_cell_fill(entity, colour)}"/>']
    if entity == 4:
        fill = "#f4c76b" if state == 0 else "#b87928"
        parts.append(
            f'<rect x="{x + pad}" y="{y + pad}" width="{size - 2 * pad}" height="{size - 2 * pad}" '
            f'rx="2" fill="{fill}" stroke="#704214" stroke-width="2"/>'
        )
        if state != 0:
            parts.append(f'<circle cx="{center_x + size * 0.18}" cy="{center_y}" r="{size * 0.05}" fill="#fff7dc"/>')
    elif entity == 5:
        parts.append(f'<circle cx="{center_x - size * 0.08}" cy="{center_y}" r="{size * 0.13}" fill="#e2a72e"/>')
        parts.append(
            f'<path d="M {center_x + size * 0.03} {center_y} H {x + size * 0.78} '
            f'M {x + size * 0.65} {center_y} V {y + size * 0.62}" '
            'stroke="#8f5f0d" stroke-width="3" stroke-linecap="round"/>'
        )
    elif entity == 8:
        parts.append(f'<circle cx="{center_x}" cy="{center_y}" r="{size * 0.24}" fill="#44a366"/>')
        parts.append(f'<circle cx="{center_x}" cy="{center_y}" r="{size * 0.11}" fill="#eaf7ee"/>')
    elif entity == 10:
        parts.append(f'<polygon points="{_player_points(center_x, center_y, size * 0.32, state)}" fill="#1f5f6f"/>')
    return parts


def _cell_fill(entity: int, colour: int = 0) -> str:
    if entity == 2:
        if colour != 5:
            return _wall_colour_fill(colour)
        return "#2f3a45"
    return "#fbfcfd"


def _wall_colour_fill(colour: int) -> str:
    hue = (int(colour) * 137) % 360
    saturation = 48 + int(colour) % 32
    lightness = 30 + (int(colour) // 8) % 26
    return _hsl_to_hex(hue / 360.0, saturation / 100.0, lightness / 100.0)


def _hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    def channel(offset: float) -> int:
        k = (offset + hue * 12.0) % 12.0
        a = saturation * min(lightness, 1.0 - lightness)
        value = lightness - a * max(-1.0, min(k - 3.0, 9.0 - k, 1.0))
        return int(round(255.0 * value))

    return f"#{channel(0):02x}{channel(8):02x}{channel(4):02x}"


def _player_points(cx: float, cy: float, radius: float, direction: int) -> str:
    if direction == 0:
        points = [(cx + radius, cy), (cx - radius * 0.7, cy - radius * 0.75), (cx - radius * 0.7, cy + radius * 0.75)]
    elif direction == 1:
        points = [(cx, cy + radius), (cx - radius * 0.75, cy - radius * 0.7), (cx + radius * 0.75, cy - radius * 0.7)]
    elif direction == 2:
        points = [(cx - radius, cy), (cx + radius * 0.7, cy - radius * 0.75), (cx + radius * 0.7, cy + radius * 0.75)]
    else:
        points = [(cx, cy - radius), (cx - radius * 0.75, cy + radius * 0.7), (cx + radius * 0.75, cy + radius * 0.7)]
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _render_pdf(symbolic: np.ndarray) -> bytes:
    height, width, _ = symbolic.shape
    cell = 32.0
    margin = 18.0
    page_width = width * cell + 2 * margin
    page_height = height * cell + 2 * margin
    commands = ["q", "1 1 1 rg", f"0 0 {page_width:.2f} {page_height:.2f} re f"]
    for row in range(height):
        for col in range(width):
            x = margin + col * cell
            y = margin + (height - 1 - row) * cell
            commands.extend(_pdf_cell(symbolic[row, col], x, y, cell))
    commands.extend(["0.60 0.66 0.71 RG", "0.45 w"])
    for row in range(height + 1):
        y = margin + row * cell
        commands.append(f"{margin:.2f} {y:.2f} m {margin + width * cell:.2f} {y:.2f} l S")
    for col in range(width + 1):
        x = margin + col * cell
        commands.append(f"{x:.2f} {margin:.2f} m {x:.2f} {margin + height * cell:.2f} l S")
    commands.append("Q")
    return _pdf_document("\n".join(commands), page_width, page_height)


def _pdf_cell(cell: np.ndarray, x: float, y: float, size: float) -> list[str]:
    entity = int(cell[0])
    colour = int(cell[1])
    state = int(cell[2])
    commands = [_pdf_fill(_cell_fill(entity, colour)), f"{x:.2f} {y:.2f} {size:.2f} {size:.2f} re f"]
    cx = x + size / 2
    cy = y + size / 2
    if entity == 4:
        fill = "#f4c76b" if state == 0 else "#b87928"
        pad = size * 0.16
        commands.extend([_pdf_fill(fill), f"{x + pad:.2f} {y + pad:.2f} {size - 2 * pad:.2f} {size - 2 * pad:.2f} re f"])
    elif entity == 5:
        commands.extend([_pdf_fill("#e2a72e"), _pdf_diamond(cx - size * 0.08, cy, size * 0.14)])
        commands.extend([_pdf_stroke("#8f5f0d", 2.1), f"{cx + size * 0.04:.2f} {cy:.2f} m {x + size * 0.78:.2f} {cy:.2f} l S"])
    elif entity == 8:
        commands.extend([_pdf_fill("#44a366"), _pdf_diamond(cx, cy, size * 0.25)])
        commands.extend([_pdf_fill("#eaf7ee"), _pdf_diamond(cx, cy, size * 0.11)])
    elif entity == 10:
        commands.extend([_pdf_fill("#1f5f6f"), _pdf_polygon(_player_pdf_points(cx, cy, size * 0.32, state))])
    return commands


def _pdf_fill(hex_color: str) -> str:
    r, g, b = _hex_rgb(hex_color)
    return f"{r:.4f} {g:.4f} {b:.4f} rg"


def _pdf_stroke(hex_color: str, width: float) -> str:
    r, g, b = _hex_rgb(hex_color)
    return f"{r:.4f} {g:.4f} {b:.4f} RG\n{width:.2f} w"


def _hex_rgb(hex_color: str) -> tuple[float, float, float]:
    value = hex_color.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) / 255 for idx in (0, 2, 4))  # type: ignore[return-value]


def _pdf_diamond(cx: float, cy: float, radius: float) -> str:
    return _pdf_polygon([(cx, cy + radius), (cx + radius, cy), (cx, cy - radius), (cx - radius, cy)])


def _pdf_polygon(points: list[tuple[float, float]]) -> str:
    first, *rest = points
    commands = [f"{first[0]:.2f} {first[1]:.2f} m"]
    commands.extend(f"{x:.2f} {y:.2f} l" for x, y in rest)
    commands.append("h f")
    return "\n".join(commands)


def _player_pdf_points(cx: float, cy: float, radius: float, direction: int) -> list[tuple[float, float]]:
    if direction == 0:
        return [(cx + radius, cy), (cx - radius * 0.7, cy - radius * 0.75), (cx - radius * 0.7, cy + radius * 0.75)]
    if direction == 1:
        return [(cx, cy - radius), (cx - radius * 0.75, cy + radius * 0.7), (cx + radius * 0.75, cy + radius * 0.7)]
    if direction == 2:
        return [(cx - radius, cy), (cx + radius * 0.7, cy - radius * 0.75), (cx + radius * 0.7, cy + radius * 0.75)]
    return [(cx, cy + radius), (cx - radius * 0.75, cy - radius * 0.7), (cx + radius * 0.75, cy - radius * 0.7)]


def _pdf_document(content: str, width: float, height: float) -> bytes:
    stream = content.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.2f} {height:.2f}] /Contents 4 0 R >>".encode("ascii"),
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)
