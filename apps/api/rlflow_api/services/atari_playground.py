"""Interactive single-env Atari backend for the Environment playground.

Separate from the training path (which uses envpool, vectorised + Linux-only):
the playground needs one renderable env, so it uses ``gymnasium`` + ``ale-py``
(cross-platform). Frames are rendered to a base64 PNG embedded in an ``<svg>``
so the existing playground UI (which drops ``snapshot.svg`` into the DOM) works
unchanged. The PNG/PDF encoders are dependency-free (stdlib ``zlib``) to keep
Pillow/matplotlib out of the API import path.

``gymnasium``/``ale-py`` are imported lazily so the API stays importable without
the ``atari`` extra installed.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import numpy as np


def _import_gymnasium():
    import ale_py
    import gymnasium as gym

    # ale-py registers the ``ALE/*`` ids; explicit call is a no-op if already done.
    gym.register_envs(ale_py)
    return gym


def _gym_task_id(task_id: str) -> str:
    return task_id if task_id.startswith("ALE/") else f"ALE/{task_id}"


def build_atari_env(config: dict[str, Any]) -> Any:
    gym = _import_gymnasium()
    return gym.make(
        _gym_task_id(str(config.get("task_id", "MontezumaRevenge-v5"))),
        render_mode="rgb_array",
        repeat_action_probability=float(config.get("repeat_action_probability", 0.25)),
        full_action_space=bool(config.get("full_action_space", False)),
    )


def _current_frame(env: Any, observation: Any) -> np.ndarray:
    frame = env.render()
    if frame is None:
        frame = observation
    return np.asarray(frame, dtype=np.uint8)


def reset_atari(env: Any, seed: int) -> dict[str, Any]:
    observation, _info = env.reset(seed=int(seed))
    return {
        "observation": np.asarray(observation),
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "frame": _current_frame(env, observation),
    }


def step_atari(env: Any, action: int) -> dict[str, Any]:
    observation, reward, terminated, truncated, _info = env.step(int(action))
    return {
        "observation": np.asarray(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "frame": _current_frame(env, observation),
    }


def atari_action_labels(env: Any) -> list[str]:
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        return [str(index) for index in range(int(env.action_space.n))]
    return [str(name).replace("_", " ").title() for name in meanings]


# --- dependency-free frame rendering ---------------------------------------


def _png_bytes(rgb: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(rgb[:, :, :3].astype(np.uint8))
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))
    compressed = zlib.compress(raw, 9)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", compressed)
        + _chunk(b"IEND", b"")
    )


def frame_svg(frame: np.ndarray) -> str:
    height, width, _ = frame.shape
    encoded = base64.b64encode(_png_bytes(frame)).decode("ascii")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'preserveAspectRatio="xMidYMid meet" role="img">'
        f'<image href="data:image/png;base64,{encoded}" x="0" y="0" '
        f'width="{width}" height="{height}" style="image-rendering:pixelated"/>'
        "</svg>"
    )


def frame_pdf(frame: np.ndarray) -> bytes:
    rgb = np.ascontiguousarray(frame[:, :, :3].astype(np.uint8))
    height, width, _ = rgb.shape
    image_data = zlib.compress(rgb.tobytes(), 9)
    content = f"q\n{width} 0 0 {height} 0 0 cm\n/Im0 Do\nQ".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            "/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
        ).encode("ascii"),
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        (
            b"<< /Type /XObject /Subtype /Image /Width "
            + str(width).encode()
            + b" /Height "
            + str(height).encode()
            + b" /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length "
            + str(len(image_data)).encode()
            + b" >>\nstream\n"
            + image_data
            + b"\nendstream"
        ),
    ]
    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n".encode("ascii")
    )
    return bytes(pdf)
