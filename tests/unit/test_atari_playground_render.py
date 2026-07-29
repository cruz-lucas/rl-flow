"""Dependency-free frame-rendering tests for the Atari playground service.

These do not need ale-py/gymnasium — they exercise the stdlib PNG/SVG/PDF
encoders directly, so they run in CI without the atari extra.
"""

from __future__ import annotations

import numpy as np

from rlflow_api.services.atari_playground import frame_pdf, frame_svg


def _frame() -> np.ndarray:
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # solid red
    return frame


def test_frame_svg_embeds_a_png_data_uri():
    svg = frame_svg(_frame())
    assert svg.startswith("<svg")
    assert 'viewBox="0 0 10 8"' in svg
    assert "data:image/png;base64," in svg
    assert svg.strip().endswith("</svg>")


def test_frame_pdf_is_a_valid_pdf_with_an_image():
    pdf = frame_pdf(_frame())
    assert pdf.startswith(b"%PDF-1.4")
    assert b"/Subtype /Image" in pdf
    assert pdf.rstrip().endswith(b"%%EOF")
