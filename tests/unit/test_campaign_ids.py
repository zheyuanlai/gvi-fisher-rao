from __future__ import annotations

import pytest

from fr_gvi.algorithms import Method
from fr_gvi.experiments.campaign import _method_slug


def test_method_tags_prevent_stepsize_sweep_collisions() -> None:
    first = _method_slug(Method.FR_R, {"step_size": 0.05, "tag": "h005"})
    second = _method_slug(Method.FR_R, {"step_size": 0.1, "tag": "h01"})
    assert first != second
    assert first.endswith("h005")


def test_method_tag_rejects_path_characters() -> None:
    with pytest.raises(ValueError, match="method tag"):
        _method_slug(Method.FR_R, {"tag": "../escape"})

