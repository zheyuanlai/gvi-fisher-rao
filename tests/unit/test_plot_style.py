from __future__ import annotations

import matplotlib.pyplot as plt

from fr_gvi.plotting.style import COLORS, display_label, figure_text, normalize_figure_text


def test_figure_text_uses_single_dashes_without_changing_method_keys() -> None:
    assert figure_text("S--FB--GVI") == "S-FB-GVI"
    assert display_label({"method": "FR--R--STL", "variant": "fr-r-stl-qr"}) == "FR-R-STL+QR"
    assert "FR--R" in COLORS


def test_normalize_figure_text_covers_axes_annotations_and_legends() -> None:
    figure, axis = plt.subplots()
    axis.set_title("FR--R")
    axis.set_xlabel("gradient--Hessian oracle pairs")
    axis.plot([0.0, 1.0], [0.0, 1.0], label="FB--GVI")
    axis.text(0.5, 0.5, "S--FB--GVI")
    axis.legend()

    normalize_figure_text(figure)

    rendered_text = [
        artist.get_text() for artist in figure.findobj() if hasattr(artist, "get_text")
    ]
    assert rendered_text
    assert all("--" not in text for text in rendered_text)
    plt.close(figure)
