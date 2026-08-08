from __future__ import annotations

import matplotlib.pyplot as plt

from fr_gvi.plotting.style import COLORS, display_label, figure_text, normalize_figure_text


def test_figure_text_uses_single_dashes_without_changing_method_keys() -> None:
    assert figure_text("FR--R--STL") == "FR-R-STL"
    assert display_label({"method": "FR--R--STL", "variant": "fr-r-stl-qr"}) == "FR-R-STL+QR"
    assert "FR--R" in COLORS


def test_bures_wasserstein_baseline_is_displayed_geometry_first() -> None:
    """The stored identifier stays put; only what a reader sees is renamed.

    ``FR--R`` and ``FR--KL`` name their geometry, ``FB--GVI`` names only its
    discretization, so the baseline is shown as ``BW-FB`` to make the three labels
    symmetric.  It is deliberately not plain ``BW``, which would name a geometry
    whose other algorithms this study excludes.
    """

    assert figure_text("FB--GVI") == "BW-FB"
    assert figure_text("S--FB--GVI") == "S-BW-FB"
    assert figure_text(figure_text("FB--GVI")) == "BW-FB"
    assert "FB--GVI" in COLORS


def test_star_subscripts_are_braced_for_mathtext() -> None:
    """TeX classes ``\\star`` as a binary operator, which opens a gap in subscripts."""

    assert figure_text(r"$\kappa_\star$") == r"$\kappa_{\star}$"
    assert figure_text(r"$\lambda_{0,\star}$") == r"$\lambda_{0,{\star}}$"
    assert figure_text(figure_text(r"$\kappa_\star$")) == r"$\kappa_{\star}$"


def test_dimension_is_spelled_as_the_manuscript_spells_it() -> None:
    """A panel labelled ``d`` beside text saying ``N_\\theta`` makes the reader translate."""

    assert figure_text(r"$d=10$") == r"$N_\theta=10$"
    assert figure_text("dimension $d$") == r"dimension $N_\theta$"
    assert figure_text(r"$d\in\{10,50\}$") == r"$N_\theta\in\{10,50\}$"
    assert figure_text("$d^2$") == r"$N_\theta^2$"
    assert figure_text(r"$O(nQ + nd^2) = O(d^3)$") == r"$O(nQ + nN_\theta^2) = O(N_\theta^3)$"
    assert figure_text("$n=10d$") == r"$n=10N_\theta$"
    # Inside math a parenthesised d is the dimension, unlike the panel letter.
    assert figure_text("$O(d)$") == r"$O(N_\theta)$"


def test_dimension_rewrite_leaves_identifiers_and_words_alone() -> None:
    """Job identifiers name directories on disk and must keep their spelling."""

    for untouched in (
        "F2global_d10_k1_rho1",
        "F4real_ionosphere_fb-gvi",
        "wdbc",
        "dimension",
        "and the seed used",
        "held-out predictive",
        "standardized",
        r"$\Delta t$",
        # The fourth panel letter, in a title and in its caption.  Rewriting it
        # renumbers every four-panel figure in the paper.
        "(d) Across datasets",
        "**(d)** The quadratic rescue, charged in joint oracle pairs.",
        # An abbreviation, not an identifier -- and it is an axis label.  The
        # stars are already braced so this fixture isolates the dimension rule.
        r"across-seed s.d. of $\|a_n - a_{\star}\|_{\star}$",
        "the certified step, and d is not the dimension here",
    ):
        assert figure_text(untouched) == untouched


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
