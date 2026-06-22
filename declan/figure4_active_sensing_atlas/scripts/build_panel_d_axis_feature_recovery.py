"""Build the Figure 4D axis-conditioned feature-recovery panel."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

try:  # pragma: no cover - script-mode import path fallback
    from . import build_selected_figure4_v4_design as v4
except ImportError:  # pragma: no cover
    import build_selected_figure4_v4_design as v4


PANEL_D = v4.FIGURES / "panel_D"


def build() -> list[Path]:
    v4._configure_matplotlib()
    PANEL_D.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(4.2, 3.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 1, left=0.18, right=0.96, top=0.76, bottom=0.20)
    values = v4._plot_d(gs[0, 0])

    png = PANEL_D / "D2_axis_feature_recovery.png"
    pdf = PANEL_D / "D2_axis_feature_recovery.pdf"
    csv = PANEL_D / "panel_D_axis_feature_recovery_values.csv"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    values.to_csv(csv, index=False)
    return [png, pdf, csv]


def main() -> None:
    for path in build():
        print(path)


if __name__ == "__main__":
    main()
