"""Plot train vs validation curves to a PNG file."""
from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter

# Validated colorblind-safe pair. Val is also dashed, so color is not the
# only cue.
SERIES_STYLE = {
    "train": {"color": "#2a78d6", "linestyle": "-"},
    "val": {"color": "#eb6834", "linestyle": "--"},
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

History = dict[str, list[float]]


def plot_history(
    history: History, output_path: Path, best_epoch: int | None = None
) -> None:
    """Save side-by-side loss and accuracy curves for train and val.

    If best_epoch is given, a vertical marker shows the checkpointed epoch.

    Uses the object-oriented Figure API instead of pyplot, so it works
    without a display (Kaggle, SSH) and keeps no global plotting state.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = Figure(figsize=(11, 4), facecolor=SURFACE)
    ax_loss, ax_acc = fig.subplots(1, 2)

    panels = ((ax_loss, "loss", "Cross-entropy loss"),
              (ax_acc, "acc", "Accuracy"))
    for ax, metric, title in panels:
        for split, style in SERIES_STYLE.items():
            values = history[f"{split}_{metric}"]
            ax.plot(epochs, values, linewidth=2, marker="o", markersize=5,
                    label=split, **style)
            # Direct label at the line end, in ink rather than series color.
            ax.annotate(split, xy=(epochs[-1], values[-1]), xytext=(8, 0),
                        textcoords="offset points", va="center",
                        color=INK, fontsize=9)
        _style_axes(ax, title)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=":")

    ax_acc.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_loss.legend(frameon=False, labelcolor=INK)
    if best_epoch is not None:
        # Caption below the plots, so it never collides with the curves.
        fig.text(0.01, 0.01, f"Dotted line: best checkpoint "
                 f"(lowest val loss, epoch {best_epoch})",
                 color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)


def _style_axes(ax, title: str) -> None:
    """Recessive axes: hairline grid, muted ticks, no top/right frame."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelcolor=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.margins(x=0.08)  # Room for the direct labels at the right edge.
    ax.set_xlim(left=0.5)  # Epochs start at 1.
