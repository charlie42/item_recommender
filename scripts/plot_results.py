import argparse
import math
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

CV_COLOR = "#2a78d6"
TEST_COLOR = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


class OutputCurve:
    """AUC by number of items for one output."""

    def __init__(self, name, result):
        self.name = name.removeprefix("Diag.")
        folds = result["cv_performances"]
        self.n_items = np.array(sorted(set().union(*(f["perf_on_features"] for f in folds))))
        aucs = np.array([
            [f["perf_on_features"].get(n, np.nan) for n in self.n_items] for f in folds
        ])
        self.cv_mean = np.nanmean(aucs, axis=0)
        self.cv_min = np.nanmin(aucs, axis=0)
        self.cv_max = np.nanmax(aucs, axis=0)
        self.cv_all_items = np.mean([f["perf_all_features"] for f in folds])
        self.chosen_n = len(result["items"])
        self.test_auc = result["test_performance"]


class ResultsPlotter:
    """Plot AUC by number of items, one panel per output."""

    def __init__(self, results, ncols=4):
        self.curves = sorted(
            (OutputCurve(name, result) for name, result in results.items()),
            key=lambda curve: curve.cv_all_items,
            reverse=True,
        )
        self.ncols = min(ncols, len(self.curves))
        self.nrows = math.ceil(len(self.curves) / self.ncols)

    def plot(self, path):
        """Draw the panels and save the figure."""
        fig, axes = plt.subplots(
            self.nrows, self.ncols, sharex=True, sharey=True, squeeze=False,
            figsize=(3 * self.ncols, 2.3 * self.nrows + 0.6), layout="constrained",
            facecolor=SURFACE,
        )
        for ax, curve in zip(axes.flat, self.curves):
            self._plot_panel(ax, curve)
        for i in range(len(self.curves), self.nrows * self.ncols):
            axes.flat[i].set_axis_off()
            axes.flat[i - self.ncols].tick_params(labelbottom=True)
        axes[0, 0].set_ylim(min(0.4, min(np.nanmin(c.cv_min) for c in self.curves) - 0.02), 1.0)
        axes[0, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.supxlabel("Number of items", color=SECONDARY_INK)
        fig.supylabel("AUC", color=SECONDARY_INK)
        fig.legend(
            *axes[0, 0].get_legend_handles_labels(), loc="outside upper center",
            ncol=5 if self.ncols >= 4 else 2, frameon=False, fontsize=9, labelcolor=SECONDARY_INK,
        )
        fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")

    def _plot_panel(self, ax, curve):
        """Draw one output's panel."""
        ax.set_facecolor(SURFACE)
        ax.fill_between(
            curve.n_items, curve.cv_min, curve.cv_max, color=CV_COLOR, alpha=0.18,
            linewidth=0, label="CV range across folds",
        )
        ax.plot(curve.n_items, curve.cv_mean, color=CV_COLOR, linewidth=1.75, label="Mean CV AUC")
        ax.axhline(curve.cv_all_items, color=SECONDARY_INK, linewidth=1, linestyle="--",
                   label="Mean CV AUC, all items")
        ax.axhline(0.5, color=MUTED_INK, linewidth=0.8, linestyle=":", label="Chance")
        ax.axvline(curve.chosen_n, color=AXIS, linewidth=0.8, zorder=1)
        ax.plot(curve.chosen_n, curve.test_auc, "o", color=TEST_COLOR, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                label="Test AUC, recommended items")
        label = f"{curve.chosen_n} item{'s' * (curve.chosen_n != 1)} · test AUC {curve.test_auc:.2f}"
        ax.text(0.97, 0.05, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=SECONDARY_INK)
        ax.set_title(curve.name, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK, labelsize=8)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", type=str, required=True, help="Path to results.joblib")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output PNG")
    return parser.parse_args()


def main():
    args = parse_arguments()
    ResultsPlotter(joblib.load(args.results_path)).plot(args.output_path)
    print(f"Plot saved to {args.output_path}")


if __name__ == "__main__":
    main()
