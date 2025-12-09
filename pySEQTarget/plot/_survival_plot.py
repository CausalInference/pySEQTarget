import itertools

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _survival_plot(self):
    if self.plot_type == "risk":
        plot_data = self.km_data.filter(pl.col("estimate") == "risk")
    elif self.plot_type == "survival":
        plot_data = self.km_data.filter(pl.col("estimate") == "survival")
    else:
        plot_data = self.km_data.filter(pl.col("estimate") == "incidence")

    if self.subgroup_colname is None:
        fig = _plot_single(self, plot_data)
    else:
        fig = _plot_subgroups(self, plot_data)

    return fig


def _plot_single(self, plot_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_data(self, plot_data, ax)

    if self.plot_title is None:
        self.plot_title = f"Cumulative {self.plot_type.title()}"

    ax.set_xlabel("Followup")
    ax.set_ylabel(self.plot_type.title())
    ax.set_title(self.plot_title)
    ax.legend()
    ax.grid()

    return fig


def _plot_subgroups(self, plot_data):
    subgroups = sorted(plot_data[self.subgroup_colname].unique().to_list())
    n_subgroups = len(subgroups)
    n_cols = min(3, n_subgroups)
    n_rows = (n_subgroups + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for idx, subgroup_val in enumerate(subgroups):
        ax = axes[idx]
        subgroup_data = plot_data.filter(pl.col(self.subgroup_colname) == subgroup_val)
        _plot_data(self, subgroup_data, ax)
        subgroup_label = (
            str(subgroup_val).title() if isinstance(subgroup_val, str) else subgroup_val
        )
        ax.set_xlabel("Followup")
        ax.set_ylabel(self.plot_type.title())
        ax.set_title(
            f"{self.subgroup_colname.title()}: {subgroup_label}",
            fontsize=10,
            style="italic",
        )
        ax.legend()
        ax.grid()

    for idx in range(n_subgroups, len(axes)):
        axes[idx].set_visible(False)

    if self.plot_title:
        fig.suptitle(self.plot_title, fontsize=14)
    else:
        fig.suptitle(f"Cumulative {self.plot_type.title()}", fontsize=14)

    plt.tight_layout()
    return fig


def _plot_data(self, plot_data, ax):
    color_cycle = itertools.cycle(self.plot_colors) if self.plot_colors else None

    for idx, i in enumerate(self.treatment_level):
        subset = plot_data.filter(pl.col(self.treatment_col) == i)
        if subset.is_empty():
            continue

        label = f"treatment = {i}"
        if self.plot_labels and idx < len(self.plot_labels):
            label = self.plot_labels[idx]

        color = next(color_cycle) if color_cycle else None

        (line,) = ax.plot(
            subset["followup"], subset["pred"], "-", label=label, color=color
        )

        if "LCI" in subset.columns and "UCI" in subset.columns:
            ax.fill_between(
                subset["followup"],
                subset["LCI"],
                subset["UCI"],
                color=line.get_color(),
                alpha=0.2,
                label="_nolegend_",
            )
