import logging
import warnings
from collections.abc import Iterable

import matplotlib.pyplot as plt


PLT_STYLE = "seaborn-v0_8-poster"  # e.g., "seaborn-v0_8-paper", "xkcd"


def plot_image_series(x, title="", sample_indices=[0], time_indices=[0, 2, 4, 6], ylabel_rotation=90, **subplot_kwargs):
    """Plot a series of images with font sizes based on current rcParams.

    Parameters
    ----------
    x : array-like
        4D array-like (samples, time, H, W, C).
    title : str, optional
        Overall figure title.
    sample_indices : list, optional
        Sample indices to show.
    time_indices : list, optional
        Time indices to show.
    ylabel_rotation : int, optional
        Rotation angle for row labels.
    subplot_kwargs
        Additional arguments passed to plt.subplots.
    """
    # Suppress warnings and matplotlib logging.
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    style = plt.style.context(PLT_STYLE)

    with style:
        # Pull font sizes from the current style's rcParams.
        title_fontsize = plt.rcParams.get("figure.titlesize", 18)
        label_fontsize = plt.rcParams.get("axes.labelsize", 14)
        axes_title_fontsize = plt.rcParams.get("axes.titlesize", 16)

        # Create subplots.
        fig, axes = plt.subplots(len(sample_indices), len(time_indices), sharex=True, sharey=True, constrained_layout=True, **subplot_kwargs)

        # Ensure axes is always 2D.
        if len(sample_indices) == 1:
            axes = [axes]
        if len(time_indices) == 1:
            axes = [[ax] for ax in axes]

        for s_idx, sample in enumerate(sample_indices):
            for t_idx, timestep in enumerate(time_indices):
                ax = axes[s_idx][t_idx]
                ax.imshow(x[sample][timestep])
                ax.axis("off")

                # Column labels (top row only).
                if s_idx == 0:
                    ax.set_title(f"Step {timestep}", fontsize=label_fontsize)

                # Row labels (first column only).
                if t_idx == 0:
                    ax.text(
                        -0.05,
                        0.5,
                        f"Sample {sample}",
                        va="center",
                        ha="right",
                        fontsize=label_fontsize,
                        transform=ax.transAxes,
                        rotation=ylabel_rotation,
                    )

        # Optional figure title.
        if title:
            fig.suptitle(title, fontsize=title_fontsize, y=1.02)

        plt.show()


def plot_training_history(history, keys=[("loss", "val_loss"), ("accuracy", "val_accuracy")], **subplot_kwargs):
    """Plot a Keras-generated training history."""
    metrics = history
    epochs = range(1, len(metrics[keys[0][0]]) + 1)

    fig, axes = plt.subplots(1, len(keys), **subplot_kwargs)
    if not isinstance(axes, Iterable):
        axes = [axes]

    # Plot metrics.
    for sub_keys, ax in zip(keys, axes):
        for key in sub_keys:
            ax.plot(epochs, metrics[key], label=key)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_rdm(rdm, category_labels=None, distance_metric=r"Pearson Correlation Distance (1 - $\rho$)", **imshow_kwargs):
    """Display a square representational dissimilarity matrix (RDM)."""
    category_labels = range(len(rdm)) if category_labels is None else category_labels

    # Visualize RDM.
    plt.imshow(rdm, **imshow_kwargs)
    plt.colorbar(label=distance_metric)
    plt.xticks(ticks=range(len(category_labels)), labels=category_labels, rotation=90)
    plt.yticks(ticks=range(len(category_labels)), labels=category_labels)
    plt.xlabel("Categories")
    plt.ylabel("Categories")
    plt.show()