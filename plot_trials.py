import matplotlib.pyplot as plt
from typing import Iterable, Sequence, Optional, Tuple

def basic_plot(
    y: Sequence,
    x: Optional[Sequence] = None,
    *,
    xlabel: str = "x",
    ylabel: str = "y",
    title: Optional[str] = None,
    grid: bool = True,
    save_path: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Quick line plot helper.

    Args:
        y: A 1D sequence of values, or a sequence of sequences for multiple series.
        x: Optional x-values. If None, uses index positions.
        xlabel, ylabel, title: Text labels.
        grid: Show a light grid if True.
        save_path: If provided (e.g. 'plot.png'), saves the figure.
        show: Call plt.show() if True.

    Returns:
        (fig, ax) for further customization.
    """
    fig, ax = plt.subplots()

    # Detect if y is "multiple series" (e.g., list of lists)
    is_multi = len(y) > 0 and hasattr(y[0], "__iter__") and not isinstance(y[0], (str, bytes))

    # Basic length checks
    if x is not None:
        n = len(x)
        if is_multi:
            for series in y:
                if len(series) != n:
                    raise ValueError("All series in y must have the same length as x.")
        else:
            if len(y) != n:
                raise ValueError("x and y must have the same length.")

    # Plot
    if is_multi:
        for i, series in enumerate(y, start=1):
            if x is None:
                ax.plot(series, marker="o", label=f"series {i}")
            else:
                ax.plot(x, series, marker="o", label=f"series {i}")
        ax.legend()
    else:
        if x is None:
            ax.plot(y, marker="o")
        else:
            ax.plot(x, y, marker="o")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax

basic_plot([1, 3, 2, 5], title="Demo", ylabel="Value")