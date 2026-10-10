import matplotlib.pyplot as plt
from matplotlib import gridspec

from .plotutils import figsave, _pop_save_path_aliases, _raise_unrecognized_kwargs


def build_dashboard(
    panels,
    *,
    nrows,
    ncols,
    figsize=(14, 8),
    dpi=100,
    facecolor="white",
    save_path=None,
    show=False,
    wspace=0.25,
    hspace=0.35,
    **save_kwargs,
):
    """
    Build a custom dashboard from user-provided panel renderers.

    Parameters
    ----------
    panels : list[dict]
        Each entry describes one panel::

            {
                "loc": (r, c) or (r0, r1, c0, c1),
                "projection": None or "3d",
                "render": callable(ax, fig, **kwargs),
                "kwargs": dict (optional),
            }
    nrows, ncols : int
        GridSpec dimensions.
    save_path : str or None
        If provided, saves the figure.
    show : bool
        If True, displays the figure.

    Returns
    -------
    fig, axes, outputs
        axes: list of axes created (one per panel)
        outputs: list of return values from each panel's render()
    """
    save_path, save_kwargs = _pop_save_path_aliases(save_kwargs, save_path=save_path)
    _raise_unrecognized_kwargs(save_kwargs, "build_dashboard")

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=facecolor)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig)
    gs.update(wspace=wspace, hspace=hspace)

    axes = []
    outputs = []

    for p in panels:
        loc = p["loc"]
        projection = p.get("projection", None)
        render = p["render"]
        rkwargs = p.get("kwargs", {})

        if len(loc) == 2:
            r, c = loc
            ss = gs[r, c]
        elif len(loc) == 4:
            r0, r1, c0, c1 = loc
            ss = gs[r0:r1, c0:c1]
        else:
            raise ValueError(f"Bad loc={loc}; expected (r,c) or (r0,r1,c0,c1).")

        ax = fig.add_subplot(ss, projection=projection) if projection else fig.add_subplot(ss)
        axes.append(ax)
        outputs.append(render(ax=ax, fig=fig, **rkwargs))

    if save_path:
        figsave(fig, save_path)
    if show:
        plt.show()

    return fig, axes, outputs
