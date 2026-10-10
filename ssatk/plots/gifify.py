import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from .plotutils import _pop_save_path_aliases


def gifify(
    plot_func,
    *fargs,
    save_path=None,
    array_arg_indices=(0, 1),
    array_kw_keys=None,
    mode="cumulative",          # "cumulative", "chunks", "sliding"
    chunk_size=None,
    step=None,
    start=0,
    end=None,
    fps=12,
    loop=0,
    dpi=None,
    verbose=False,
    inject_ax=False,
    ax_arg_index=None,
    ax_kw_key=None,
    fixed_limits=True,          # lock 2D/3D axes across frames
    fix_box_aspect_3d=True,     # keep cube aspect on 3D panes
    **fkwargs
):
    """
    Animate any Matplotlib plot function by slicing two array args and compiling frames into a GIF.

    Works with pyplot functions (return None), functions returning Axes, or Figure.
    If your function expects an Axes, set `inject_ax=True` and specify where with
    `ax_arg_index` (positional) or `ax_kw_key` (keyword).

    """
    # ---------- defaults & validation ----------
    save_path, fkwargs = _pop_save_path_aliases(fkwargs, save_path=save_path)

    if save_path is None:
        from .figpath import figpath
        save_path = figpath("animation.gif")

    if array_kw_keys is not None and len(array_kw_keys) != 2:
        raise ValueError("array_kw_keys must be a tuple of two names or None.")
    if array_kw_keys is None and (not isinstance(array_arg_indices, (list, tuple)) or len(array_arg_indices) != 2):
        raise ValueError("array_arg_indices must be a tuple/list of two indices.")
    if mode not in {"cumulative", "chunks", "sliding"}:
        raise ValueError("mode must be 'cumulative', 'chunks', or 'sliding'.")

    args = list(fargs)
    kwargs = dict(fkwargs)

    # Resolve the two arrays to animate
    def _get_arrays_full():
        if array_kw_keys is not None:
            a = np.asarray(kwargs[array_kw_keys[0]])
            b = np.asarray(kwargs[array_kw_keys[1]])
        else:
            a = np.asarray(args[array_arg_indices[0]])
            b = np.asarray(args[array_arg_indices[1]])
        return a, b

    A_full, B_full = _get_arrays_full()
    n = min(len(A_full), len(B_full))
    if end is None:
        end = n
    end = min(end, n)
    if not (0 <= start < end):
        raise ValueError("Invalid start/end range.")

    # Optional 3D limit precompute (meters -> km), passed as `limit` if plot supports it
    def _as_3d(a):
        a = np.asarray(a)
        return a if (a.ndim == 2 and a.shape[1] >= 3) else None

    limit_km = None
    if fixed_limits:
        cand = _as_3d(A_full)
        if cand is None:
            cand = _as_3d(B_full)
        if cand is not None:
            try:
                maxabs_km = float(np.max(np.abs(cand[:, :3])) / 1e3)  # meters -> km
                limit_km = max(10.0, maxabs_km) * 1.02               # small pad, min cube
                kwargs.setdefault("limit", limit_km)                 # only if plot supports it
                if verbose:
                    print(f"[gifify] Precomputed 3D limits (km): ±{limit_km:.3f}")
            except Exception as e:
                if verbose:
                    print(f"[gifify] Could not precompute 3D limits: {e}")

    if mode in {"chunks", "sliding"} and (chunk_size is None or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer for 'chunks' or 'sliding'.")
    if step is None:
        step = 1 if mode == "cumulative" else chunk_size
    if step <= 0:
        raise ValueError("step must be positive.")

    # ---------- build spans ----------
    spans = []
    if mode == "cumulative":
        for i1 in range(start + 1, end + 1, step):
            spans.append((start, i1))
        if spans[-1][1] != end:
            spans.append((start, end))
    elif mode == "chunks":
        i = start
        while i < end:
            i1 = min(i + chunk_size, end)
            spans.append((i, i1))
            i += step
        if not spans:
            spans.append((start, min(start + chunk_size, end)))
    else:  # sliding
        i = start
        while i < end:
            i0 = i
            i1 = min(i0 + chunk_size, end)
            if i0 >= i1:
                break
            spans.append((i0, i1))
            i += step
        if not spans:
            spans.append((start, min(start + chunk_size, end)))

    def _apply_slice(i0, i1):
        args_s = list(args)
        kwargs_s = dict(kwargs)
        if array_kw_keys is not None:
            kwargs_s[array_kw_keys[0]] = np.asarray(kwargs_s[array_kw_keys[0]])[i0:i1]
            kwargs_s[array_kw_keys[1]] = np.asarray(kwargs_s[array_kw_keys[1]])[i0:i1]
        else:
            args_s[array_arg_indices[0]] = np.asarray(args_s[array_arg_indices[0]])[i0:i1]
            args_s[array_arg_indices[1]] = np.asarray(args_s[array_arg_indices[1]])[i0:i1]
        return args_s, kwargs_s

    def _resolve_fig(ret):
        if hasattr(ret, "figure") and ret.__class__.__name__.lower().endswith("axes"):
            return ret.figure
        if "Figure" in str(type(ret)):
            return ret
        return plt.gcf()

    # ---------- geometry-based axes matching (to keep limits stable) ----------
    def _ax_geom_key(ax, rnd=3):
        try:
            pos = ax.get_position().frozen()
            x0, y0, w, h = pos.x0, pos.y0, pos.width, pos.height
        except Exception:
            x0 = y0 = 0.0
            w = h = 1.0
        is3d = hasattr(ax, "get_zlim")
        return (round(x0, rnd), round(y0, rnd), round(w, rnd), round(h, rnd), "3d" if is3d else "2d")

    def _capture_limits_and_layout(fig):
        """
        Return a mapping keyed by axis geometry containing:
          - x/y/z limits
          - axis labels and title (to prevent disappearing text)
        """
        limits_map = {}
        for ax in fig.axes:
            key = _ax_geom_key(ax)
            try:
                xlim = getattr(ax, "get_xlim3d", ax.get_xlim)()
            except Exception:
                xlim = ax.get_xlim()
            try:
                ylim = getattr(ax, "get_ylim3d", ax.get_ylim)()
            except Exception:
                ylim = ax.get_ylim()
            zlim = ax.get_zlim() if hasattr(ax, "get_zlim") else None

            limits_map[key] = {
                "x": xlim,
                "y": ylim,
                "z": zlim,
                "xlabel": ax.get_xlabel() or "",
                "ylabel": ax.get_ylabel() or "",
                "title": ax.get_title() or "",
            }
        return limits_map

    def _apply_limits_by_layout(fig, limits_map):
        for ax in fig.axes:
            key = _ax_geom_key(ax)
            ent = limits_map.get(key)
            if not ent:
                continue
            try:
                # Limits
                if ent["x"] is not None:
                    if hasattr(ax, "set_xlim3d"):
                        ax.set_xlim3d(ent["x"])
                    else:
                        ax.set_xlim(ent["x"])
                if ent["y"] is not None:
                    if hasattr(ax, "set_ylim3d"):
                        ax.set_ylim3d(ent["y"])
                    else:
                        ax.set_ylim(ent["y"])
                if ent["z"] is not None and hasattr(ax, "set_zlim3d"):
                    ax.set_zlim3d(ent["z"])
                # Labels & title: keep them present every frame
                try:
                    ax.set_xlabel(ent.get("xlabel", ""))
                    ax.set_ylabel(ent.get("ylabel", ""))
                    ttl = ent.get("title", "")
                    if ttl:
                        ax.set_title(ttl)
                except Exception:
                    pass
                # Optional: stabilize 3D geometry
                if fix_box_aspect_3d and hasattr(ax, "set_box_aspect"):
                    ax.set_box_aspect((1, 1, 1))
                try:
                    ax.set_proj_type("ortho")
                except Exception:
                    pass
            except Exception:
                pass

    # ---------- probe render (capture fixed limits/layout) ----------
    saved_limits = None
    if fixed_limits:
        try:
            probe_args, probe_kwargs = _apply_slice(start, end)
            for key in ("save_path", "show"):
                if key in probe_kwargs:
                    probe_kwargs[key] = False

            if inject_ax:
                fig_probe, ax_inj = plt.subplots()
                if ax_kw_key is not None:
                    probe_kwargs[ax_kw_key] = ax_inj
                elif ax_arg_index is not None:
                    probe_args = list(probe_args)
                    probe_args.insert(ax_arg_index, ax_inj)
                else:
                    probe_kwargs["ax"] = ax_inj

            plt.close("all")
            ret_probe = plot_func(*probe_args, **probe_kwargs)
            fig_probe = _resolve_fig(ret_probe)
            try:
                fig_probe.canvas.draw_idle()
                plt.pause(0.001)
            except Exception:
                pass

            saved_limits = _capture_limits_and_layout(fig_probe)
            plt.close(fig_probe)
            if verbose:
                print(f"[gifify] Captured {len(saved_limits)} axes for fixed limits/layout.")
        except Exception as e:
            saved_limits = None
            if verbose:
                print(f"[gifify] Probe failed; continuing without fixed limits (reason: {e})")

    # ---------- frame render ----------
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        if verbose:
            print(f"Saving frames to temp dir: {tmpdir}")

        with imageio.get_writer(save_path, mode="I", duration=1.0 / fps, loop=loop) as writer:
            for frame_idx, (i0, i1) in enumerate(spans):
                plt.close("all")
                args_s, kwargs_s = _apply_slice(i0, i1)
                if fixed_limits and (limit_km is not None):
                    kwargs_s.setdefault("limit", limit_km)

                if inject_ax:
                    fig_tmp, injected_ax = plt.subplots()
                    if ax_kw_key is not None:
                        kwargs_s[ax_kw_key] = injected_ax
                    elif ax_arg_index is not None:
                        args_s = list(args_s)
                        args_s.insert(ax_arg_index, injected_ax)
                    else:
                        kwargs_s["ax"] = injected_ax

                ret = plot_func(*args_s, **kwargs_s)
                fig_to_save = _resolve_fig(ret)

                # lock axes/labels
                if fixed_limits and saved_limits:
                    _apply_limits_by_layout(fig_to_save, saved_limits)

                if dpi is not None:
                    try:
                        fig_to_save.set_dpi(dpi)
                    except Exception:
                        pass

                frame_path = os.path.join(tmpdir, f"frame_{frame_idx:06d}.png")
                fig_to_save.savefig(frame_path, dpi=dpi if dpi is not None else fig_to_save.get_dpi(), bbox_inches="tight", pad_inches=0.2)
                writer.append_data(imageio.imread(frame_path))
                plt.close(fig_to_save)

                if verbose:
                    print(f"Rendered frame {frame_idx + 1}/{len(spans)}  slice=[{i0}:{i1}]")

    if verbose:
        print(f"GIF saved to {save_path}")
    return {
        "frames": len(spans),
        "path": save_path,
        "mode": mode,
        "chunk_size": chunk_size,
        "step": step,
        "range": (start, end),
        "fixed_limits": fixed_limits,
    }
