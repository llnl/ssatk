from pathlib import Path
from os import PathLike
from glob import glob
import warnings

import numpy as np
import imageio.v2 as imageio
from PIL import Image

from ssatk._sorting import natural_key


def _sort_frames(paths, warn_on_ambiguous: bool = True) -> list:
    """
    Sort paths naturally. Warn if keys collide or paths look unsortable.
    """
    paths = [str(p) for p in paths]
    if not paths:
        return []

    try:
        keys = [natural_key(p) for p in paths]
    except Exception as e:
        if warn_on_ambiguous:
            warnings.warn(
                f"Could not build natural sort keys: {e}. "
                f"Falling back to lexicographic ordering."
            )
        return sorted(paths)

    # Detect key collisions
    key_map = {}
    for p, k in zip(paths, keys):
        key_map.setdefault(tuple(k), []).append(p)
    collisions = [v for v in key_map.values() if len(v) > 1]

    if warn_on_ambiguous and collisions:
        sample = collisions[0]
        warnings.warn(
            "Multiple files share the same natural sort key. "
            "Order may be ambiguous. Examples: "
            + ", ".join(sample[:3])
            + (" ..." if len(sample) > 3 else "")
        )

    try:
        paths_sorted = [p for _, p in sorted(zip(keys, paths), key=lambda x: x[0])]
    except Exception:
        if warn_on_ambiguous:
            warnings.warn(
                "Natural sort failed. Falling back to lexicographic ordering."
            )
        paths_sorted = sorted(paths)

    # Heuristic: if lexicographic differs a lot from natural, hint to the user
    lex = sorted(paths)
    if warn_on_ambiguous and paths_sorted[:5] != lex[:5]:
        warnings.warn(
            "Natural sort order differs from lexicographic. "
            "If this is not intended, consider zero-padding numeric indices."
        )

    return paths_sorted


def _fit_to_canvas(
    img: Image.Image,
    target_size: tuple,
    bg_color=(255, 255, 255, 0),
    resample=Image.LANCZOS,
) -> Image.Image:
    """
    Preserve aspect ratio: scale to fit within target_size, then pad to canvas.
    Always returns RGBA image with exact target_size.
    """
    W, H = target_size
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    w, h = img.size
    if w == 0 or h == 0 or W <= 0 or H <= 0:
        raise ValueError("Invalid image or target size.")

    scale = min(W / float(w), H / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), resample=resample)

    canvas = Image.new("RGBA", (W, H), bg_color)
    x0 = (W - new_w) // 2
    y0 = (H - new_h) // 2
    canvas.paste(img, (x0, y0), img)
    return canvas


def _resolve_frames(frames):
    """
    Accept either:
      - iterable of paths
      - a glob pattern string/path like '/folder/frame_*.jpg'

    Returns a list of file paths as strings.
    Requires at least 2 frames.
    """
    if isinstance(frames, (str, PathLike)):
        paths = glob(str(frames))
    else:
        paths = [str(p) for p in frames]

    if not paths:
        raise ValueError("No frames found.")

    if len(paths) < 2:
        raise ValueError("At least 2 frames are required to make a GIF.")

    return paths


def write_gif(
    gif_name: str,
    frames,
    fps: int = 30,
    *,
    duration: float = None,   # seconds per frame; overrides fps if provided
    loop: int = 0,            # 0 = forever
    sort_frames: bool = True,
    warn_on_ambiguous: bool = True,
    uniform_size: bool = True,          # make all frames the same size
    target_size: tuple = None,          # if None, use size of first frame
    bg_color=(255, 255, 255, 0),        # padding color if sizes differ (RGBA)
    verbose: bool = False,
) -> None:
    """
    Write frames to an animated GIF with robust sorting and size normalization.

    Parameters
    ----------
    gif_name : str
        Output path ending with .gif
    frames : iterable of str/Path or str/Path glob pattern
        A list or iterable of image paths, or a glob pattern such as
        ``'/folder/frame_*.png'``. It must resolve to at least two frames.
    fps : int
        Frames per second (ignored if duration is provided)
    duration : float or None
        Seconds per frame. If provided, it overrides fps.
    loop : int
        0 = loop forever; positive integers loop that many times.
    sort_frames : bool
        If True, apply natural sort to the input paths.
    warn_on_ambiguous : bool
        Emit warnings when sorting is ambiguous.
    uniform_size : bool
        If True, all frames are resized/padded to a common size.
    target_size : (W, H) or None
        If None, use the size of the first image as the canvas.
    bg_color : tuple
        RGBA background for padding (when uniform_size is True).
    verbose : bool
        If True, print progress messages while writing.
    """
    out = Path(gif_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".gif":
        raise ValueError("gif_name must end with .gif")

    paths = _resolve_frames(frames)

    if sort_frames:
        paths = _sort_frames(paths, warn_on_ambiguous=warn_on_ambiguous)

    # Determine duration
    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        frame_duration = float(duration)
    else:
        if fps <= 0:
            raise ValueError("fps must be positive")
        frame_duration = 1.0 / float(fps)

    # Determine target size
    if uniform_size and target_size is None:
        with Image.open(paths[0]) as first_img:
            target_size = first_img.size

    # ImageIO's Pillow GIF backend expects milliseconds here, even though the
    # public SSATK API documents `duration` as seconds per frame. Passing the
    # seconds value directly produces zero-delay GIFs on current ImageIO/Pillow.
    frame_duration_ms = frame_duration * 1000.0

    # Write GIF
    if verbose:
        print(f"Writing gif: {out}")
    with imageio.get_writer(out, mode="I", duration=frame_duration_ms, loop=loop) as writer:
        for p in paths:
            arr = imageio.imread(p)
            img = Image.fromarray(arr)

            if uniform_size:
                if target_size is None:
                    target_size = img.size
                img = _fit_to_canvas(img, target_size, bg_color=bg_color)
                arr = np.array(img)

            writer.append_data(arr)

    if verbose:
        print(f"Wrote {out}")


if __name__ == "__main__":
    # Example 1: wildcard pattern
    # write_gif("output.gif", "/folder/frame_*.jpg", fps=10)

    # Example 2: explicit list
    # write_gif("output.gif", ["frame_1.jpg", "frame_2.jpg", "frame_10.jpg"], fps=10)

    pass
