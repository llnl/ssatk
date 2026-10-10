from .write_gifs import _sort_frames
import numpy as np
import warnings
import glob
from pathlib import Path

try:
    import cv2
except (ImportError, OSError) as exc:
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


def write_video(
    video_name,
    frames,
    fps: int = 30,
    *,
    sort_frames: bool = True,
    warn_on_ambiguous: bool = True,
    target_size: tuple = None,         # (width, height); if None, use first frame
    freeze_last_seconds: float = 0.0,  # extra time to hold last frame
    verbose: bool = False,
):
    """
    Write frames to an MP4 video file, with natural sorting and uniform size.

    Parameters
    ----------
    video_name : str
        Output path ending with .mp4
    frames : iterable of str, str, or Path
        - Iterable of image file paths, OR
        - Path to a folder (all files in it are used), OR
        - Wildcard pattern string (e.g. ``'frames_*.png'`` or
          ``'root/**/frame_*.png'``).
    fps : int
        Frame rate.
    sort_frames : bool
        If True, apply natural sort to the input paths.
    warn_on_ambiguous : bool
        Emit warnings when sorting is ambiguous.
    target_size : (W, H) or None
        Output frame size in pixels. If None, use the size of the first frame.
    freeze_last_seconds : float
        Extra number of seconds to “freeze” the last frame by repeating it.
    verbose : bool
        If True, print progress messages while writing.
    """
    if _CV2_IMPORT_ERROR is not None:
        raise ImportError(
            "write_video requires OpenCV; install ssatk[video]."
        ) from _CV2_IMPORT_ERROR

    # Normalize frames into a list of file paths
    if isinstance(frames, (str, Path)):
        pattern = str(frames)
        p = Path(pattern)
        if p.is_dir():
            # All files directly under this directory
            paths = [str(x) for x in p.iterdir() if x.is_file()]
        else:
            # Treat as a glob pattern (supports *, ?, **)
            recursive = "**" in pattern
            paths = glob.glob(pattern, recursive=recursive)
    else:
        paths = list(frames)

    if not paths:
        raise ValueError("frames list is empty (no files found)")

    if sort_frames:
        paths = _sort_frames(paths, warn_on_ambiguous=warn_on_ambiguous)

    if verbose:
        print(f"Writing video: {video_name}")

    # Read first frame to determine output size
    first = cv2.imread(paths[0])
    if first is None:
        raise ValueError(f"Could not read first frame: {paths[0]}")

    if target_size is None:
        h, w = first.shape[:2]
        W, H = int(w), int(h)
    else:
        W, H = int(target_size[0]), int(target_size[1])

    # Resize first frame if needed
    if (first.shape[1], first.shape[0]) != (W, H):
        first = cv2.resize(first, (W, H))

    # Ensure 3-channel BGR
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(video_name, fourcc, float(fps), (W, H))

    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for '{video_name}'. "
            "Your OpenCV/FFmpeg build may not support this codec/container."
        )

    last_frame = first
    writer.write(first)

    # Write remaining frames, always resizing to (W, H)
    for pth in paths[1:]:
        img = cv2.imread(pth)
        if img is None:
            warnings.warn(f"Could not read frame: {pth}; skipping.")
            continue

        if (img.shape[1], img.shape[0]) != (W, H):
            img = cv2.resize(img, (W, H))

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        writer.write(img)
        last_frame = img

    # Freeze on last frame
    if freeze_last_seconds > 0 and last_frame is not None:
        extra_frames = int(np.round(freeze_last_seconds * float(fps)))
        for _ in range(extra_frames):
            writer.write(last_frame)

    writer.release()
    if verbose:
        print(f"Wrote: {video_name}")
