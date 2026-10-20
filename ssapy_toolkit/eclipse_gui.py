"""GUI-safe render-job adapter for eclipse products.

The canonical rendering API is intentionally synchronous and deterministic.
This module provides a toolkit-neutral worker abstraction with structured
progress, preview completion, warnings, best-effort cancellation, and a stable
result schema.  Desktop applications can use it without depending on Qt, Tk,
wx, or a particular web-view implementation.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable
from uuid import uuid4
import traceback

from ssapy_toolkit.eclipse_api_impl import RenderRequest, build_event, render_product, validate_event
from ssapy_toolkit.compute.eclipse_reference_events import ReferenceEvent

ProgressCallback = Callable[["RenderProgress"], None]
WarningCallback = Callable[[str], None]
PreviewCallback = Callable[[str | dict[str, object]], None]


@dataclass(frozen=True)
class RenderProgress:
    job_id: str
    stage: str
    fraction: float
    message: str
    output: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "$schema": "ssapy-toolkit.eclipse.gui-progress/2.2",
            "job_id": self.job_id,
            "stage": self.stage,
            "fraction": float(self.fraction),
            "message": self.message,
            "output": self.output,
        }


@dataclass(frozen=True)
class RenderJobResult:
    job_id: str
    status: str
    outputs: dict[str, object]
    warnings: tuple[str, ...]
    validation: dict[str, object] | None
    requested_minimum_state_count: int | None
    resolved_state_count: int | None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"

    def to_dict(self) -> dict[str, object]:
        return {
            "$schema": "ssapy-toolkit.eclipse.gui-render-result/2.2",
            "job_id": self.job_id,
            "status": self.status,
            "outputs": self.outputs,
            "warnings": list(self.warnings),
            "validation": self.validation,
            "requested_minimum_state_count": self.requested_minimum_state_count,
            "resolved_state_count": self.resolved_state_count,
            "error": self.error,
        }


class RenderCancelledError(RuntimeError):
    """Raised internally when a cooperative cancellation point is reached."""


# Friendly short alias for GUI code.
RenderCancelled = RenderCancelledError


_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = Lock()


def _shared_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ssapy-eclipse")
        return _EXECUTOR


def shutdown_gui_executor(*, wait: bool = True) -> None:
    """Shut down the package-owned worker pool; safe to call repeatedly."""
    global _EXECUTOR
    with _EXECUTOR_LOCK:
        executor, _EXECUTOR = _EXECUTOR, None
    if executor is not None:
        executor.shutdown(wait=bool(wait), cancel_futures=True)


class EclipseRenderJob:
    """One background-capable eclipse render job.

    Cancellation is cooperative between expensive stages.  A renderer already
    writing one atomic HTML/PNG/MP4 product is allowed to finish safely; the job
    then stops before the next stage.  Applications requiring immediate hard
    cancellation can run this object inside their own process and terminate the
    process after calling :meth:`cancel`.
    """

    def __init__(
        self,
        request: RenderRequest,
        *,
        create_preview: bool = True,
        progress_callback: ProgressCallback | None = None,
        warning_callback: WarningCallback | None = None,
        preview_callback: PreviewCallback | None = None,
        job_id: str | None = None,
    ) -> None:
        self.request = request
        self.create_preview = bool(create_preview)
        self.progress_callback = progress_callback
        self.warning_callback = warning_callback
        self.preview_callback = preview_callback
        self.job_id = str(job_id or uuid4())
        self._cancel = Event()
        self._future: Future[RenderJobResult] | None = None

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def future(self) -> Future[RenderJobResult] | None:
        return self._future

    def cancel(self) -> bool:
        """Request cancellation and cancel a queued future when possible."""
        self._cancel.set()
        return bool(self._future.cancel()) if self._future is not None else True

    def _check_cancelled(self) -> None:
        if self.cancelled:
            raise RenderCancelledError("eclipse render cancelled by user")

    def _safe_callback(self, callback, value, warnings: list[str]) -> None:
        if callback is None:
            return
        try:
            callback(value)
        except Exception as exc:  # a GUI callback must never corrupt the render
            warnings.append(f"callback failed: {type(exc).__name__}: {exc}")

    def _emit(self, warnings: list[str], stage: str, fraction: float, message: str,
              output: str | None = None) -> None:
        progress = RenderProgress(
            self.job_id, stage, max(0.0, min(1.0, float(fraction))), str(message), output
        )
        self._safe_callback(self.progress_callback, progress, warnings)

    def _warn(self, warnings: list[str], message: str) -> None:
        warnings.append(str(message))
        self._safe_callback(self.warning_callback, str(message), warnings)

    @staticmethod
    def _preview_path(request: RenderRequest) -> Path:
        output = Path(request.output).expanduser()
        if str(request.product).lower() == "scientific-suite" or not output.suffix:
            return output / "preview_interactive_v22_2.html"
        return output.with_name(output.stem + "_preview" + output.suffix)

    @staticmethod
    def _normalise_output(value: Any) -> dict[str, object]:
        return dict(value) if isinstance(value, dict) else {"primary": str(value)}

    def run(self) -> RenderJobResult:
        warnings: list[str] = []
        outputs: dict[str, object] = {}
        validation_payload: dict[str, object] | None = None
        requested: int | None = None
        resolved: int | None = None
        try:
            self._emit(warnings, "build-event", 0.02, "Building immutable eclipse event")
            self._check_cancelled()
            if isinstance(self.request.event, ReferenceEvent):
                event = self.request.event
            else:
                requested = int(self.request.event.minimum_sample_count)
                event = build_event(self.request.event)
            requested = int(event.metadata.get("requested_minimum_state_count", requested or len(event.jd)))
            resolved = int(len(event.jd))

            self._emit(warnings, "validate-event", 0.12, f"Validating {resolved} resolved solver states")
            self._check_cancelled()
            validation = validate_event(event)
            validation_payload = validation.to_dict()
            for warning in validation.warnings:
                self._warn(warnings, warning)
            if not validation.passed:
                raise RuntimeError("event validation failed before rendering")

            if self.create_preview:
                self._emit(warnings, "preview", 0.22, "Rendering fast interactive preview")
                self._check_cancelled()
                # Scientific-suite preview is deliberately a single fast 3-D
                # view rather than a nested suite directory.
                preview_request = RenderRequest(
                    product="cinematic",
                    output=self._preview_path(self.request),
                    event=event,
                    quality="motion",
                    animate=False,
                    playback_seconds=None,
                    photometry=self.request.photometry,
                )
                preview = render_product(preview_request)
                outputs["preview"] = preview
                self._safe_callback(self.preview_callback, preview, warnings)
                self._emit(warnings, "preview-complete", 0.42, "Preview complete", str(preview))

            self._check_cancelled()
            self._emit(warnings, "final-render", 0.48, "Rendering final eclipse product")
            final = render_product(replace(self.request, event=event))
            outputs["final"] = final
            self._check_cancelled()
            self._emit(warnings, "complete", 1.0, "Eclipse render completed")
            return RenderJobResult(
                self.job_id, "completed", outputs, tuple(warnings), validation_payload,
                requested, resolved,
            )
        except RenderCancelledError as exc:
            self._emit(warnings, "cancelled", 1.0, str(exc))
            return RenderJobResult(
                self.job_id, "cancelled", outputs, tuple(warnings), validation_payload,
                requested, resolved, str(exc),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self._warn(warnings, message)
            self._emit(warnings, "failed", 1.0, message)
            return RenderJobResult(
                self.job_id, "failed", outputs, tuple(warnings), validation_payload,
                requested, resolved, message + "\n" + traceback.format_exc(),
            )

    def start(self, executor: ThreadPoolExecutor | None = None) -> Future[RenderJobResult]:
        """Submit the job to a GUI-safe worker thread and return its Future."""
        if self._future is not None and not self._future.done():
            raise RuntimeError("render job is already running")
        self._future = (executor or _shared_executor()).submit(self.run)
        return self._future


__all__ = [
    "EclipseRenderJob", "RenderCancelled", "RenderCancelledError",
    "RenderJobResult", "RenderProgress", "shutdown_gui_executor",
]
