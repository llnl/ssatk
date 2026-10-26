"""Temporary, range-readable HTTP files for remote HDF5 demonstrations.

The class in this module deliberately implements only ranged reads.  It is
intended for ``h5py.File`` and keeps downloaded blocks in a temporary
directory rather than in a persistent cache.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+|\*)$")


class TemporaryHTTPFile(io.RawIOBase):
    """A seekable, read-only file backed by HTTP range requests.

    ``expected_etag`` is required so that every request is protected against
    silently reading a changed remote object.  Only blocks touched by reads
    are downloaded, and all block files disappear when this object closes.
    """

    def __init__(
        self,
        url: str,
        *,
        expected_etag: str,
        max_bytes: int = 64 * 1024 * 1024,
        block_size: int = 64 * 1024,
        timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self.url = url
        self.source_url = url
        self.expected_etag = expected_etag
        self.source_etag = expected_etag
        self.max_bytes = max_bytes
        self.block_size = block_size
        self.timeout = timeout
        self.bytes_downloaded = 0
        self._position = 0
        self._cache: dict[int, Path] = {}
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        try:
            self._validate_arguments()
            self.size = self._head()
            self.source_size = self.size
            self._temporary_directory = tempfile.TemporaryDirectory(
                prefix="ssatk-http-"
            )
        except BaseException:
            self._cleanup()
            raise

    def _validate_arguments(self) -> None:
        if not isinstance(self.url, str) or not self.url:
            raise ValueError("url must be a non-empty string")
        if not self.expected_etag:
            raise ValueError("expected_etag is required")
        if isinstance(self.max_bytes, bool) or self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if isinstance(self.block_size, bool) or self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")

    def _head(self) -> int:
        request = Request(self.url, method="HEAD")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                status = getattr(response, "status", response.getcode())
                if status != 200:
                    raise OSError(f"HEAD returned HTTP {status}")
                length_text = response.headers.get("Content-Length")
                etag = response.headers.get("ETag")
        except HTTPError as exc:
            raise OSError(f"HEAD request failed for {self.url}: {exc}") from exc
        except URLError:
            raise
        if length_text is None:
            raise OSError("HEAD response lacks Content-Length")
        try:
            length = int(length_text)
        except ValueError as exc:
            raise OSError("HEAD response has invalid Content-Length") from exc
        if length < 0:
            raise OSError("HEAD response has negative Content-Length")
        if etag != self.expected_etag:
            raise OSError(
                f"ETag mismatch: expected {self.expected_etag!r}, got {etag!r}"
            )
        return length

    def _fetch_block(self, start: int) -> Path:
        cached = self._cache.get(start)
        if cached is not None:
            return cached
        end = min(start + self.block_size, self.size) - 1
        expected_length = end - start + 1
        if self.bytes_downloaded + expected_length > self.max_bytes:
            raise OSError("download byte cap exceeded")
        data = self._fetch_range(start, end, expected_length)
        if self._temporary_directory is None:
            raise ValueError("temporary storage is unavailable")
        path = Path(self._temporary_directory.name) / f"{start:020d}.block"
        with path.open("wb") as stream:
            stream.write(data)
        self._cache[start] = path
        self.bytes_downloaded += expected_length
        return path

    def _fetch_range(self, start: int, end: int, expected_length: int) -> bytes:
        """Retry a dropped connection; never retry a verified mismatch.

        ETag, Content-Range, Content-Length and truncated-body checks mean the
        remote object no longer matches what was pinned, so repeating the
        request could silently accept a changed file instead of failing
        loudly. Only the connection attempt and the body read can fail for
        purely transient reasons, so only those are retried.
        """
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                return self._request_range(start, end, expected_length)
            except (URLError, TimeoutError, ConnectionError):
                if attempt == attempts:
                    raise
                time.sleep(0.2 * attempt)

    def _request_range(self, start: int, end: int, expected_length: int) -> bytes:
        request = Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "If-Match": self.expected_etag,
            },
            method="GET",
        )
        try:
            response = urlopen(request, timeout=self.timeout)
        except HTTPError as exc:
            raise OSError(f"range request failed for {self.url}: {exc}") from exc
        try:
            status = getattr(response, "status", response.getcode())
            if status != 206:
                raise OSError(f"range request returned HTTP {status}, expected 206")
            response_etag = response.headers.get("ETag")
            if response_etag != self.expected_etag:
                raise OSError(
                    f"GET ETag mismatch: expected {self.expected_etag!r}, "
                    f"got {response_etag!r}"
                )
            content_range = response.headers.get("Content-Range")
            match = _CONTENT_RANGE.fullmatch(content_range or "")
            if (
                match is None
                or int(match.group(1)) != start
                or int(match.group(2)) != end
                or match.group(3) != str(self.size)
            ):
                raise OSError(
                    f"unexpected Content-Range {content_range!r}; "
                    f"expected bytes {start}-{end}/{self.size}"
                )
            content_length = response.headers.get("Content-Length")
            if content_length != str(expected_length):
                raise OSError(
                    f"unexpected Content-Length {content_length!r}; "
                    f"expected {expected_length}"
                )
            data = response.read(expected_length)
        finally:
            response.close()
        if len(data) != expected_length:
            raise OSError(
                f"range response contained {len(data)} bytes, expected {expected_length}"
            )
        return data

    def _read_at(self, position: int, size: int) -> bytes:
        result = bytearray()
        while size:
            start = (position // self.block_size) * self.block_size
            offset = position - start
            count = min(size, self.block_size - offset, self.size - position)
            if count <= 0:
                break
            path = self._fetch_block(start)
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read(count)
            if len(chunk) != count:
                raise OSError("cached block was truncated")
            result.extend(chunk)
            position += count
            size -= count
        return bytes(result)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        self._checkClosed()
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        self._checkClosed()
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise ValueError("negative seek position")
        self._position = position
        return position

    def read(self, size: int = -1) -> bytes:
        self._checkClosed()
        if size is None or size < 0:
            size = self.size - self._position
        if size == 0 or self._position >= self.size:
            return b""
        size = min(size, self.size - self._position)
        data = self._read_at(self._position, size)
        self._position += len(data)
        return data

    def readinto(self, buffer: bytearray | memoryview) -> int:
        self._checkClosed()
        view = memoryview(buffer)
        if view.readonly:
            raise TypeError("readinto() argument must be writable")
        try:
            data = self.read(view.nbytes)
            view.cast("B")[: len(data)] = data
            return len(data)
        finally:
            view.release()

    def _cleanup(self) -> None:
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None
        self._cache.clear()

    def close(self) -> None:
        if not self.closed:
            self._cleanup()
        super().close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
