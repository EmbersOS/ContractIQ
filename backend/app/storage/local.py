"""Local filesystem storage backend."""
from __future__ import annotations

import os

from app.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """Stores files on the local filesystem under *base_dir*."""

    def __init__(self, base_dir: str = "./uploads") -> None:
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _full_path(self, path: str) -> str:
        """Resolve *path* relative to base_dir, preventing path traversal."""
        full = os.path.abspath(os.path.join(self.base_dir, path))
        try:
            if os.path.commonpath([self.base_dir, full]) != self.base_dir:
                raise ValueError(f"Path traversal detected: {path!r}")
        except ValueError:
            raise ValueError(f"Path traversal detected: {path!r}")
        return full

    def write_file(self, path: str, data: bytes) -> str:
        full = self._full_path(path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as fh:
            fh.write(data)
        return path

    def read_file(self, path: str) -> bytes:
        full = self._full_path(path)
        with open(full, "rb") as fh:
            return fh.read()

    def delete_file(self, path: str) -> None:
        full = self._full_path(path)
        if os.path.exists(full):
            os.remove(full)

    def get_url(self, path: str) -> str:
        return self._full_path(path)

