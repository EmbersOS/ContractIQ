"""Abstract base class for storage backends."""
from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Abstract storage backend — all implementations must satisfy this interface."""

    @abstractmethod
    def write_file(self, path: str, data: bytes) -> str:
        """Write *data* to *path* and return the stored path."""

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        """Read and return the bytes stored at *path*."""

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """Delete the file at *path*."""

    @abstractmethod
    def get_url(self, path: str) -> str:
        """Return a URL or resolvable reference for *path*."""

