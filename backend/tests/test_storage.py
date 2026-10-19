"""Tests for LocalStorageBackend."""
from __future__ import annotations

import os
import tempfile

import pytest

from app.storage.local import LocalStorageBackend


@pytest.fixture
def storage(tmp_path):
    return LocalStorageBackend(base_dir=str(tmp_path))


def test_round_trip(storage):
    """Write then read should return identical bytes."""
    data = b"hello, contractiq"
    storage.write_file("sub/test.bin", data)
    assert storage.read_file("sub/test.bin") == data


def test_delete(storage):
    """File should not exist after delete."""
    storage.write_file("to_delete.txt", b"bye")
    storage.delete_file("to_delete.txt")
    with pytest.raises(FileNotFoundError):
        storage.read_file("to_delete.txt")


def test_get_url(storage, tmp_path):
    """get_url should return an absolute path string under base_dir."""
    storage.write_file("myfile.pdf", b"data")
    url = storage.get_url("myfile.pdf")
    assert isinstance(url, str)
    assert os.path.isabs(url)
    assert os.path.exists(url)


def test_overwrite(storage):
    """Writing to the same path twice should overwrite content."""
    storage.write_file("overwrite.txt", b"first")
    storage.write_file("overwrite.txt", b"second")
    assert storage.read_file("overwrite.txt") == b"second"


def test_path_traversal_rejected(storage):
    """Path traversal attempts should raise ValueError."""
    with pytest.raises(ValueError):
        storage.write_file("../../etc/passwd", b"evil")


def test_prefix_collision_path_traversal_rejected(tmp_path):
    """Paths matching base_dir prefix but escaping to sibling dir should be rejected."""
    base = tmp_path / "uploads"
    base.mkdir()
    backend = LocalStorageBackend(base_dir=str(base))
    with pytest.raises(ValueError):
        backend.write_file("../uploads_fake/secret.txt", b"evil")

