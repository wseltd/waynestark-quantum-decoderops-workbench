"""SHA256 helpers for packaging, fingerprinting, and artefact verification.

Digests produced here must match ``sha256sum`` on Linux byte-for-byte so that
customers can independently verify shipped tarball contents.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

# 1 MiB. Chosen so 100 MB+ ONNX engines and .pt checkpoints stream without
# materialising the whole file in memory, while remaining large enough that
# per-read overhead is negligible on typical storage.
DEFAULT_CHUNK_SIZE: int = 1048576


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase hex SHA256 digest of ``data``.

    Args:
        data: Bytes to hash.

    Returns:
        64-character lowercase hexadecimal digest.
    """
    return hashlib.sha256(data).hexdigest()


def sha256_file(
    path: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> str:
    """Return the lowercase hex SHA256 digest of the file at ``path``.

    The file is read in binary mode in ``chunk_size`` blocks so that very large
    artefacts (ONNX engines, PyTorch checkpoints) do not have to fit in memory.

    Args:
        path: Filesystem path to a regular file.
        chunk_size: Read buffer size in bytes. Defaults to ``DEFAULT_CHUNK_SIZE``
            (1 MiB).

    Returns:
        64-character lowercase hexadecimal digest.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` exists but is not a regular file.
        ValueError: If ``chunk_size`` is not a positive integer.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size}"
        )

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"sha256_file: path does not exist: {resolved}")
    if not resolved.is_file():
        raise IsADirectoryError(
            f"sha256_file: path is not a regular file: {resolved}"
        )

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
