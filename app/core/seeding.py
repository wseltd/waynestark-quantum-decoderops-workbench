"""Deterministic per-worker RNG seed derivation.

The reproducibility fingerprint embeds a master seed plus the per-worker
seeds derived from it. Workers must seed independent RNG streams from a
single integer the operator can record, and two runs of the same
benchmark on different machines must produce identical worker seeds for
the same ``(master_seed, worker_index)`` pair.

Why SHA256 with explicit byte widths (the design choice this module exists
to make):
    * Cross-platform stability is the contract. ``hash()`` is randomised
      per process, ``random.Random`` reseeding diverges across CPython
      versions, and naive ``master_seed * 1000 + i`` collides whenever a
      caller picks an adjacent master seed. SHA256 over a fixed-width
      big-endian encoding has none of those failure modes.
    * 8-byte master + 4-byte worker is a deliberate, documented frame.
      Big-endian fixes byte order across architectures; fixed widths
      mean ``derive_worker_seed(1, 256) != derive_worker_seed(256, 1)``,
      which would not hold for ``str(master) + str(worker)``.

Why mask to 63 bits: ``numpy.random.SeedSequence`` accepts uint64 but
``random.Random.seed`` and several adapter libraries truncate to a
signed 64-bit integer. Masking to the low 63 bits gives a value that is
non-negative and round-trips through every common RNG entry point we
use without overflow surprises. Rejected: returning the full 64-bit
digest prefix and asking callers to mask — that pushes a subtle
correctness requirement onto every call site.

Restraint: this module does not read environment variables, does not
touch the filesystem, does not initialise any global RNG, and does not
import numpy. Seeds are derived on demand; callers own RNG construction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

# 8-byte uint64 master seed and 4-byte uint32 worker index. These widths
# are the wire format of the seed derivation: do not change without
# bumping a fingerprint schema version, since it would silently re-derive
# every historical seed.
_MASTER_SEED_BYTES = 8
_WORKER_INDEX_BYTES = 4
# Typed as a Literal so int.to_bytes / int.from_bytes accept it without a
# cast — they refuse a plain ``str`` typed alias.
_BYTE_ORDER: Literal["big"] = "big"

# Mask to 63 bits so the result fits any common RNG seed parameter
# (numpy uint64, stdlib signed 64-bit) without overflow.
_SEED_BITS = 63
_SEED_MASK = (1 << _SEED_BITS) - 1

_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1


def derive_worker_seed(master_seed: int, worker_index: int) -> int:
    """Derive a deterministic per-worker RNG seed from a master seed.

    Computes ``SHA256(master_seed_be8 || worker_index_be4)`` and returns
    the first 8 bytes of the digest interpreted as a big-endian uint64,
    masked to 63 bits.

    Args:
        master_seed: The run-level master seed. Must be a non-negative
            integer that fits in an unsigned 64-bit field.
        worker_index: Zero-based worker index. Must be a non-negative
            integer that fits in an unsigned 32-bit field.

    Returns:
        An integer in ``[0, 2**63)`` suitable for seeding any
        numpy/stdlib RNG without overflow.

    Raises:
        ValueError: If ``master_seed`` or ``worker_index`` is not an
            integer of the right sign and width. The error message names
            the offending parameter so callers can fix it without
            spelunking through a stack trace.
    """
    # bool is a subclass of int; reject it explicitly so True/False
    # cannot silently masquerade as a seed.
    if not isinstance(master_seed, int) or isinstance(master_seed, bool):
        raise ValueError(
            f"master_seed must be an int, got {type(master_seed).__name__}"
        )
    if not isinstance(worker_index, int) or isinstance(worker_index, bool):
        raise ValueError(
            f"worker_index must be an int, got {type(worker_index).__name__}"
        )
    if master_seed < 0 or master_seed > _UINT64_MAX:
        raise ValueError(
            "master_seed must be in [0, 2**64); "
            f"got {master_seed}"
        )
    if worker_index < 0 or worker_index > _UINT32_MAX:
        raise ValueError(
            "worker_index must be in [0, 2**32); "
            f"got {worker_index}"
        )

    payload = master_seed.to_bytes(
        _MASTER_SEED_BYTES, _BYTE_ORDER, signed=False
    ) + worker_index.to_bytes(
        _WORKER_INDEX_BYTES, _BYTE_ORDER, signed=False
    )
    digest = hashlib.sha256(payload).digest()
    head = int.from_bytes(digest[:_MASTER_SEED_BYTES], _BYTE_ORDER, signed=False)
    return head & _SEED_MASK


def derive_worker_seeds(master_seed: int, num_workers: int) -> list[int]:
    """Derive ``num_workers`` per-worker seeds from a single master seed.

    Args:
        master_seed: The run-level master seed; same constraints as
            :func:`derive_worker_seed`.
        num_workers: Non-negative count of workers. ``0`` returns an
            empty list, which is a legal degenerate case for a runner
            that decides at the last moment not to fan out.

    Returns:
        A list of integer seeds, one per worker, in worker-index order.

    Raises:
        ValueError: If ``num_workers`` is not a non-negative int.
            ``master_seed`` errors are surfaced from
            :func:`derive_worker_seed`.
    """
    if not isinstance(num_workers, int) or isinstance(num_workers, bool):
        raise ValueError(
            f"num_workers must be an int, got {type(num_workers).__name__}"
        )
    if num_workers < 0 or num_workers > _UINT32_MAX:
        raise ValueError(
            "num_workers must be in [0, 2**32); "
            f"got {num_workers}"
        )
    return [derive_worker_seed(master_seed, i) for i in range(num_workers)]


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """Master seed plus the derived per-worker seeds for one run.

    Stored verbatim in the reproducibility fingerprint. ``worker_seeds``
    is a tuple (not a list) so the dataclass stays hashable and the
    fingerprint cannot be mutated after the run starts.
    """

    master_seed: int
    worker_seeds: tuple[int, ...]
