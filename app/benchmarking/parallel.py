"""Parallel benchmark runner — execute RunConfigs across a spawn pool (T035).

Accepts an already-expanded list of :class:`RunConfig` (from T033) and a
picklable decoder-factory spec (``(module_path, callable_name)`` tuple)
that the worker imports inside the subprocess. Decoder objects never
cross the pickle boundary.

Workers use multiprocessing.get_context('spawn') — ``fork`` is unsafe with
CUDA and PyTorch; spawn is the only portable option.

Each worker derives its own numpy Generator seed from
``derive_worker_seed(master_seed, run_config.worker_seed_slot)`` so
results are deterministic across workers, runs, and hosts.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import replace
from typing import Callable, Sequence

from app.benchmarking.orchestrator import RunConfig
from app.benchmarking.runner import RunResult, run_single
from app.core.seeding import derive_worker_seed as _derive_one
from app.decoders.protocol import Decoder

__all__ = [
    "MAX_WORKERS_CAP",
    "derive_worker_seeds",
    "run_parallel",
]


MAX_WORKERS_CAP: int = 32

_logger = logging.getLogger(__name__)


def derive_worker_seeds(master_seed: int, configs: Sequence[RunConfig]) -> list[int]:
    """Derive per-RunConfig seeds from a master seed.

    Uses each RunConfig's ``worker_seed_slot`` — not its position in the
    list — so the seed is stable regardless of how the caller re-orders
    or slices the sweep.
    """
    return [_derive_one(master_seed, cfg.worker_seed_slot) for cfg in configs]


def _import_factory(spec: tuple[str, str]) -> Callable[[str], Decoder]:
    """Import the decoder factory inside the worker process."""
    module_path, name = spec
    import importlib

    module = importlib.import_module(module_path)
    factory = getattr(module, name)
    if not callable(factory):
        raise TypeError(
            f"decoder factory {module_path}:{name} is not callable"
        )
    return factory


def _worker_entry(
    args: tuple[RunConfig, tuple[str, str], int, int, int],
) -> RunResult:
    """Run one RunConfig in a subprocess. Never raises to the pool."""
    config, factory_spec, _master_seed, num_detectors, batch_size = args
    pid = os.getpid()
    _logger.info(
        "worker start",
        extra={"run_id": config.run_id, "pid": pid},
    )
    started = time.time()
    try:
        factory = _import_factory(factory_spec)
        result = run_single(
            config,
            decoder_factory=factory,
            num_detectors=num_detectors,
            batch_size=batch_size,
        )
    except BaseException as exc:  # noqa: BLE001 — must not kill pool
        tb = traceback.format_exception_only(type(exc), exc)
        msg = "".join(tb).strip()
        result = RunResult(
            run_id=config.run_id,
            config=config,
            shots_total=0,
            batches=0,
            per_batch_latency_ns=[],
            corrections_digest="",
            decoder_metadata={},
            started_at=started,
            finished_at=time.time(),
            error=f"worker_crashed:{pid}:{msg}",
        )
    _logger.info(
        "worker end",
        extra={"run_id": config.run_id, "pid": pid, "ok": result.ok},
    )
    return result


def _resolve_max_workers(requested: int | None, n_configs: int) -> int:
    if n_configs <= 0:
        return 1
    cpu = os.cpu_count() or 1
    default = min(cpu, n_configs)
    if requested is None:
        chosen = default
    else:
        if requested < 1:
            raise ValueError(f"max_workers must be >= 1; got {requested}")
        chosen = requested
    return min(chosen, MAX_WORKERS_CAP, n_configs)


def run_parallel(
    configs: Sequence[RunConfig],
    decoder_factory_spec: tuple[str, str],
    master_seed: int,
    max_workers: int | None = None,
    *,
    num_detectors: int = 32,
    batch_size: int = 1024,
) -> list[RunResult]:
    """Execute RunConfigs in a spawn-context process pool, in submission order.

    Args:
        configs: RunConfigs to execute. Empty input returns ``[]``.
        decoder_factory_spec: ``(module_path, callable_name)`` tuple. The
            worker imports this and calls it with the backend name to
            build a Decoder. Must be picklable.
        master_seed: Master seed for per-worker RNG derivation.
        max_workers: Pool size. Defaults to
            ``min(os.cpu_count(), len(configs))``. Clamped to
            :data:`MAX_WORKERS_CAP`.

    Returns:
        RunResults in the same order as ``configs``.
    """
    if not configs:
        return []

    workers = _resolve_max_workers(max_workers, len(configs))
    ctx = mp.get_context("spawn")
    args_iter = [
        (cfg, decoder_factory_spec, master_seed, num_detectors, batch_size)
        for cfg in configs
    ]

    results: list[RunResult] = []
    with ctx.Pool(processes=workers) as pool:
        for r in pool.imap(_worker_entry, args_iter):
            results.append(r)
    return results
