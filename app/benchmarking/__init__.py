"""Benchmarking layer — reproducible parameter sweeps over the decoder stack.

Module map:

    sweep.py           SweepSpec / NoiseSpec / SweepPoint (T032) — declare
                       what to run; deterministic expansion + canonical hash.
    orchestrator.py    expand_sweep / RunConfig / compute_run_id (T033) —
                       Cartesian product of axes into concrete RunConfigs
                       with stable run_id hashing.
    runner.py          RunResult / run_single / generate_syndromes (T034) —
                       execute a single RunConfig against a Decoder and
                       return a structured RunResult.
"""
