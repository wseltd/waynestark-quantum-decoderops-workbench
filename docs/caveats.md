# Caveats and Known Limitations

## Tier 3 dependencies are opt-in
TensorRT, cudaq, cudaq-qec, cuQuantum, and nvidia-modelopt are detected
via `CapabilityReport` and are NOT installed by the workbench. If a
backend is unavailable, the deployment-readiness report emits a precise
blocker row. This is an intentional honest-unavailability posture.

## Decoder accuracy on synthetic syndromes
The benchmark runner's synthetic sampler produces uniform random bits
for plumbing tests only. Decoder accuracy on synthetic input is
undefined and MUST NOT be reported as a performance signal. Production
runs use a Stim-backed sampler via `stim.Circuit.compile_detector_sampler`.

## Fake stubs in `fixtures/fake_models/`
`tiny_ising_fast_stub.pt` is labelled and sentinel-gated. It exists to
exercise adapter plumbing only. Never use it for reported benchmark
numbers; real benchmarks use `vendor/Ising-Decoding/` checkpoints.

## PostgreSQL is Alembic-only
DuckDB uses `Base.metadata.create_all` directly. PostgreSQL REQUIRES
Alembic-managed migrations (`alembic upgrade head`).

## Determinism budget
Byte-reproducible tarballs use a fixed mtime (2024-01-01 UTC). PDF
determinism requires a frozen timestamp and the reportlab default creation
date is explicitly nulled.

## Multi-GPU
Default is device 0 only. Device 1 (display GPU) requires explicit
operator consent before targeting GPU device 1 (the display GPU).
