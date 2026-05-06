# Licensing

## This project
- Source code: Apache-2.0 (tentative; pending operator review).

## Third-party dependencies
- **Python packages**: see `requirements/core.lock.txt` and
  `requirements/gpu-cu13.lock.txt` for exact versions and hashes. Each
  package's license is honoured per its PyPI metadata.
- **NVIDIA Ising-Decoding checkpoints** (`vendor/Ising-Decoding/models/*.pt`):
  Apache-2.0. Redistributable provided the Apache-2.0 NOTICE is preserved.

## Redistribution rules
- Tarballs produced by `app.packaging.tarball.build_tarball` include the
  manifest + artefacts. They MUST NOT embed proprietary NVIDIA binaries
  beyond the Apache-2.0 Ising-Decoding release.

## Tier-3 runtimes (customer-installed, NOT packaged)
- TensorRT, cudaq, cudaq-qec, cuQuantum, nvidia-modelopt — customer
  licensing applies. This workbench detects and consumes them via
  `CapabilityReport`; it never ships them.

## References
- `research-and-resourced.md` — full licensing matrix.
