# Fake Models — Test Stubs ONLY

These files are NOT real NVIDIA Ising-Decoding checkpoints. They are
test-only stubs used to exercise loader and adapter plumbing without
touching the real weights.

**Rules:**
- MUST NOT be used to produce reported benchmark numbers
- MUST NOT be packaged into customer artefacts
- Real benchmarks MUST use vendor/Ising-Decoding checkpoints
- The stub is clearly labelled with `_FAKE_STUB: True` and a `not_real`
  sentinel so adapters can assert and reject when it's loaded in
  production paths.
