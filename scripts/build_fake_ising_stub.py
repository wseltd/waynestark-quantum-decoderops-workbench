"""Build the fake ising_fast stub (T118) — NOT a real NVIDIA checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch


def build_stub(output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    linear = torch.nn.Linear(4, 4)
    payload = {
        "_FAKE_STUB": True,
        "not_real": (
            "this is a unit-test stub, not an NVIDIA Ising-Decoding "
            "checkpoint"
        ),
        "receptive_field": 9,
        "arch": "ising_fast_stub",
        "state_dict": linear.state_dict(),
        "metadata": {
            "source": "fixtures/fake_models",
            "license": "test-only",
            "created_by": "T118",
        },
    }
    torch.save(payload, output_path)
    return output_path


if __name__ == "__main__":
    build_stub(Path("fixtures/fake_models/tiny_ising_fast_stub.pt"))
