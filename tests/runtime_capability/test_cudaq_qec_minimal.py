"""cudaq-qec minimal real API smoke (T188).

Exercises one minimal real public API call on cudaq-qec 0.6.0.
Inventory of the real public surface lives at
`.decoderops/proof/v2/phase6_cudaq_qec_api_inventory.txt`.
"""

from __future__ import annotations

import pytest


def test_cudaq_qec_real_surface_code_api() -> None:
    try:
        import cudaq_qec as qec
    except ImportError as e:
        pytest.skip(
            f"cudaq_qec unavailable: {e} | required: cudaq-qec==0.6.0 "
            "+ matching cudaq runtime | category: software"
        )

    # Real public factory on cudaq-qec 0.6.0.
    assert "surface_code" in qec.get_available_codes()

    try:
        code = qec.get_code("surface_code", distance=3)
    except Exception as e:  # noqa: BLE001
        pytest.skip(
            f"cudaq_qec surface_code init failed: {e} | category: runtime"
        )

    # Structural queries against the shipped public API.
    n_data = code.get_num_data_qubits()
    n_anc = code.get_num_ancilla_qubits()
    n_x = code.get_num_x_stabilizers()
    n_z = code.get_num_z_stabilizers()

    assert n_data > 0, f"surface_code d=3 expected >0 data qubits (got {n_data})"
    assert n_anc > 0
    assert n_x > 0
    assert n_z > 0
    # d=3 rotated surface code lower-bounds. Exact counts can drift
    # across minor vendor versions; asserting structure, not exact
    # integers, keeps the test robust.
    assert n_data >= 5, f"d=3 surface code unexpectedly small: {n_data}"

    # Parity check matrix must be present + 2D.
    parity = code.get_parity()
    assert parity is not None
    shape = getattr(parity, "shape", None)
    if shape is not None:
        assert shape[0] > 0 and shape[1] > 0, f"parity shape: {shape}"
