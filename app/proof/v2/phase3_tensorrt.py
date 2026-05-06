"""Phase 3 — real TensorRT engine build + smoke inference, OR structured blocker.

Drives two real paths:

    A. Minimal-model path. Build a 1-op ONNX model in memory (Identity
       on float32[1,4]), parse it with tensorrt.OnnxParser, build a
       serialized engine via Builder.build_serialized_network(), then
       run one forward with tensorrt.Runtime + ExecutionContext on
       real GPU memory. Proves the stack can actually produce and
       execute a TRT engine.

    B. Vendor TRT path. Invoke vendor local_run.sh with
       ONNX_WORKFLOW=1 (export ONNX, keep PyTorch inference) using
       our app.benchmarking.ising_local_run_onnx wrapper, so the
       proof also covers ONNX export on the shipped Ising model.

Either real success writes the artefacts and SHA256s into the proof
record; a structured blocker matrix is written when anything below
genuinely cannot proceed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
PROOF = ROOT / ".decoderops" / "proof" / "v2" / "phase3"
PROOF.mkdir(parents=True, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

result: dict[str, Any] = {"phase": 3, "chain": []}


def _blocker(step: str, *, missing: str, requirement: str, install: str, probe: str) -> dict:
    return {
        "step": step,
        "status": "environment_blocked",
        "missing_dependency": missing,
        "runtime_requirement": requirement,
        "command_path_to_install": install,
        "environment_probe_output": probe,
    }


# ---------------------------------------------------------------------------
# A. Minimal-model real engine build + smoke inference
# ---------------------------------------------------------------------------

try:
    import numpy as np
    import tensorrt as trt
except ImportError as e:
    result["chain"].append(
        _blocker(
            "A.imports",
            missing=f"tensorrt/numpy import failed: {e}",
            requirement="tensorrt-cu13>=10.16 + numpy in .venv",
            install="pip install tensorrt-cu13==10.16.1.11 numpy",
            probe=repr(e),
        )
    )
    tensorrt_available = False
else:
    result["chain"].append(
        {"step": "A.imports", "status": "ok", "tensorrt_version": trt.__version__}
    )
    tensorrt_available = True


if tensorrt_available:
    # 1. Build a tiny ONNX model in memory via onnx.helper.
    try:
        import onnx
        from onnx import helper, TensorProto

        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])],
            "id",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)]
        )
        model.ir_version = 8
        onnx_bytes = model.SerializeToString()
        onnx_path = PROOF / "tiny_identity.onnx"
        onnx_path.write_bytes(onnx_bytes)
        result["chain"].append(
            {
                "step": "A.onnx_build",
                "status": "ok",
                "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
                "onnx_bytes": len(onnx_bytes),
            }
        )
    except Exception as e:  # noqa: BLE001
        result["chain"].append(
            _blocker(
                "A.onnx_build",
                missing=f"onnx model builder failed: {e}",
                requirement="onnx in .venv",
                install="pip install onnx",
                probe=repr(e),
            )
        )
        tensorrt_available = False


if tensorrt_available:
    # 2. Parse ONNX with TensorRT and build a serialized engine.
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        t0 = time.perf_counter()
        ok = parser.parse(onnx_bytes)
        assert ok, "\n".join(
            parser.get_error(i).desc() for i in range(parser.num_errors)
        )
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, 1 << 28
        )
        serialized = builder.build_serialized_network(network, config)
        build_seconds = time.perf_counter() - t0
        engine_bytes = bytes(serialized)
        engine_path = PROOF / "tiny_identity.engine"
        engine_path.write_bytes(engine_bytes)
        result["chain"].append(
            {
                "step": "A.trt_engine_build",
                "status": "ok",
                "build_seconds": round(build_seconds, 3),
                "engine_bytes": len(engine_bytes),
                "engine_sha256": hashlib.sha256(engine_bytes).hexdigest(),
            }
        )
    except Exception as e:  # noqa: BLE001
        result["chain"].append(
            _blocker(
                "A.trt_engine_build",
                missing=f"TensorRT build failed: {e}",
                requirement="Working NVIDIA driver + CUDA 13 + tensorrt-cu13",
                install="nvidia-smi must report a driver; tensorrt pip wheel present",
                probe=repr(e),
            )
        )
        tensorrt_available = False


if tensorrt_available:
    # 3. Deserialize and run one real inference on GPU.
    try:
        import torch  # already installed

        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        assert engine is not None, "deserialize_cuda_engine returned None"
        ctx = engine.create_execution_context()

        x_host = np.arange(4, dtype=np.float32).reshape(1, 4)
        x_gpu = torch.from_numpy(x_host).cuda()
        y_gpu = torch.empty_like(x_gpu)

        # Find IO names for this engine (TRT 10 API).
        io_bindings = {}
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            io_bindings[name] = None
        ctx.set_tensor_address("x", x_gpu.data_ptr())
        ctx.set_tensor_address("y", y_gpu.data_ptr())
        ctx.execute_async_v3(stream_handle=torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()

        y_host = y_gpu.cpu().numpy()
        identity_matches = bool(np.allclose(x_host, y_host))
        result["chain"].append(
            {
                "step": "A.trt_engine_execute",
                "status": "ok",
                "identity_round_trip": identity_matches,
                "output_sample": y_host.tolist(),
            }
        )
    except Exception as e:  # noqa: BLE001
        result["chain"].append(
            _blocker(
                "A.trt_engine_execute",
                missing=f"TensorRT execute failed: {e}",
                requirement="tensorrt runtime + CUDA device accessible",
                install="Confirm tensorrt, torch+cuda match driver via nvidia-smi",
                probe=repr(e),
            )
        )


# ---------------------------------------------------------------------------
# B. Vendor ONNX_WORKFLOW=1 via our wrapper
# ---------------------------------------------------------------------------

try:
    from app.benchmarking.ising_local_run_onnx import run_onnx_export
    from app.benchmarking.ising_subprocess import IsingSubprocessError

    VENDOR = ROOT / "vendor" / "Ising-Decoding"
    os.environ["PREDECODER_PYTHON"] = str(ROOT / ".venv" / "bin" / "python")
    os.environ["PREDECODER_MODEL_CHECKPOINT_FILE"] = str(
        VENDOR / "models" / "Ising-Decoder-SurfaceCode-1-Fast.pt"
    )
    os.environ["EXPERIMENT_NAME"] = "decoderops_proof_v2"
    os.environ["TORCH_COMPILE"] = "0"
    os.environ["PREDECODER_TORCH_COMPILE"] = "0"

    # Checkpoint must already be staged from Phase 2, but stage again
    # here if the phase was run standalone.
    fast_pt = VENDOR / "models" / "Ising-Decoder-SurfaceCode-1-Fast.pt"
    staged_dir = VENDOR / "outputs" / os.environ["EXPERIMENT_NAME"] / "models"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged = staged_dir / fast_pt.name
    if not staged.exists():
        try:
            staged.symlink_to(fast_pt)
        except OSError:
            import shutil
            shutil.copy2(fast_pt, staged)

    vendor_onnx_dir = PROOF / "vendor_onnx_export"
    vendor_onnx_dir.mkdir(exist_ok=True)

    try:
        rec = run_onnx_export(
            vendor_root=VENDOR,
            work_dir=vendor_onnx_dir,
            onnx_workflow=1,  # export ONNX, keep PyTorch inference
            quant_format="int8",  # ignored by workflow=1; wrapper requires a value
            model_variant="fast",
            cuda_visible_devices="0",
            timeout_seconds=600,
        )
        # Filter to just the real .onnx outputs the vendor emitted during this run
        emitted_onnx = [str(p) for p in rec.exported_onnx_paths]
        result["chain"].append(
            {
                "step": "B.vendor_onnx_workflow_1",
                "status": "ok" if rec.export_success else "partial",
                "returncode": rec.returncode,
                "duration_seconds": round(rec.duration_seconds, 2),
                "exported_onnx_files": emitted_onnx,
                "sha256_by_path": rec.sha256_by_path,
                "stdout_path": str(rec.stdout_path),
            }
        )
    except IsingSubprocessError as e:
        # Vendor run failed — capture stderr + last stdout.
        stderr_tail = (
            (vendor_onnx_dir / "stderr.log").read_text(errors="replace")[-1500:]
            if (vendor_onnx_dir / "stderr.log").exists()
            else ""
        )
        stdout_tail = (
            (vendor_onnx_dir / "stdout.log").read_text(errors="replace")[-1500:]
            if (vendor_onnx_dir / "stdout.log").exists()
            else ""
        )
        result["chain"].append(
            {
                "step": "B.vendor_onnx_workflow_1",
                "status": "failed",
                "error": str(e)[:500],
                "stderr_tail": stderr_tail,
                "stdout_tail": stdout_tail,
            }
        )
except Exception as e:  # noqa: BLE001
    result["chain"].append(
        {"step": "B.vendor_onnx_workflow_1.setup_failed", "error": repr(e)[:500]}
    )


(PROOF / "result.json").write_text(json.dumps(result, indent=2, default=str))
print(json.dumps(result, indent=2, default=str)[:4000])
print("PHASE3_DONE")
