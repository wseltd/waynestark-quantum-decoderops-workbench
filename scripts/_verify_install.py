"""
Quantum DecoderOps Workbench - environment verifier.

Invoked by scripts/verify_install.sh. Produces:

  - human-readable report on stdout
  - machine-readable .decoderops/environment_report.json

The JSON schema is consumed at runtime by app/core/capability.py (built in
later later product tickets) to surface backend availability in API responses,
the compatibility matrix, and the deployment-readiness report.

Design goals:

  * No imports of Workbench modules. This runs BEFORE any product code is
    installed and must stay self-contained on the Python stdlib + whatever
    Tier 1 provided.
  * Never raise. Every probe is wrapped; any failure becomes a structured
    status entry with a precise reason string.
  * Distinguish three failure modes per probe:
        "ok"           - imported and functional
        "degraded"     - imported but a capability check failed
        "unavailable"  - import failed (package absent)
    Tier 3 packages may be "unavailable" without dropping the overall
    status below "ready" for CPU-only use, but the primary GPU stack
    (torch CUDA + onnxruntime CUDAExecutionProvider) must be "ok" for
    any run that wants to claim GPU capability.
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib
import json
import os
import platform
import shutil
import socket
import subprocess  # nosec B404 - used only with fixed argv lists, never shell=True, no user input
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / ".decoderops"
REPORT_PATH = STATE_DIR / "environment_report.json"
ISING_ASSETS_PATH = STATE_DIR / "ising_assets.json"
BOOTSTRAP_CORE_PATH = STATE_DIR / "bootstrap_core.json"
BOOTSTRAP_GPU_PATH = STATE_DIR / "bootstrap_gpu.json"


# --------------------------------------------------------------------------- #
# Probe results
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Probe:
    status: str  # "ok" | "degraded" | "unavailable"
    version: str | None = None
    details: dict[str, Any] = dataclasses.field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status}
        if self.version is not None:
            out["version"] = self.version
        if self.details:
            out["details"] = self.details
        if self.reason is not None:
            out["reason"] = self.reason
        return out


def _try_import(mod_name: str, version_attr: str = "__version__") -> Probe:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:
        return Probe(
            status="unavailable",
            reason=f"ImportError: {type(exc).__name__}: {exc}",
        )
    version = getattr(mod, version_attr, None)
    if version is None:
        version = "unknown"
    return Probe(status="ok", version=str(version))


# --------------------------------------------------------------------------- #
# Host / Python / venv / GPU facts
# --------------------------------------------------------------------------- #


def collect_host() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "os_release": platform.release(),
        "kernel": platform.version(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def collect_python() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    }


def collect_gpu() -> dict[str, Any]:
    out: dict[str, Any] = {"nvidia_smi_available": False, "devices": []}
    nv = shutil.which("nvidia-smi")
    if not nv:
        out["reason"] = "nvidia-smi not on PATH"
        return out
    out["nvidia_smi_available"] = True
    try:
        # argv is a fixed literal list resolved from `which nvidia-smi`;
        # no shell, no user input, no string interpolation.
        res = subprocess.run(  # nosec B603
            [
                nv,
                "--query-gpu=index,name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.SubprocessError as exc:
        out["reason"] = f"nvidia-smi invocation failed: {exc}"
        return out
    driver_versions: set[str] = set()
    devices: list[dict[str, Any]] = []
    for line in res.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        idx, name, driver, mem_mib, cc = parts
        driver_versions.add(driver)
        devices.append(
            {
                "index": int(idx),
                "name": name,
                "memory_mib": int(mem_mib),
                "compute_capability": cc,
            }
        )
    out["devices"] = devices
    out["driver_version"] = sorted(driver_versions)[0] if driver_versions else None
    if out["driver_version"]:
        try:
            major = int(out["driver_version"].split(".", 1)[0])
            out["cuda_13_floor_ok"] = major >= 580
            out["cuda_13_floor_required"] = 580
        except ValueError:
            out["cuda_13_floor_ok"] = None
    return out


# --------------------------------------------------------------------------- #
# Tier 1 probes
# --------------------------------------------------------------------------- #


TIER1_MODULES: list[str] = [
    "stim",
    "pymatching",
    "sinter",
    "numpy",
    "scipy",
    "torch",
    "pandas",
    "pyarrow",
    "onnx",
    "onnxruntime",
    "fastapi",
    "pydantic",
    "typer",
    "rich",
    "duckdb",
    "sqlalchemy",
    "psycopg",
    "alembic",
    "jinja2",
    "reportlab",
    "markdown",
    "structlog",
    "hydra",
    "omegaconf",
    "safetensors",
    "matplotlib",
    "ldpc",
    "beliefmatching",
    "pytest",
    "hypothesis",
    "ruff",
    "mypy",
]


def collect_tier1() -> dict[str, dict[str, Any]]:
    return {mod: _try_import(mod).to_dict() for mod in TIER1_MODULES}


# --------------------------------------------------------------------------- #
# Tier 3 probes (GPU + accelerated runtimes)
# --------------------------------------------------------------------------- #


def probe_torch_cuda() -> Probe:
    base = _try_import("torch")
    if base.status != "ok":
        return base
    try:
        import torch
    except Exception as exc:  # pragma: no cover - _try_import succeeded above
        return Probe(status="unavailable", reason=str(exc))
    details: dict[str, Any] = {
        "cuda_built": bool(getattr(torch.version, "cuda", None)),
        "cuda_runtime_version": getattr(torch.version, "cuda", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if details["cuda_available"]:
        devs = []
        for i in range(details["device_count"]):
            try:
                devs.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": list(torch.cuda.get_device_capability(i)),
                    }
                )
            except Exception as exc:
                devs.append({"index": i, "error": str(exc)})
        details["devices"] = devs
        # try a minimal compute op to confirm CUDA runtime is functional
        try:
            x = torch.ones(4, device="cuda")
            _ = (x + x).cpu()
            details["smoke_compute_ok"] = True
        except Exception as exc:
            details["smoke_compute_ok"] = False
            details["smoke_compute_error"] = f"{type(exc).__name__}: {exc}"
            return Probe(
                status="degraded",
                version=torch.__version__,
                details=details,
                reason="torch imports but CUDA compute smoke-op failed",
            )
        return Probe(status="ok", version=torch.__version__, details=details)
    if not details["cuda_built"]:
        return Probe(
            status="degraded",
            version=torch.__version__,
            details=details,
            reason="torch wheel has no CUDA build; reinstall from cu130 index",
        )
    return Probe(
        status="degraded",
        version=torch.__version__,
        details=details,
        reason="torch has CUDA build but no CUDA device visible at runtime",
    )


def probe_onnxruntime_gpu() -> Probe:
    base = _try_import("onnxruntime")
    if base.status != "ok":
        return base
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        return Probe(status="unavailable", reason=str(exc))
    providers = list(ort.get_available_providers())
    details: dict[str, Any] = {"providers": providers}
    if "CUDAExecutionProvider" in providers:
        return Probe(status="ok", version=ort.__version__, details=details)
    return Probe(
        status="degraded",
        version=ort.__version__,
        details=details,
        reason="CUDAExecutionProvider not registered; installed wheel may be CPU-only or wrong CUDA major",
    )


def probe_tensorrt() -> Probe:
    p = _try_import("tensorrt")
    if p.status != "ok":
        return p
    try:
        import tensorrt as trt  # type: ignore

        p.details["builder"] = trt.Builder(trt.Logger(trt.Logger.ERROR)).__class__.__name__
    except Exception as exc:
        p.status = "degraded"
        p.reason = f"TensorRT imports but Builder init failed: {type(exc).__name__}: {exc}"
    return p


def probe_cuquantum() -> Probe:
    return _try_import("cuquantum")


def probe_cudaq() -> Probe:
    p = _try_import("cudaq")
    if p.status == "ok":
        try:
            import cudaq  # type: ignore

            p.details["num_available_targets"] = len(cudaq.get_targets())
        except Exception as exc:
            p.status = "degraded"
            p.reason = f"cudaq imports but get_targets() failed: {type(exc).__name__}: {exc}"
    return p


def probe_cudaq_qec() -> Probe:
    p = _try_import("cudaq_qec")
    if p.status != "ok":
        alt = _try_import("cudaq.qec")
        if alt.status == "ok":
            return alt
    return p


def probe_modelopt() -> Probe:
    return _try_import("modelopt")


def collect_tier3() -> dict[str, dict[str, Any]]:
    return {
        "torch_cuda": probe_torch_cuda().to_dict(),
        "onnxruntime_gpu": probe_onnxruntime_gpu().to_dict(),
        "tensorrt": probe_tensorrt().to_dict(),
        "cuquantum": probe_cuquantum().to_dict(),
        "cudaq": probe_cudaq().to_dict(),
        "cudaq_qec": probe_cudaq_qec().to_dict(),
        "nvidia_modelopt": probe_modelopt().to_dict(),
    }


# --------------------------------------------------------------------------- #
# Ising asset inspection
# --------------------------------------------------------------------------- #


def collect_ising_assets() -> dict[str, Any]:
    if not ISING_ASSETS_PATH.exists():
        return {
            "status": "missing",
            "reason": f"{ISING_ASSETS_PATH} not present; run scripts/fetch_ising_assets.sh",
        }
    try:
        with ISING_ASSETS_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"failed to read {ISING_ASSETS_PATH}: {exc}",
        }
    # re-verify the file paths still exist (no re-hash; that's the fetcher's job)
    missing: list[str] = []
    for model in data.get("models", []):
        p = Path(model.get("abspath", ""))
        if not p.exists():
            missing.append(model.get("relpath") or str(p))
    if missing:
        data["runtime_check"] = {"status": "missing", "missing": missing}
    else:
        data["runtime_check"] = {"status": "present"}
    return data


# --------------------------------------------------------------------------- #
# Bootstrap records
# --------------------------------------------------------------------------- #


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "path": str(path)}


# --------------------------------------------------------------------------- #
# Overall status
# --------------------------------------------------------------------------- #


REQUIRED_TIER1 = {
    "stim",
    "pymatching",
    "numpy",
    "torch",
    "onnx",
    "onnxruntime",
    "fastapi",
    "pydantic",
    "typer",
    "duckdb",
    "sqlalchemy",
    "psycopg",
    "reportlab",
    "jinja2",
    "pytest",
}


def compute_overall(
    tier1: dict[str, dict[str, Any]],
    tier3: dict[str, dict[str, Any]],
    ising: dict[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []

    tier1_missing = [
        name
        for name in REQUIRED_TIER1
        if tier1.get(name, {}).get("status") != "ok"
    ]
    if tier1_missing:
        blockers.append(
            "Tier 1 required modules missing or failing import: "
            + ", ".join(sorted(tier1_missing))
        )

    if ising.get("status") not in ("ok",):
        blockers.append(
            f"Ising asset inventory status={ising.get('status')}; "
            f"message={ising.get('message') or ising.get('reason') or 'unknown'}"
        )
    elif ising.get("runtime_check", {}).get("status") == "missing":
        blockers.append(
            "Ising checkpoint files disappeared after asset fetch: "
            + ", ".join(ising["runtime_check"].get("missing", []))
        )

    torch_cuda = tier3.get("torch_cuda", {})
    ort_gpu = tier3.get("onnxruntime_gpu", {})
    gpu_primary_ok = (
        torch_cuda.get("status") == "ok"
        and ort_gpu.get("status") == "ok"
    )

    if blockers:
        return "missing_required", blockers

    if gpu_primary_ok:
        # still note any optional Tier 3 that is unavailable
        optional = ["tensorrt", "cuquantum", "cudaq", "cudaq_qec", "nvidia_modelopt"]
        missing_opt = [k for k in optional if tier3.get(k, {}).get("status") != "ok"]
        if missing_opt:
            # ready but degraded on optionals
            return "degraded", [
                f"Optional Tier 3 capability unavailable: {k} "
                f"({tier3[k].get('reason') or tier3[k].get('status')})"
                for k in missing_opt
            ]
        return "ready", []

    # Tier 1 and Ising OK but GPU primary not functional - degraded (CPU-only)
    gpu_reason = []
    if torch_cuda.get("status") != "ok":
        gpu_reason.append(
            f"torch_cuda={torch_cuda.get('status')} "
            f"({torch_cuda.get('reason') or torch_cuda.get('details', {}).get('cuda_runtime_version')})"
        )
    if ort_gpu.get("status") != "ok":
        gpu_reason.append(
            f"onnxruntime_gpu={ort_gpu.get('status')} "
            f"({ort_gpu.get('reason')})"
        )
    return "degraded", [
        "Primary GPU stack not fully operational (CPU baseline still usable): "
        + "; ".join(gpu_reason)
    ]


# --------------------------------------------------------------------------- #
# Human-readable report
# --------------------------------------------------------------------------- #


def print_human_report(report: dict[str, Any]) -> None:
    sep = "=" * 72
    print(sep)
    print("  Quantum DecoderOps Workbench - environment verification")
    print(sep)
    print(f"  overall status : {report['overall_status'].upper()}")
    print(f"  generated      : {report['generated_utc']}")
    print(f"  repo root      : {report['repo_root']}")
    print()

    py = report["python"]
    print("PYTHON")
    print(f"  interpreter : {py['executable']}")
    print(f"  version     : {py['version']} ({py['implementation']})")
    print(f"  in_venv     : {py['in_venv']}")
    print()

    gpu = report["gpu"]
    print("GPU")
    if not gpu.get("nvidia_smi_available"):
        print(f"  nvidia-smi unavailable ({gpu.get('reason', 'no reason recorded')})")
    else:
        print(f"  driver           : {gpu.get('driver_version')}")
        print(f"  cuda13 floor ok  : {gpu.get('cuda_13_floor_ok')}")
        for d in gpu.get("devices", []):
            print(
                f"  device[{d['index']}]       : {d['name']} "
                f"({d['memory_mib']} MiB, cc={d['compute_capability']})"
            )
    print()

    print("TIER 1 PACKAGES")
    for name, info in sorted(report["tier1"].items()):
        marker = "  ok " if info["status"] == "ok" else "FAIL"
        ver = info.get("version", "")
        reason = f"  ({info.get('reason')})" if info.get("reason") else ""
        print(f"  [{marker}] {name:<22} {ver}{reason}")
    print()

    print("TIER 3 CAPABILITIES")
    for name, info in report["tier3"].items():
        marker = {"ok": "  ok ", "degraded": "WARN", "unavailable": "MISS"}.get(
            info["status"], "????"
        )
        ver = info.get("version", "")
        reason = f" :: {info.get('reason')}" if info.get("reason") else ""
        print(f"  [{marker}] {name:<22} {ver}{reason}")
    print()

    ising = report["ising_assets"]
    print("ISING ASSETS")
    print(f"  status : {ising.get('status')}")
    vendor = ising.get("vendor", {})
    if vendor:
        print(f"  vendor : {vendor.get('url')}  @  {vendor.get('commit')}")
    for m in ising.get("models", []):
        print(
            f"  model  : {m.get('relpath')} "
            f"({m.get('size_bytes')} bytes, sha256={m.get('sha256','?')[:16]}...)"
        )
    print()

    if report["blockers"]:
        print("BLOCKERS")
        for b in report["blockers"]:
            print(f"  - {b}")
        print()

    print("ARTIFACTS")
    print(f"  environment report : {report['report_path']}")
    print(f"  bootstrap core log : {report['bootstrap_core'].get('log_file')}")
    print(f"  bootstrap gpu log  : {report['bootstrap_gpu'].get('log_file')}")
    print(f"  ising assets json  : {ISING_ASSETS_PATH}")
    print(sep)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    tier1 = collect_tier1()
    tier3 = collect_tier3()
    ising = collect_ising_assets()

    overall, blockers = compute_overall(tier1, tier3, ising)

    report = {
        "schema": "decoderops.environment_report.v1",
        "generated_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo_root": str(REPO_ROOT),
        "overall_status": overall,
        "blockers": blockers,
        "host": collect_host(),
        "python": collect_python(),
        "gpu": collect_gpu(),
        "tier1": tier1,
        "tier3": tier3,
        "ising_assets": ising,
        "bootstrap_core": read_json(BOOTSTRAP_CORE_PATH),
        "bootstrap_gpu": read_json(BOOTSTRAP_GPU_PATH),
        "report_path": str(REPORT_PATH),
    }

    with REPORT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print_human_report(report)

    if overall == "missing_required":
        return 1
    if overall == "degraded":
        return 0  # informational; script exits ok for CI purposes
    return 0


if __name__ == "__main__":
    sys.exit(main())
