#!/usr/bin/env bash
#
# scripts/bootstrap_gpu_runtime_env.sh
# ---------------------------------------------------------------------------
# Quantum DecoderOps Workbench - Tier 3 GPU runtime bootstrap (CUDA 13.x).
#
# PURPOSE
#   Install the GPU-accelerated runtime stack inside the existing project
#   .venv/, targeting NVIDIA Blackwell / Hopper / Ada with CUDA 13.x.
#   Each Tier 3 package (torch-cu13, onnxruntime-gpu, tensorrt-cu13,
#   cuquantum-python-cu13, cudaq, cudaq-qec, nvidia-modelopt) is a real
#   product capability: installed when feasible, detected and reported
#   precisely when not.
#
# PRECONDITIONS
#   - scripts/bootstrap_core_env.sh must have completed successfully.
#   - .venv/ exists and contains the Tier 1 stack.
#   - nvidia-smi must be on PATH; the NVIDIA kernel driver must be
#     recent enough to expose CUDA 13.0 user-space libraries.
#
# DESIGN
#   This script never hard-fails on a single Tier 3 package failure
#   because the product must support partial Tier 3 availability
#   (e.g. a customer site with GPU but no TensorRT licence).
#   Instead, each package install is wrapped in a per-package function
#   that records success / failure / reason into a status dict, and the
#   script exits non-zero only if the PRIMARY GPU stack (torch-cu13 +
#   onnxruntime-gpu) both fail. Every per-package result is persisted
#   to .decoderops/bootstrap_gpu.json and surfaced by verify_install.sh.
#
# SAFETY
#   - Everything is installed into .venv/. No system packages, no sudo.
#   - CUDA runtime libs are shipped inside pip wheels; no system CUDA
#     toolkit is installed or required.
#   - No writes outside the project directory.
#
# ARTIFACTS
#   .decoderops/bootstrap_gpu.json         machine-readable per-package status
#   .decoderops/logs/bootstrap_gpu_<stamp>.log   full install log
#   requirements/gpu-cu13.lock.txt         pip freeze of actually-installed Tier 3
#
# EXIT CODES
#   0  primary GPU stack (torch-cu13 + onnxruntime-gpu) imports successfully
#   1  infrastructure error (no .venv, no nvidia-smi)
#   2  driver is older than the CUDA-13 floor (580.x)
#   3  primary GPU stack (torch-cu13 + onnxruntime-gpu) both failed to import
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME="bootstrap_gpu_runtime_env"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/.decoderops/logs"
STATE_DIR="${REPO_ROOT}/.decoderops"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${STAMP}.log"
STATUS_FILE="${STATE_DIR}/bootstrap_gpu.json"
REQ_FILE="${REPO_ROOT}/requirements/gpu-cu13.txt"
LOCK_FILE="${REPO_ROOT}/requirements/gpu-cu13.lock.txt"
VENV_DIR="${REPO_ROOT}/.venv"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

log()   { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}"; }
info()  { log "INFO  $*"; }
warn()  { log "WARN  $*"; }
step()  { log "STEP  $*"; }
banner() {
  local line="================================================================"
  log "${line}"; log "  $1"; log "${line}"
}

# results dict (simple parallel arrays; assembled to JSON at the end)
declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_VERSION=()
declare -a RESULT_MESSAGE=()

record_result() {
  # record_result <pkg> <ok|fail> <version_or_empty> <message>
  RESULT_NAMES+=("$1")
  RESULT_STATUS+=("$2")
  RESULT_VERSION+=("$3")
  RESULT_MESSAGE+=("$4")
}

write_status_json() {
  local overall_status="$1" overall_msg="$2"
  python3 - "${STATUS_FILE}" "${overall_status}" "${overall_msg}" "${STAMP}" "${LOG_FILE}" \
    "${DRIVER_VERSION:-unknown}" "${GPU_SUMMARY:-unknown}" "${#RESULT_NAMES[@]}" \
    "${RESULT_NAMES[@]:-}" "${RESULT_STATUS[@]:-}" "${RESULT_VERSION[@]:-}" "${RESULT_MESSAGE[@]:-}" \
    <<'PY' || true
import json, sys, os, datetime
path = sys.argv[1]
overall_status, overall_msg, stamp, log_file, driver, gpu_summary = sys.argv[2:8]
n = int(sys.argv[8])
rest = sys.argv[9:]
names    = rest[0:n]
statuses = rest[n:2*n]
versions = rest[2*n:3*n]
messages = rest[3*n:4*n]
packages = []
for i in range(n):
    packages.append({
        "name": names[i],
        "status": statuses[i],
        "version": versions[i] or None,
        "message": messages[i],
    })
payload = {
    "schema": "decoderops.bootstrap.gpu.v1",
    "script": "scripts/bootstrap_gpu_runtime_env.sh",
    "run_stamp_utc": stamp,
    "finished_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "overall_status": overall_status,
    "overall_message": overall_msg,
    "log_file": log_file,
    "nvidia_driver_version": driver,
    "gpu_summary": gpu_summary,
    "packages": packages,
}
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

fatal() {
  warn "FATAL $1"
  write_status_json "failed" "$1"
  exit "${2:-1}"
}

# -- phase 0: preflight ----------------------------------------------------

banner "Quantum DecoderOps Workbench - Tier 3 GPU runtime bootstrap (cu13)"
info "repo root   = ${REPO_ROOT}"
info "log file    = ${LOG_FILE}"
info "status file = ${STATUS_FILE}"

[[ -d "${VENV_DIR}" ]] || fatal "venv not found at ${VENV_DIR}; run scripts/bootstrap_core_env.sh first" 1
[[ -f "${REQ_FILE}"  ]] || fatal "missing ${REQ_FILE}" 1
command -v nvidia-smi >/dev/null 2>&1 || fatal "nvidia-smi not on PATH; NVIDIA driver not installed?" 1

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

info "venv python = $(${VENV_PY} -c 'import sys;print(sys.executable)')"
info "venv version = $(${VENV_PY} -c 'import sys;print(sys.version.split()[0])')"

# -- phase 1: GPU + driver detection --------------------------------------

banner "Phase 1 / 4 : detecting NVIDIA GPU and driver"

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
GPU_SUMMARY="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | paste -sd '|' -)"
info "driver version = ${DRIVER_VERSION}"
info "GPUs detected  = ${GPU_SUMMARY}"

DRIVER_MAJOR="${DRIVER_VERSION%%.*}"
if (( DRIVER_MAJOR < 580 )); then
  fatal "driver ${DRIVER_VERSION} is older than 580.x (CUDA 13 requires 580+); upgrade the NVIDIA driver" 2
fi
info "driver ${DRIVER_VERSION} meets CUDA 13 floor (>=580)"

# Detect Blackwell specifically so we can warn about the cu12 stack.
if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "blackwell\|RTX PRO\|B200\|GB200\|GB300"; then
  info "Blackwell-class GPU detected - cu13 wheels REQUIRED (cu12 stack known to have issues per research pack)"
fi

# -- phase 2: install Tier 3 packages individually ------------------------

banner "Phase 2 / 4 : installing Tier 3 packages (per-package logging)"

# helper: install one pip package with its own log line and record outcome.
# Extra args after $2 are passed through to pip as individual argv tokens
# (no IFS-dependent splitting) so flags like --index-url work correctly.
install_pkg() {
  local pkg_name="$1" pkg_spec="$2"
  shift 2
  local -a extra_args=("$@")
  step "installing ${pkg_name} (${pkg_spec})${extra_args:+ ${extra_args[*]}}"
  if "${VENV_PIP}" install "${extra_args[@]}" "${pkg_spec}" 2>&1 | tee -a "${LOG_FILE}"; then
    info "  [install-ok] ${pkg_name}"
    return 0
  else
    warn "  [install-fail] ${pkg_name}"
    return 1
  fi
}

# 2.1 torch on cu13
# Tier 1 already installed torch==2.11.0+cu130 from the PyPI default index
# on Linux x86_64 (PyTorch now ships cu13 as the default Linux wheel), so
# no --index-url is needed here. If Tier 1 left a CPU wheel in place (e.g.
# on a future Python/platform combo where the default is CPU), the plain
# `pip install torch --upgrade` below will still pull the latest available.
step "verifying torch is already the +cu130 build from Tier 1"
TORCH_TAG="$(${VENV_PY} -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"
info "  current torch build = ${TORCH_TAG}"
if [[ "${TORCH_TAG}" == *"+cu"* ]]; then
  info "  keeping existing cu-build torch (no reinstall)"
  TORCH_OK=1
else
  step "torch is CPU or missing; installing cu13 wheel"
  if install_pkg "torch-cu13" "torch" --upgrade; then
    TORCH_OK=1
  else
    TORCH_OK=0
  fi
fi

# 2.2 onnxruntime-gpu
# Research pack: "ONNX Runtime currently exposes a nightly CUDA 13.x path".
# We attempt the stable wheel first; if the resulting wheel is cu12 or fails
# on a CUDA init smoke test, the verify step will flag it as degraded.
if install_pkg "onnxruntime-gpu" "onnxruntime-gpu"; then
  ORT_OK=1
else
  ORT_OK=0
fi

# 2.3 tensorrt-cu13
if install_pkg "tensorrt-cu13" "tensorrt-cu13"; then
  TRT_OK=1
else
  TRT_OK=0
fi

# 2.4 nvidia-modelopt[onnx]   (required by Ising-Decoding quantisation path; py<3.13)
PY_MAJOR_MINOR="$(${VENV_PY} -c 'import sys;print("%d.%d" % sys.version_info[:2])')"
if [[ "${PY_MAJOR_MINOR}" == "3.13" ]]; then
  warn "skipping nvidia-modelopt[onnx]: upstream marker requires python<3.13 (current ${PY_MAJOR_MINOR})"
  record_result "nvidia-modelopt" "skipped" "" "python ${PY_MAJOR_MINOR} excluded by upstream marker python_version<3.13"
else
  if install_pkg "nvidia-modelopt" "nvidia-modelopt[onnx]"; then
    MODELOPT_OK=1
  else
    MODELOPT_OK=0
  fi
fi

# 2.5 cuquantum-python-cu13
if install_pkg "cuquantum-python-cu13" "cuquantum-python-cu13"; then
  CUQUANTUM_OK=1
else
  CUQUANTUM_OK=0
fi

# 2.6 cudaq
if install_pkg "cudaq" "cudaq"; then
  CUDAQ_OK=1
else
  CUDAQ_OK=0
fi

# 2.7 cudaq-qec
if install_pkg "cudaq-qec" "cudaq-qec"; then
  CUDAQQEC_OK=1
else
  CUDAQQEC_OK=0
fi

# 2.8 cupy cross-version cleanup
# Some transitive dependencies (cudaq meta-package variants, older modelopt
# chains) pull cupy-cuda12x while cuquantum-python-cu13 and cuda-quantum-cu13
# pull cupy-cuda13x. On Blackwell (CUDA 13 only) the cu12 wheel is dead
# weight and triggers a noisy UserWarning on every import that can break
# downstream capability probes. Keep only cupy-cuda13x.
step "cupy cleanup: ensuring only cupy-cuda13x remains (Blackwell requires cu13)"
if "${VENV_PIP}" show cupy-cuda12x >/dev/null 2>&1; then
  info "  cupy-cuda12x detected; uninstalling"
  "${VENV_PIP}" uninstall -y cupy-cuda12x 2>&1 | tee -a "${LOG_FILE}" || \
    warn "  cupy-cuda12x uninstall reported non-zero (continuing)"
else
  info "  cupy-cuda12x not present - ok"
fi
if "${VENV_PIP}" show cupy-cuda13x >/dev/null 2>&1; then
  info "  cupy-cuda13x present - ok"
  # Cross-version uninstall can leave cupy-cuda13x with a corrupted package
  # namespace where `import cupy` returns an empty module (no ndarray, no
  # __version__). Force-reinstall to restore a clean state AND pin the
  # version to 13.6.x to satisfy cuda-quantum-cu13's constraint
  # (cupy-cuda13x~=13.6.0). Without the pin, pip floats to 14.x which is
  # numerically newer and imports fine today but violates cudaq's stated
  # requirement.
  #
  # Name convention reminder: "cuda13x" in the wheel name is the CUDA
  # major version (CUDA 13). "13.6.0" is CuPy's own library version.
  # Two unrelated numbering axes - see requirements/gpu-cu13.txt.
  step "force-reinstalling cupy-cuda13x==13.6.* (CUDA 13 wheel, cupy v13.6.x)"
  if "${VENV_PIP}" install --force-reinstall --no-deps 'cupy-cuda13x>=13.6.0,<13.7.0' 2>&1 | tee -a "${LOG_FILE}"; then
    info "  cupy-cuda13x force-reinstall ok"
  else
    warn "  cupy-cuda13x force-reinstall failed; cuquantum import may still break"
  fi
else
  warn "  cupy-cuda13x NOT present; cuquantum/cudaq GPU paths will fail"
fi

# -- phase 3: per-package import + CUDA-init smoke tests ------------------

banner "Phase 3 / 4 : import and CUDA-init smoke tests"

probe_torch_cu13() {
  "${VENV_PY}" -W ignore - <<'PY' 2>&1
import sys, json
try:
    import torch
    out = {
        "version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_built": bool(getattr(torch.version, "cuda", None)),
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    print("OK " + json.dumps(out))
    sys.exit(0)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    sys.exit(1)
PY
}

probe_onnxruntime_gpu() {
  "${VENV_PY}" -W ignore - <<'PY' 2>&1
import json, sys
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    out = {"version": ort.__version__, "providers": providers}
    if "CUDAExecutionProvider" not in providers:
        print(f"FAIL CUDAExecutionProvider missing; providers={providers}")
        sys.exit(1)
    print("OK " + json.dumps(out))
    sys.exit(0)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    sys.exit(1)
PY
}

probe_simple_import() {
  # $1 = module name, $2 = dotted path to version attribute (optional, default __version__)
  local mod="$1" ver_attr="${2:-__version__}"
  "${VENV_PY}" -W ignore - "${mod}" "${ver_attr}" <<'PY' 2>&1
import importlib, sys
mod, ver_attr = sys.argv[1], sys.argv[2]
try:
    m = importlib.import_module(mod)
    ver = getattr(m, ver_attr, "unknown")
    print(f"OK version={ver}")
    sys.exit(0)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    sys.exit(1)
PY
}

# IMPORTANT: every probe block uses the `if out="$(probe_...)"; then`
# pattern. Under `set -e`, that is the only form that lets a failed
# probe fall into the else-branch instead of aborting the whole script.
# Exit code from the probe heredoc is authoritative; stdout is just for
# logging.

# torch
if [[ "${TORCH_OK:-0}" == "1" ]]; then
  if out="$(probe_torch_cu13)"; then
    info "  [probe-ok]   torch-cu13 ${out#OK }"
    record_result "torch-cu13" "ok" "$(${VENV_PY} -c 'import torch;print(torch.__version__)')" "${out#OK }"
  else
    warn "  [probe-fail] torch-cu13 ${out}"
    record_result "torch-cu13" "fail" "" "${out#FAIL }"
    TORCH_OK=0
  fi
else
  record_result "torch-cu13" "fail" "" "pip install failed; see log"
fi

# onnxruntime-gpu
if [[ "${ORT_OK:-0}" == "1" ]]; then
  if out="$(probe_onnxruntime_gpu)"; then
    info "  [probe-ok]   onnxruntime-gpu ${out#OK }"
    record_result "onnxruntime-gpu" "ok" "$(${VENV_PY} -c 'import onnxruntime;print(onnxruntime.__version__)')" "${out#OK }"
  else
    warn "  [probe-fail] onnxruntime-gpu ${out}"
    record_result "onnxruntime-gpu" "degraded" "" "${out#FAIL }"
    ORT_OK=0
  fi
else
  record_result "onnxruntime-gpu" "fail" "" "pip install failed; see log"
fi

# tensorrt
if [[ "${TRT_OK:-0}" == "1" ]]; then
  if out="$(probe_simple_import tensorrt)"; then
    info "  [probe-ok]   tensorrt ${out}"
    record_result "tensorrt-cu13" "ok" "$(${VENV_PY} -c 'import tensorrt;print(tensorrt.__version__)')" "${out}"
  else
    warn "  [probe-fail] tensorrt ${out}"
    record_result "tensorrt-cu13" "fail" "" "${out#FAIL }"
  fi
else
  record_result "tensorrt-cu13" "fail" "" "pip install failed; see log"
fi

# modelopt
if [[ "${PY_MAJOR_MINOR}" != "3.13" ]]; then
  if [[ "${MODELOPT_OK:-0}" == "1" ]]; then
    if out="$(probe_simple_import modelopt __version__)"; then
      info "  [probe-ok]   nvidia-modelopt ${out}"
      record_result "nvidia-modelopt" "ok" "$(${VENV_PY} -c 'import modelopt;print(modelopt.__version__)')" "${out}"
    else
      warn "  [probe-fail] nvidia-modelopt ${out}"
      record_result "nvidia-modelopt" "fail" "" "${out#FAIL }"
    fi
  else
    record_result "nvidia-modelopt" "fail" "" "pip install failed; see log"
  fi
fi

# cuquantum-python
if [[ "${CUQUANTUM_OK:-0}" == "1" ]]; then
  if out="$(probe_simple_import cuquantum __version__)"; then
    info "  [probe-ok]   cuquantum-python ${out}"
    record_result "cuquantum-python-cu13" "ok" "$(${VENV_PY} -c 'import cuquantum;print(cuquantum.__version__)')" "${out}"
  else
    warn "  [probe-fail] cuquantum-python ${out}"
    record_result "cuquantum-python-cu13" "fail" "" "${out#FAIL }"
  fi
else
  record_result "cuquantum-python-cu13" "fail" "" "pip install failed; see log"
fi

# cudaq
if [[ "${CUDAQ_OK:-0}" == "1" ]]; then
  if out="$(probe_simple_import cudaq __version__)"; then
    info "  [probe-ok]   cudaq ${out}"
    record_result "cudaq" "ok" "$(${VENV_PY} -c 'import cudaq;print(getattr(cudaq,"__version__","unknown"))')" "${out}"
  else
    warn "  [probe-fail] cudaq ${out}"
    record_result "cudaq" "fail" "" "${out#FAIL }"
  fi
else
  record_result "cudaq" "fail" "" "pip install failed; see log"
fi

# cudaq-qec: upstream name varies between cudaq_qec and cudaq.qec
if [[ "${CUDAQQEC_OK:-0}" == "1" ]]; then
  if out="$(probe_simple_import cudaq_qec __version__)" || out="$(probe_simple_import cudaq.qec __version__)"; then
    info "  [probe-ok]   cudaq-qec ${out}"
    record_result "cudaq-qec" "ok" "unknown" "${out}"
  else
    warn "  [probe-fail] cudaq-qec ${out}"
    record_result "cudaq-qec" "fail" "" "${out#FAIL }"
  fi
else
  record_result "cudaq-qec" "fail" "" "pip install failed; see log"
fi

# -- phase 4: lock file + overall status ----------------------------------

banner "Phase 4 / 4 : writing lock file and status record"

"${VENV_PIP}" freeze --exclude-editable > "${LOCK_FILE}"
info "lock file has $(wc -l < "${LOCK_FILE}") pinned packages"

# Primary stack = torch-cu13 AND onnxruntime-gpu
if [[ "${TORCH_OK:-0}" == "1" && "${ORT_OK:-0}" == "1" ]]; then
  overall_status="ok"
  overall_code=0
  overall_msg="Primary GPU stack operational (torch-cu13 + onnxruntime-gpu); some optional Tier 3 packages may still be unavailable - see packages[]"
else
  overall_status="failed"
  overall_code=3
  overall_msg="Primary GPU stack broken: torch-cu13=$([[ "${TORCH_OK:-0}" == "1" ]] && echo ok || echo fail), onnxruntime-gpu=$([[ "${ORT_OK:-0}" == "1" ]] && echo ok || echo fail)"
fi

write_status_json "${overall_status}" "${overall_msg}"

banner "Tier 3 GPU bootstrap finished: ${overall_status}"
info "per-package results:"
for i in "${!RESULT_NAMES[@]}"; do
  info "  ${RESULT_NAMES[$i]} = ${RESULT_STATUS[$i]}  (${RESULT_VERSION[$i]})"
done
info ""
info "status JSON: ${STATUS_FILE}"
info "log file   : ${LOG_FILE}"

if [[ "${overall_code}" != "0" ]]; then
  warn "primary GPU stack not fully operational - review ${STATUS_FILE}"
fi

exit "${overall_code}"
