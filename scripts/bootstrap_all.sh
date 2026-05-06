#!/usr/bin/env bash
#
# scripts/bootstrap_all.sh
# ---------------------------------------------------------------------------
# Quantum DecoderOps Workbench - orchestrated end-to-end setup.
#
# RUNS (in order):
#   1. scripts/bootstrap_core_env.sh
#   2. scripts/fetch_ising_assets.sh
#   3. scripts/bootstrap_gpu_runtime_env.sh         (if --with-gpu given AND nvidia-smi is on PATH)
#   4. scripts/verify_install.sh
#
# USAGE
#   bash scripts/bootstrap_all.sh                 # CPU-only setup + Ising assets + verify
#   bash scripts/bootstrap_all.sh --with-gpu      # above + Tier 3 GPU runtime (cu13)
#   bash scripts/bootstrap_all.sh --auto-gpu      # auto-run GPU bootstrap if nvidia-smi present
#
# ENV OVERRIDES
#   DECODEROPS_PYTHON=/abs/path/to/python       # pick a specific interpreter
#   DECODEROPS_ISING_REF=<branch|tag|sha>       # pin the vendor checkout
#
# EXIT CODES
#   0  all selected phases completed AND verify_install.sh exited with
#      overall_status in ("ready", "degraded")
#   >0 first phase that failed; look at the preceding log/status JSONs
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/.decoderops/logs"
LOG_FILE="${LOG_DIR}/bootstrap_all_${STAMP}.log"
mkdir -p "${LOG_DIR}"

WITH_GPU="no"
AUTO_GPU="no"
for arg in "$@"; do
  case "${arg}" in
    --with-gpu)  WITH_GPU="yes" ;;
    --auto-gpu)  AUTO_GPU="yes" ;;
    -h|--help)
      sed -n '2,35p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "unknown arg: ${arg}" >&2
      exit 2
      ;;
  esac
done

log()   { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}"; }
banner() {
  local line="================================================================"
  log "${line}"; log "  $1"; log "${line}"
}

run_phase() {
  local name="$1" script="$2"
  banner "phase: ${name}"
  log "invoking ${script}"
  if bash "${script}" 2>&1 | tee -a "${LOG_FILE}"; then
    log "phase ok: ${name}"
    return 0
  else
    log "phase FAILED: ${name}"
    return 1
  fi
}

banner "Quantum DecoderOps Workbench - bootstrap_all (stamp=${STAMP})"
log "repo root = ${REPO_ROOT}"
log "with_gpu  = ${WITH_GPU}"
log "auto_gpu  = ${AUTO_GPU}"
log "log file  = ${LOG_FILE}"

run_phase "Tier 1 core env"         "${REPO_ROOT}/scripts/bootstrap_core_env.sh" || exit 10
run_phase "NVIDIA Ising assets"     "${REPO_ROOT}/scripts/fetch_ising_assets.sh" || exit 20

gpu_decision="skip"
if [[ "${WITH_GPU}" == "yes" ]]; then
  gpu_decision="run"
elif [[ "${AUTO_GPU}" == "yes" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "auto-gpu: nvidia-smi found on PATH"
    gpu_decision="run"
  else
    log "auto-gpu: no nvidia-smi; skipping GPU phase"
  fi
fi

if [[ "${gpu_decision}" == "run" ]]; then
  # GPU bootstrap is allowed to exit non-zero without killing the orchestrator:
  # verify_install.sh will record the partial state and the operator can decide.
  run_phase "Tier 3 GPU runtime (cu13)" "${REPO_ROOT}/scripts/bootstrap_gpu_runtime_env.sh" || \
    log "GPU runtime bootstrap returned non-zero; continuing to verification to capture state"
fi

run_phase "verification" "${REPO_ROOT}/scripts/verify_install.sh" || exit 40

banner "bootstrap_all finished"
log "review: .decoderops/environment_report.json"
log "log:    ${LOG_FILE}"
exit 0
