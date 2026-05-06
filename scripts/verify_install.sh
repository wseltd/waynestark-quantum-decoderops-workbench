#!/usr/bin/env bash
#
# scripts/verify_install.sh
# ---------------------------------------------------------------------------
# Quantum DecoderOps Workbench - environment verifier (entry point).
#
# Thin shell wrapper around scripts/_verify_install.py. The Python module
# is where the real logic lives - it probes every Tier 1 import, every
# Tier 3 capability, the Ising asset inventory, and the GPU/driver state,
# and writes a machine-readable report to .decoderops/environment_report.json
# that app/core/capability.py (built later via later product tickets) consumes
# to surface backend availability in the API, the compatibility matrix,
# and the deployment-readiness report.
#
# PRECONDITIONS
#   - scripts/bootstrap_core_env.sh has run (.venv/ exists).
#   - Optionally: scripts/bootstrap_gpu_runtime_env.sh has run.
#   - Optionally: scripts/fetch_ising_assets.sh has run.
#
# EXIT CODES
#   0  overall_status in ("ready", "degraded") - safe to proceed with
#      the planning + run pipeline; any "degraded" capabilities are
#      still explicitly recorded in the JSON report.
#   1  overall_status == "missing_required" - Tier 1 core or Ising assets
#      broken; do NOT run the planning + run pipeline yet.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/.venv"
VERIFIER="${REPO_ROOT}/scripts/_verify_install.py"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "FATAL: .venv not found at ${VENV_DIR}; run scripts/bootstrap_core_env.sh first" >&2
  exit 1
fi
if [[ ! -f "${VERIFIER}" ]]; then
  echo "FATAL: missing ${VERIFIER}" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

exec "${VENV_DIR}/bin/python" "${VERIFIER}"
