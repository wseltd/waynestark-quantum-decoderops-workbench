#!/usr/bin/env bash
#
# scripts/bootstrap_core_env.sh
# ---------------------------------------------------------------------------
# Quantum DecoderOps Workbench - Tier 1 bootstrap.
#
# PURPOSE
#   Provision the project-local Python virtual environment at .venv/ with
#   the full Tier 1 dependency set declared in requirements/core.txt, plus
#   a pinned lock file at requirements/core.lock.txt that reproducible
#   rebuilds will use.
#
# SCOPE / SAFETY
#   - All installs are confined to .venv/ inside the repo directory.
#   - No apt/dnf/brew calls. No sudo. No writes to /usr/, /etc/, /opt/.
#   - No modifications to shell rc files or global git config.
#   - If a .venv/ already exists, it is RENAMED (not deleted) to
#     .venv-prebootstrap-<utcstamp>/ so the operator can inspect or
#     remove it manually. No destructive overwrite.
#
# PYTHON INTERPRETER SELECTION
#   Per research-and-resourced.md §"Exact toolchain" and
#   nvidia-ising-calibration.md §"Technical requirements", the venv
#   targets Python 3.12 by default (Ising-Decoding supports 3.11-3.13;
#   nvidia-modelopt[onnx] requires <3.13 for the quantisation path).
#   Order of preference: python3.12 > python3.11 > python3.13 (warn).
#   Override with DECODEROPS_PYTHON=/abs/path/to/python.
#
# ARTIFACTS PRODUCED
#   .venv/                              - the provisioned virtual environment
#   requirements/core.lock.txt          - pip-freeze of resolved Tier 1 deps
#   .decoderops/logs/bootstrap_core_<stamp>.log   - full log of this run
#   .decoderops/bootstrap_core.json     - machine-readable status record
#
# EXIT CODES
#   0  all Tier 1 packages installed AND imported successfully
#   1  infrastructure error (missing python, failed venv creation)
#   2  pip install failure on required package
#   3  import smoke-test failure (installed but not importable)
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME="bootstrap_core_env"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/.decoderops/logs"
STATE_DIR="${REPO_ROOT}/.decoderops"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${STAMP}.log"
STATUS_FILE="${STATE_DIR}/bootstrap_core.json"
LOCK_FILE="${REPO_ROOT}/requirements/core.lock.txt"
REQ_FILE="${REPO_ROOT}/requirements/core.txt"
VENV_DIR="${REPO_ROOT}/.venv"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"

# -- helpers ---------------------------------------------------------------

log()   { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}"; }
info()  { log "INFO  $*"; }
warn()  { log "WARN  $*"; }
step()  { log "STEP  $*"; }
fail()  { log "FATAL $*"; exit_with_status "failed" "$1" "${2:-1}"; }

exit_with_status() {
  local status="$1" msg="$2" code="${3:-0}"
  python3 - "${STATUS_FILE}" "${status}" "${msg}" "${STAMP}" "${LOG_FILE}" \
    "${VENV_DIR}" "${PY_BIN:-unknown}" "${PY_VERSION:-unknown}" <<'PY' || true
import json, sys, os, datetime
path, status, msg, stamp, log_file, venv, py_bin, py_ver = sys.argv[1:9]
payload = {
    "schema": "decoderops.bootstrap.core.v1",
    "script": "scripts/bootstrap_core_env.sh",
    "run_stamp_utc": stamp,
    "finished_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": status,
    "message": msg,
    "log_file": log_file,
    "venv_dir": venv,
    "python_binary": py_bin,
    "python_version": py_ver,
}
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
  exit "${code}"
}

banner() {
  local line="================================================================"
  log "${line}"
  log "  ${1}"
  log "${line}"
}

# -- phase 0: preflight ----------------------------------------------------

banner "Quantum DecoderOps Workbench - Tier 1 core bootstrap"
info "repo root       = ${REPO_ROOT}"
info "log file        = ${LOG_FILE}"
info "status file     = ${STATUS_FILE}"
info "requirements    = ${REQ_FILE}"

if [[ ! -f "${REQ_FILE}" ]]; then
  fail "requirements file missing at ${REQ_FILE}" 1
fi

# -- phase 1: python interpreter detection --------------------------------

banner "Phase 1 / 5 : selecting Python interpreter"

PY_BIN=""
if [[ -n "${DECODEROPS_PYTHON:-}" ]]; then
  if [[ -x "${DECODEROPS_PYTHON}" ]]; then
    PY_BIN="${DECODEROPS_PYTHON}"
    info "DECODEROPS_PYTHON override = ${PY_BIN}"
  else
    fail "DECODEROPS_PYTHON=${DECODEROPS_PYTHON} is not executable" 1
  fi
else
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PY_BIN="$(command -v "${candidate}")"
      info "found candidate: ${PY_BIN}"
      break
    fi
  done
fi

if [[ -z "${PY_BIN}" ]]; then
  fail "no Python 3.11/3.12/3.13 interpreter found on PATH" 1
fi

PY_VERSION="$("${PY_BIN}" -c 'import sys;print("%d.%d.%d" % sys.version_info[:3])')"
PY_MAJOR_MINOR="$("${PY_BIN}" -c 'import sys;print("%d.%d" % sys.version_info[:2])')"
info "selected python = ${PY_BIN} (${PY_VERSION})"

case "${PY_MAJOR_MINOR}" in
  3.11|3.12) info "Python ${PY_MAJOR_MINOR} is a primary target; all Tier 1 + Tier 3 paths supported" ;;
  3.13)      warn "Python 3.13 is within Ising-Decoding range but nvidia-modelopt[onnx] requires <3.13; ONNX quantisation export path will be unavailable" ;;
  *)         fail "Python ${PY_MAJOR_MINOR} is outside the supported 3.11-3.13 range per Ising-Decoding; set DECODEROPS_PYTHON to a 3.12 or 3.11 interpreter" 1 ;;
esac

# ensure python -m venv is available
if ! "${PY_BIN}" -c "import ensurepip, venv" 2>/dev/null; then
  fail "selected Python lacks venv/ensurepip modules; install python${PY_MAJOR_MINOR}-venv and re-run" 1
fi

# -- phase 2: relocate any existing .venv ---------------------------------

banner "Phase 2 / 5 : preparing virtual environment directory"

if [[ -e "${VENV_DIR}" ]]; then
  BACKUP_DIR="${REPO_ROOT}/.venv-prebootstrap-${STAMP}"
  warn "existing ${VENV_DIR} found; relocating to ${BACKUP_DIR} (NOT deleted)"
  mv "${VENV_DIR}" "${BACKUP_DIR}"
  info "previous venv preserved at ${BACKUP_DIR}"
fi

info "creating fresh venv at ${VENV_DIR}"
"${PY_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

info "venv python    = $(${VENV_PY} -c 'import sys;print(sys.executable)')"
info "venv version   = $(${VENV_PY} -c 'import sys;print(sys.version.split()[0])')"

# -- phase 3: upgrade base tooling ----------------------------------------

banner "Phase 3 / 5 : upgrading pip / setuptools / wheel"

# Redirect tee into the log but fail on pip errors via set -e
"${VENV_PIP}" install --upgrade pip setuptools wheel 2>&1 | tee -a "${LOG_FILE}"

info "pip        = $(${VENV_PIP} --version)"
info "setuptools = $(${VENV_PY} -c 'import setuptools;print(setuptools.__version__)')"
info "wheel      = $(${VENV_PY} -c 'import wheel;print(wheel.__version__)')"

# -- phase 4: install Tier 1 requirements ---------------------------------

banner "Phase 4 / 5 : installing Tier 1 packages from requirements/core.txt"

if ! "${VENV_PIP}" install -r "${REQ_FILE}" 2>&1 | tee -a "${LOG_FILE}"; then
  fail "pip install -r ${REQ_FILE} failed; see ${LOG_FILE}" 2
fi

info "writing lock file to ${LOCK_FILE}"
"${VENV_PIP}" freeze --exclude-editable > "${LOCK_FILE}"
info "lock file contains $(wc -l < "${LOCK_FILE}") pinned packages"

# -- phase 5: import smoke test -------------------------------------------

banner "Phase 5 / 5 : import smoke-test for critical Tier 1 modules"

SMOKE_MODULES=(
  "stim"
  "pymatching"
  "sinter"
  "numpy"
  "scipy"
  "torch"
  "onnx"
  "onnxruntime"
  "pandas"
  "pyarrow"
  "fastapi"
  "pydantic"
  "typer"
  "rich"
  "duckdb"
  "sqlalchemy"
  "psycopg"
  "alembic"
  "jinja2"
  "reportlab"
  "markdown"
  "structlog"
  "hydra"
  "omegaconf"
  "safetensors"
  "matplotlib"
  "ldpc"
  "beliefmatching"
  "pytest"
  "hypothesis"
  "ruff"
  "mypy"
)

FAILED_IMPORTS=()
for mod in "${SMOKE_MODULES[@]}"; do
  if out="$("${VENV_PY}" -c "import importlib,sys; m=importlib.import_module('${mod}'); print(getattr(m,'__version__','unknown'))" 2>&1)"; then
    info "  [ok]   ${mod} == ${out}"
  else
    warn "  [fail] ${mod}: ${out}"
    FAILED_IMPORTS+=("${mod}")
  fi
done

if [[ ${#FAILED_IMPORTS[@]} -gt 0 ]]; then
  fail "smoke-test failed for: ${FAILED_IMPORTS[*]}" 3
fi

banner "Tier 1 bootstrap SUCCESS"
info "summary:"
info "  venv            = ${VENV_DIR}"
info "  python          = ${PY_VERSION}"
info "  packages pinned = $(wc -l < "${LOCK_FILE}")"
info "  log             = ${LOG_FILE}"
info "  status record   = ${STATUS_FILE}"
info ""
info "NEXT STEPS:"
info "  1. bash scripts/fetch_ising_assets.sh       # clone NVIDIA/Ising-Decoding + git-lfs pull"
info "  2. bash scripts/bootstrap_gpu_runtime_env.sh # optional Tier 3 GPU runtime (cu13)"
info "  3. bash scripts/verify_install.sh            # full env verification + JSON report"

exit_with_status "ok" "Tier 1 core bootstrap completed successfully" 0
