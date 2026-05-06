#!/usr/bin/env bash
#
# scripts/fetch_ising_assets.sh
# ---------------------------------------------------------------------------
# Quantum DecoderOps Workbench - Tier 2: real NVIDIA Ising-Decoding asset
# fetch.
#
# Tier taxonomy (per product spec):
#   Tier 1 = core Python stack           -> scripts/bootstrap_core_env.sh
#   Tier 2 = REAL NVIDIA ISING ARTEFACTS -> THIS SCRIPT
#   Tier 3 = GPU / accelerated runtimes  -> scripts/bootstrap_gpu_runtime_env.sh
#
# Tier 2 is NOT optional and NOT replaced by fake checkpoints. Per the
# research pack, stubbing the decoder model with synthetic torch tensors
# would turn the Workbench into a toy; integration tests MUST run
# against the real shipped .pt files.
#
# PURPOSE
#   Clone NVIDIA/Ising-Decoding into vendor/Ising-Decoding at a known
#   commit, pull its git-lfs-tracked model checkpoints, install the
#   vendor's public inference requirements into the project venv, run
#   NVIDIA's shipped python-compat check, compute SHA256 of the two
#   shipped decoder models, and record the full asset inventory to
#   .decoderops/ising_assets.json so the Workbench can treat these as
#   first-class real product artefacts.
#
#   The two checkpoints are REQUIRED for real-artefact integration tests
#   and for any report claim about Ising-Decoder pipelines. Per
#   research-and-resourced.md, stubbing these with fake torch tensors
#   would turn the Workbench into a toy. We refuse that path here.
#
# UPSTREAM
#   Source (Apache-2.0):  https://github.com/NVIDIA/Ising-Decoding
#   Shipped checkpoints:
#     models/Ising-Decoder-SurfaceCode-1-Fast.pt      (receptive field 9)
#     models/Ising-Decoder-SurfaceCode-1-Accurate.pt  (receptive field 13)
#
# PRECONDITIONS
#   - scripts/bootstrap_core_env.sh has run and .venv/ exists.
#   - git and git-lfs are installed on PATH.
#   - Outbound HTTPS to github.com is reachable.
#
# SAFETY
#   - Clones only into ./vendor/Ising-Decoding (inside project dir).
#   - No global git config changes.
#   - Installs vendor requirements into the existing .venv.
#   - No sudo, no apt-get.
#
# IDEMPOTENCY
#   - If vendor/Ising-Decoding already exists, fetches + resets to the
#     pinned commit instead of re-cloning.
#
# ARTIFACTS
#   vendor/Ising-Decoding/                      cloned vendor repo
#   .decoderops/ising_assets.json               machine-readable asset inventory
#   .decoderops/logs/fetch_ising_assets_<stamp>.log
#
# EXIT CODES
#   0  both shipped checkpoints are present, load-verified, and recorded
#   1  infrastructure error (no .venv, no git, no git-lfs)
#   2  git clone / git-lfs pull failure
#   3  vendor pip-install failure
#   4  checkpoint file missing or still an LFS pointer after pull
#   5  torch.load sanity-check of a checkpoint failed
# ---------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

SCRIPT_NAME="fetch_ising_assets"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO_ROOT}/.decoderops/logs"
STATE_DIR="${REPO_ROOT}/.decoderops"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}_${STAMP}.log"
STATUS_FILE="${STATE_DIR}/ising_assets.json"
VENV_DIR="${REPO_ROOT}/.venv"

VENDOR_ROOT="${REPO_ROOT}/vendor"
VENDOR_DIR="${VENDOR_ROOT}/Ising-Decoding"
VENDOR_URL="https://github.com/NVIDIA/Ising-Decoding.git"

# Pin to a known-good ref. We deliberately track the default branch HEAD
# for now (main) so the fetch script picks up upstream fixes; when the
# product cuts a release, a DECODEROPS_ISING_COMMIT override will pin
# this to a specific commit SHA. The actual commit resolved is always
# recorded into ising_assets.json.
VENDOR_REF="${DECODEROPS_ISING_REF:-main}"

MODEL_RELPATHS=(
  "models/Ising-Decoder-SurfaceCode-1-Fast.pt"
  "models/Ising-Decoder-SurfaceCode-1-Accurate.pt"
)

mkdir -p "${LOG_DIR}" "${STATE_DIR}" "${VENDOR_ROOT}"

log()   { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_FILE}"; }
info()  { log "INFO  $*"; }
warn()  { log "WARN  $*"; }
step()  { log "STEP  $*"; }
banner() {
  local line="================================================================"
  log "${line}"; log "  $1"; log "${line}"
}

write_status_json() {
  # args: overall_status message [model_meta_json]
  local overall_status="$1" overall_msg="$2" model_json="${3:-[]}"
  python3 - "${STATUS_FILE}" "${overall_status}" "${overall_msg}" "${STAMP}" \
    "${LOG_FILE}" "${VENDOR_DIR}" "${VENDOR_URL}" "${VENDOR_REF}" \
    "${VENDOR_COMMIT:-unknown}" "${model_json}" <<'PY' || true
import json, sys, os, datetime
(path, overall_status, overall_msg, stamp, log_file,
 vendor_dir, vendor_url, vendor_ref, vendor_commit, model_json) = sys.argv[1:11]
try:
    models = json.loads(model_json)
except Exception:
    models = []
payload = {
    "schema": "decoderops.ising_assets.v1",
    "script": "scripts/fetch_ising_assets.sh",
    "run_stamp_utc": stamp,
    "finished_utc": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": overall_status,
    "message": overall_msg,
    "log_file": log_file,
    "vendor": {
        "url": vendor_url,
        "ref": vendor_ref,
        "commit": vendor_commit,
        "path": vendor_dir,
        "license": "Apache-2.0",
    },
    "models": models,
}
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
    fh.write("\n")
PY
}

fatal() {
  warn "FATAL $1"
  write_status_json "failed" "$1" "[]"
  exit "${2:-1}"
}

# -- phase 0: preflight ----------------------------------------------------

banner "Quantum DecoderOps Workbench - NVIDIA Ising-Decoding asset fetch"
info "repo root     = ${REPO_ROOT}"
info "log file      = ${LOG_FILE}"
info "status file   = ${STATUS_FILE}"
info "vendor url    = ${VENDOR_URL}"
info "vendor ref    = ${VENDOR_REF}"
info "vendor dir    = ${VENDOR_DIR}"

command -v git >/dev/null 2>&1      || fatal "git not on PATH" 1
command -v git-lfs >/dev/null 2>&1  || fatal "git-lfs not on PATH (required for Ising checkpoints)" 1
[[ -d "${VENV_DIR}" ]]              || fatal "venv not found at ${VENV_DIR}; run scripts/bootstrap_core_env.sh first" 1

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

# make sure git-lfs hooks are in the user's git config (user-scope, not system)
git lfs install --skip-repo 2>&1 | tee -a "${LOG_FILE}" || warn "git lfs install returned non-zero (may already be set up)"

# -- phase 1: clone or update vendor repo ---------------------------------

banner "Phase 1 / 4 : cloning or updating vendor repository"

if [[ -d "${VENDOR_DIR}/.git" ]]; then
  info "existing vendor checkout detected; fetching updates"
  (
    cd "${VENDOR_DIR}"
    git fetch --tags origin 2>&1 | tee -a "${LOG_FILE}"
    git checkout "${VENDOR_REF}" 2>&1 | tee -a "${LOG_FILE}"
    git reset --hard "origin/${VENDOR_REF}" 2>&1 | tee -a "${LOG_FILE}" || \
      git reset --hard "${VENDOR_REF}" 2>&1 | tee -a "${LOG_FILE}"
  ) || fatal "git fetch/checkout/reset failed in ${VENDOR_DIR}" 2
else
  info "cloning ${VENDOR_URL} -> ${VENDOR_DIR}"
  if ! git clone "${VENDOR_URL}" "${VENDOR_DIR}" 2>&1 | tee -a "${LOG_FILE}"; then
    fatal "git clone failed" 2
  fi
  (
    cd "${VENDOR_DIR}"
    git checkout "${VENDOR_REF}" 2>&1 | tee -a "${LOG_FILE}"
  ) || fatal "git checkout ${VENDOR_REF} failed" 2
fi

VENDOR_COMMIT="$(cd "${VENDOR_DIR}" && git rev-parse HEAD)"
info "vendor commit = ${VENDOR_COMMIT}"

# -- phase 2: git-lfs pull for checkpoint files ---------------------------

banner "Phase 2 / 4 : git-lfs pull for model checkpoints"
(
  cd "${VENDOR_DIR}"
  git lfs pull 2>&1 | tee -a "${LOG_FILE}"
) || fatal "git lfs pull failed in ${VENDOR_DIR}" 2

# -- phase 3: install vendor public-inference requirements into venv ------

banner "Phase 3 / 4 : installing vendor public inference requirements"

VENDOR_REQ="${VENDOR_DIR}/code/requirements_public_inference.txt"
if [[ -f "${VENDOR_REQ}" ]]; then
  info "installing ${VENDOR_REQ}"
  if ! "${VENV_PIP}" install -r "${VENDOR_REQ}" 2>&1 | tee -a "${LOG_FILE}"; then
    warn "vendor pip install had errors; some vendor-only deps may be unavailable"
    # Treat as non-fatal: the core product uses stim/pymatching/onnx already
    # installed by Tier 1, and the vendor requirements mostly overlap.
    # verify_install.sh will re-check imports.
  fi
else
  warn "vendor requirements file not found at ${VENDOR_REQ} (repo layout may have changed upstream); continuing"
fi

# Run the vendor's own python-compat script if present; non-fatal.
COMPAT_SCRIPT="${VENDOR_DIR}/code/scripts/check_python_compat.sh"
if [[ -x "${COMPAT_SCRIPT}" ]]; then
  info "running vendor python-compat check"
  (cd "${VENDOR_DIR}" && bash "${COMPAT_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}") || \
    warn "vendor python-compat check reported issues (non-fatal)"
fi

# -- phase 4: verify checkpoints and record asset inventory ---------------

banner "Phase 4 / 4 : verifying shipped decoder checkpoints"

# Build a JSON array describing each expected checkpoint.
MODEL_META_JSON="["
first=1
for relpath in "${MODEL_RELPATHS[@]}"; do
  abs="${VENDOR_DIR}/${relpath}"
  info "checking ${relpath}"

  if [[ ! -f "${abs}" ]]; then
    warn "  missing: ${abs}"
    fatal "shipped checkpoint not present after git-lfs pull: ${relpath}" 4
  fi

  size_bytes="$(stat -c '%s' "${abs}")"
  if (( size_bytes < 1048576 )); then
    # < 1 MiB is almost certainly still an LFS pointer file.
    warn "  size ${size_bytes} bytes < 1 MiB; appears to still be an LFS pointer"
    fatal "checkpoint ${relpath} looks like an unresolved LFS pointer; re-run git-lfs pull" 4
  fi

  sha256="$(sha256sum "${abs}" | awk '{print $1}')"
  info "  size = ${size_bytes} bytes   sha256 = ${sha256}"

  # Sanity-check by loading with torch; catches corrupted downloads.
  load_out="$("${VENV_PY}" - "${abs}" <<'PY' 2>&1
import sys
try:
    import torch
    path = sys.argv[1]
    obj = torch.load(path, map_location="cpu", weights_only=False)
    kind = type(obj).__name__
    keys = list(obj.keys())[:5] if isinstance(obj, dict) else []
    print(f"OK kind={kind} keys_sample={keys}")
    sys.exit(0)
except Exception as e:
    print(f"FAIL {type(e).__name__}: {e}")
    sys.exit(1)
PY
)"
  if [[ "${load_out}" != OK* ]]; then
    warn "  torch.load failed: ${load_out}"
    fatal "checkpoint ${relpath} failed torch.load sanity-check" 5
  fi
  info "  torch.load ${load_out}"

  # Append to JSON
  [[ $first -eq 1 ]] || MODEL_META_JSON+=","
  first=0
  MODEL_META_JSON+="$(python3 -c "
import json
print(json.dumps({
    'relpath': '${relpath}',
    'abspath': '${abs}',
    'size_bytes': ${size_bytes},
    'sha256': '${sha256}',
    'torch_load': 'ok',
}))")"
done
MODEL_META_JSON+="]"

write_status_json "ok" "Both shipped Ising-Decoder checkpoints verified" "${MODEL_META_JSON}"

banner "Ising-Decoding assets ready"
info "vendor commit = ${VENDOR_COMMIT}"
info "status JSON   = ${STATUS_FILE}"
info "log file      = ${LOG_FILE}"

exit 0
