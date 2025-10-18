#!/usr/bin/env bash

set -euo pipefail

SERVICE="trading-bot-pi.service"
PI_DEPLOY_ROOT="/srv/trading-bot-pi/app"
PI_ARTIFACT_ROOT="${PI_DEPLOY_ROOT}/storage/artifacts"
PYTHON_BIN="${PYTHON_BIN:-/home/stephang/venv-pi/bin/python3}"
CUR_LINK="${PI_ARTIFACT_ROOT}/current"

abort() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_cmd() {
  for cmd in "$@"; do
    command -v "$cmd" >/dev/null 2>&1 || abort "Command not found: $cmd"
  done
  [[ -x "$PYTHON_BIN" ]] || abort "Python interpreter not found or not executable: $PYTHON_BIN"
}

ram_guard() {
  local available
  available=$(awk '/MemAvailable/ {printf "%.0f", $2/1024}' /proc/meminfo 2>/dev/null || echo 0)
  if [[ "$available" -lt 200 ]]; then
    abort "Not enough free RAM: ${available}MB (need >= 200MB)"
  fi
}

verify_checksum() {
  local tag=$1
  local manifest="${PI_ARTIFACT_ROOT}/${tag}/sha256sum.txt"
  [[ -f "$manifest" ]] || abort "Checksum manifest not found: $manifest"
  (cd "${PI_ARTIFACT_ROOT}/${tag}" && sha256sum --check sha256sum.txt)
}

warmup_model() {
  local tag=$1
  TAG="$tag" "$PYTHON_BIN" - <<'PY'
import json
import os
import sys

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as exc:
    sys.stderr.write(f"[ERROR] Warm-up dependencies missing: {exc}\n")
    sys.exit(2)

tag = os.environ.get("TAG")
# Keep in sync with PI_ARTIFACT_ROOT above.
base = "/srv/trading-bot-pi/app/storage/artifacts"
signature_path = os.path.join(base, "input_signature.json")
model_path = os.path.join(base, tag, "model.onnx")

if not os.path.isfile(signature_path):
    sys.stderr.write(f"[ERROR] Missing input signature: {signature_path}\n")
    sys.exit(3)
if not os.path.isfile(model_path):
    sys.stderr.write(f"[ERROR] Missing model file: {model_path}\n")
    sys.exit(4)

with open(signature_path, "r", encoding="utf-8") as fh:
    signature = json.load(fh)

dummy_inputs = {}
for name, meta in signature.items():
    shape = meta.get("shape")
    dtype = meta.get("dtype", "float32")
    if not shape:
        sys.stderr.write(f"[ERROR] Signature for {name} missing shape\n")
        sys.exit(5)
    try:
        dummy_inputs[name] = np.zeros(shape, dtype=dtype)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to build dummy input {name}: {exc}\n")
        sys.exit(6)

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
session.run(None, dummy_inputs)
print("[OK] Warm-up successful.")
PY
}

dry_run_warmup() {
  local tag=$1
  echo "[INFO] Performing warm-up dry-run for tag $tag"
  echo "[INFO] Using Python interpreter: ${PYTHON_BIN}"
  if warmup_model "$tag"; then
    echo "[INFO] Warm-up dry-run succeeded, proceeding to deployment."
  else
    abort "Warm-up dry-run failed; aborting deployment."
  fi
}

switch_and_restart() {
  local tag=$1
  ln -sfn "${PI_ARTIFACT_ROOT}/${tag}" "$CUR_LINK"
  systemctl restart "$SERVICE"
  systemctl status "$SERVICE" --no-pager
  journalctl -u "$SERVICE" -n 200 --no-pager
}

rollback() {
  local prev=$1
  [[ -d "${PI_ARTIFACT_ROOT}/${prev}" ]] || abort "Artifact tag not found: ${prev}"
  ln -sfn "${PI_ARTIFACT_ROOT}/${prev}" "$CUR_LINK"
  systemctl restart "$SERVICE"
  systemctl status "$SERVICE" --no-pager
  journalctl -u "$SERVICE" -n 200 --no-pager
}

main() {
  require_cmd awk sha256sum systemctl journalctl

  local cmd=${1:-}
  case "$cmd" in
    deploy)
      local tag=${2:-}
      [[ -n "$tag" ]] || abort "Usage: tradingctl deploy <TAG>"
      [[ -d "${PI_ARTIFACT_ROOT}/${tag}" ]] || abort "Artifact tag not found: ${PI_ARTIFACT_ROOT}/${tag}"
      ram_guard
      verify_checksum "$tag"
      dry_run_warmup "$tag"
      switch_and_restart "$tag"
      ;;
    rollback)
      local prev=${2:-}
      [[ -n "$prev" ]] || abort "Usage: tradingctl rollback <PREV_TAG>"
      rollback "$prev"
      ;;
    status)
      systemctl status "$SERVICE" --no-pager
      ;;
    logs)
      journalctl -u "$SERVICE" -n 200 --no-pager
      ;;
    *)
      abort "Usage: tradingctl {deploy <TAG>|rollback <PREV_TAG>|status|logs}"
      ;;
  esac
}

main "$@"
