#!/usr/bin/env bash

set -euo pipefail

SERVICE="trading-bot-pi.service"
PI_DEPLOY_ROOT="/srv/trading-bot-pi/app"
PI_ARTIFACT_ROOT="${PI_DEPLOY_ROOT}/storage/artifacts"
PYTHON_BIN="${PYTHON_BIN:-/home/stephang/venv-pi/bin/python3}"
CUR_LINK="${PI_ARTIFACT_ROOT}/current"
LATEST_FILE="${PI_DEPLOY_ROOT}/latest.txt"

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

resolve_tag() {
  local requested="${1:-}"
  if [[ -n "$requested" && "$requested" != "latest" ]]; then
    printf '%s\n' "$requested"
    return 0
  fi

  [[ -f "$LATEST_FILE" ]] || abort "latest.txt not found: ${LATEST_FILE}"
  local raw
  raw=$(<"$LATEST_FILE")
  raw=${raw//[$'\t\r\n ']/}
  raw=${raw%/}
  raw=${raw##*/}
  [[ -n "$raw" ]] || abort "latest.txt is empty; specify a tag explicitly"
  printf '%s\n' "$raw"
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
  local target="${PI_ARTIFACT_ROOT}/${tag}"
  [[ -d "$target" ]] || abort "Artifact tag not found: $target"

  local manifests=()
  while IFS= read -r -d '' file; do
    manifests+=("$file")
  done < <(find "$target" -type f -name 'sha256sum.txt' -print0)

  [[ ${#manifests[@]} -gt 0 ]] || abort "No checksum manifests found under: $target"

  for manifest in "${manifests[@]}"; do
    local dir
    dir=$(dirname "$manifest")
    echo "[INFO] Verifying checksums in ${dir#${PI_ARTIFACT_ROOT}/}"
    (cd "$dir" && sha256sum --check sha256sum.txt)
  done
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
signature_path = os.path.join(base, tag, "input_signature.json")
model_path = os.path.join(base, tag, "model.onnx")

if not os.path.isfile(signature_path):
    signature_path = os.path.join(base, tag, "input_signature.json")
if not os.path.isfile(signature_path):
    signature_path = os.path.join(base, "input_signature.json")
if not os.path.isfile(signature_path):
    sys.stderr.write(f"[ERROR] Missing input signature: {signature_path}\n")
    sys.exit(3)

if not os.path.isfile(model_path):
    import glob
    matches = glob.glob(os.path.join(base, tag, "*", "model.onnx"))
    for candidate in matches:
        if os.path.isfile(candidate):
            model_path = candidate
            break
if not os.path.isfile(model_path):
    sys.stderr.write(f"[ERROR] Missing model file: {model_path}\n")
    sys.exit(4)

with open(signature_path, "r", encoding="utf-8") as fh:
    signature = json.load(fh)

if isinstance(signature, dict):
    signature_map = {name: meta for name, meta in signature.items() if isinstance(meta, dict)}
else:
    signature_map = {}

session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
inputs = session.get_inputs()
feed = {}

def _fallback_shape(meta_list, index):
    for meta in meta_list:
        shape = meta.get("shape")
        if isinstance(shape, list) and index < len(shape):
            try:
                val = int(shape[index])
                if val > 0:
                    return val
            except Exception:
                continue
    return 1

meta_list = list(signature_map.values())

for idx, inp in enumerate(inputs):
    meta = signature_map.get(inp.name)
    if meta is None and idx < len(meta_list):
        meta = meta_list[idx]

    dtype = "float32"
    if meta and isinstance(meta.get("dtype"), str):
        dtype = meta["dtype"]

    try:
        np_dtype = np.dtype(dtype)
    except Exception:
        np_dtype = np.float32

    dims = []
    if inp.shape:
        for dim_idx, dim in enumerate(inp.shape):
            if isinstance(dim, int) and dim > 0:
                dims.append(int(dim))
            else:
                dims.append(_fallback_shape(meta_list if meta_list else ([meta] if meta else []), dim_idx))
    else:
        dims = [1]

    try:
        feed[inp.name] = np.zeros(dims, dtype=np_dtype)
    except Exception as exc:
        sys.stderr.write(f"[ERROR] Failed to build dummy input for {inp.name}: {exc}\n")
        sys.exit(6)

session.run(None, feed)
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
      local tag
      tag=$(resolve_tag "${2:-}")
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
      abort "Usage: tradingctl {deploy [<TAG>|latest]|rollback <PREV_TAG>|status|logs}"
      ;;
  esac
}

main "$@"
