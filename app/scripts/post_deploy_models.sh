#!/usr/bin/env bash
set -euo pipefail

# Post-deploy activator for model bundles on the Pi
# - Scans app/storage/artifacts/*/ for bundles
# - Builds/updates index.yaml mapping <SYMBOL> -> <DIR>
# - Optionally aligns app/config.yaml trading.symbols to the discovered set
# - Validates ONNX load for each bundle
# - Restarts trading service

APP_DIR="/srv/trading-bot-pi/app"
ART_DIR="$APP_DIR/storage/artifacts"
INDEX_FILE="$ART_DIR/index.yaml"
CONFIG_FILE="$APP_DIR/config.yaml"
SERVICE_NAME="tradingbot-pi"
PYTHON_BIN="$APP_DIR/.venv/bin/python3"

echo "[+] Scanning bundles in $ART_DIR"
mapfile -t BUNDLES < <(find "$ART_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#BUNDLES[@]} -eq 0 ]]; then
  echo "[!] No bundle directories found in $ART_DIR" >&2
  exit 1
fi

# Heuristic: symbol from bundle dir name replacing '_' with '-' if needed and suffixing '-USD' if not present
declare -A MAP
for d in "${BUNDLES[@]}"; do
  # If dir like BTC_USD → symbol BTC-USD; if already like XMR_USD keep mapping to XMR-USD
  sym=${d//_/\-}
  # Common suffix normalization
  case "$sym" in
    *-USD) : ;; 
    *) sym="$sym" ;; 
  esac
  MAP["$sym"]="$d"
done

echo "[+] Writing $INDEX_FILE"
{
  echo "models:"
  for sym in "${!MAP[@]}"; do
    printf "  %s: %s\n" "$sym" "${MAP[$sym]}"
  done | sort
} > "$INDEX_FILE"

echo "[+] Validating ONNX load for each bundle (CPUExecutionProvider)"
ART_DIR="$ART_DIR" "$PYTHON_BIN" - <<'PY'
import os, sys
import onnxruntime as ort
base=os.environ.get('ART_DIR')
ok=0; fail=0
for d in sorted(os.listdir(base)):
    p=os.path.join(base,d)
    if not os.path.isdir(p):
        continue
    mp=os.path.join(p,'model.onnx')
    if not os.path.exists(mp):
        print('MISSING_ONNX', d)
        fail+=1
        continue
    try:
        ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
        print('OK', d)
        ok+=1
    except Exception as e:
        print('LOAD_FAIL', d, str(e)[:200])
        fail+=1
print('SUMMARY OK', ok, 'FAIL', fail)
sys.exit(1 if fail else 0)
PY

# Align config.yaml trading.symbols to found symbols (backup first)
echo "[+] Aligning trading.symbols in $CONFIG_FILE (backup will be created)"
cp "$CONFIG_FILE" "$CONFIG_FILE.bak.$(date +%Y%m%d_%H%M%S)"

tmp_yaml=$(mktemp)
CONFIG_FILE="$CONFIG_FILE" INDEX_FILE="$INDEX_FILE" TMP_OUT="$tmp_yaml" "$PYTHON_BIN" - <<'PY'
import sys, yaml, os
cfg_path=os.environ['CONFIG_FILE']
idx_path=os.environ['INDEX_FILE']
with open(cfg_path,'r') as f:
    cfg=yaml.safe_load(f)
with open(idx_path,'r') as f:
    idx=yaml.safe_load(f)
symbols=sorted(list((idx.get('models') or {}).keys()))
cfg.setdefault('trading',{})['symbols']=symbols
with open(os.environ['TMP_OUT'],'w') as f:
    yaml.safe_dump(cfg,f,sort_keys=False)
PY
mv "$tmp_yaml" "$CONFIG_FILE"

echo "[+] Restarting $SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"
sleep 2
echo "[+] Tail service logs (last 20 lines):"
journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true

echo "[✓] Done. Bundles indexed, ONNX validated, config aligned, service restarted."


