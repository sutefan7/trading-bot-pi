# Trading Bot Pi (ML Runner)
- ML-consumer (ONNX) + execution voor Pi
- Start: systemd `tradingbot-pi.service` → `src/apps/runner/main_v2_with_ml.py --non-interactive`
- Config: `config.yaml` (ml_overlay.artifacts_dir/latest_file naar ONNX)
- Logs: `journalctl -u tradingbot-pi -f`
