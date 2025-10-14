# Trading Bot Pi (ML Runner)
- ML-consumer (ONNX) + execution voor Pi
- Start: systemd `tradingbot-pi.service` → `src/apps/runner/main_v2_with_ml.py --non-interactive`
- Config: `config.yaml` (ml_overlay.artifacts_dir/latest_file naar ONNX)
- Logs: `journalctl -u tradingbot-pi -f`

## Pi-lightweight policy

- Alleen runtime/inference dependencies in `requirements*.txt`. Verboden: scikit-learn, xgboost, lightgbm, torch, jax, numba, tensorflow, cupy.
- `pre-commit run --all-files` en CI draaien `ci/policy_check.py` om dit af te dwingen.
- Nieuwe deps? Review via Standards-Steward en Architect-Reviewer.

## tradingctl.sh

CLI voor beheer onder `/home/stephang/trading-bot-pi-clean/scripts/tradingctl.sh`.

```
./scripts/tradingctl.sh deploy <TAG>
./scripts/tradingctl.sh rollback <PREV_TAG>
./scripts/tradingctl.sh status
./scripts/tradingctl.sh logs
```

### Deploy flow
1. RAM-check (`MemAvailable` ≥ 200MB) → abort anders.
2. Checksum verificatie (`sha256sum --check`).
3. Warm-up: dummy batch via onnxruntime + `storage/artifacts/input_signature.json`.
4. Pas na succesvolle warm-up: symlink switch (`storage/artifacts/current`) + systemd restart.
5. Toon `systemctl status` en laatste 200 `journalctl` regels.

### Rollback

`./scripts/tradingctl.sh rollback <PREV_TAG>` zet de `storage/artifacts/current` symlink terug en herstart de service.

### Troubleshooting
- Warm-up fout → waarschijnlijk signature/model mismatch; check `input_signature.json`.
- Checksum mismatch → sync opnieuw of valideer manifest.
- RAM guard → vrij geheugen door modellen/logs op te ruimen.
- Review logs via `./scripts/tradingctl.sh logs` of `journalctl -u tradingbot-pi -f`.

## ✅ Policy Safelist (Examples Only)

- `app/examples/` mag eenvoudige demonstraties bevatten voor documentatie/doelgroeptesten.
- Tag alle voorbeeldcode als **NO DEPLOY** en scheid ze van productiecode.
- Voeg geen `simple_*`, `*_lite.py`, `*_demo.py` of `*_v2_test.py` toe buiten deze mappen; herstel bestaande modules i.p.v. fallback-bestanden.
