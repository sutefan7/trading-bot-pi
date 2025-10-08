# Admin Guide - Trading Bot Pi

## Roles
- stephang: admin/maintainer. Full control, can deploy, edit config, view logs.
- trader: service user. Runs the systemd service; no sudo.

## Service
- Start: sudo systemctl start tradingbot-pi
- Stop: sudo systemctl stop tradingbot-pi
- Restart: sudo systemctl restart tradingbot-pi
- Status: sudo systemctl status tradingbot-pi
- Logs (live): journalctl -u tradingbot-pi -f

## Paths
- Working dir (service): /srv/trading-bot-pi/app
- Entry point: src/apps/runner/main_v2_with_ml.py
- Venv: app/.venv (owned by stephang, readable by trader)
- Config: app/config.yaml (prod overrides: app/config.production.yaml)
- Logs: app/logs/
- Models: app/storage/artifacts/ (latest.txt -> current model dir)

## Update Code
```bash
cd /srv/trading-bot-pi/app
# edit/pull as stephang
sudo systemctl restart tradingbot-pi
```

## Update Model
```bash
# copy new model dir under app/storage/artifacts/
echo "<model_dir_name>" > app/storage/artifacts/latest.txt
sudo systemctl restart tradingbot-pi
```

## Troubleshooting
- Service won't start:
  - Check venv: ls -la app/.venv/bin/python3
  - Permissions: trader must read+exec venv, write app/logs
  - journalctl -u tradingbot-pi -e
- No data errors:
  - Verify symbols/timeframes in app/config.yaml
  - Network check: curl https://api.kraken.com/0/public/Time

## Security Notes
- trader has no sudo. stephang manages everything.
- Keep API keys out of repo (.env not committed).
