# 🍓 Raspberry Pi - Essential Files & Directories

**Voor:** Trading Bot v2.0  
**Platform:** Raspberry Pi (ARM64)  
**Purpose:** Live trading met ONNX ML inference

---

## 🎯 **KRITIEKE BESTANDEN (Bot werkt NIET zonder deze)**

### **1. Source Code**
```
✅ ESSENTIEEL - Bot core:
src/apps/runner/
├── main_v2_with_ml.py          # 🚀 MAIN ENTRY POINT
├── inference_client.py         # ML model management
├── ml_overlay.py              # ML overlay logic
├── shadow_ml_observer.py      # Shadow mode tracking
├── notification_manager.py    # Notifications
└── instance_lock.py           # Single instance protection

✅ ESSENTIEEL - Trading logic:
src/core/
├── config.py                  # Config loader
├── scheduler.py               # Trading scheduler
├── secrets.py                 # Credentials management
└── signal_types.py            # Signal definitions

src/execution/
├── broker.py                  # Broker interface (Kraken)
├── executor.py                # Trade execution
└── circuit_breaker.py         # Safety circuit breaker

src/risk/
└── risk_manager.py            # Risk management

✅ ESSENTIEEL - Data & Features:
src/data/
└── data_manager.py            # Data fetching & storage

src/features/
├── schema.py                  # Feature definitions (NEW!)
└── pipeline.py                # Feature engineering

✅ ESSENTIEEL - Strategies:
src/strategies/
├── indicators.py              # Technical indicators (NEW!)
├── trend_follow.py            # Trend strategy (NEW!)
├── mean_revert.py             # Mean reversion (NEW!)
└── breakout.py                # Breakout strategy (NEW!)

✅ ESSENTIEEL - ML Serving:
src/serving/
└── predict.py                 # ONNX inference (FIXED!)

✅ ESSENTIEEL - Filters:
src/filters/
├── regime_filter.py           # Market regime filter
└── universe_selector.py       # Asset selection

✅ ESSENTIEEL - Utils:
src/utils/
└── logger_setup.py            # Logging configuration
```

**Status:** ✅ **Alle aanwezig en getest**

---

## ⚙️ **CONFIGURATIE (Bot werkt NIET zonder deze)**

```
✅ KRITIEK:
config.yaml                    # Runtime configuratie
├── trading symbols            # Welke coins te traden
├── risk parameters           # Risk limits
├── ML settings              # ML overlay config
└── API settings             # Kraken API config

⚠️ VEILIG (maar verwijder voor GitHub):
.env                          # Secrets (NEVER commit!)
├── KRAKEN_API_KEY
├── KRAKEN_API_SECRET
└── TRADING_BOT_ENV

✅ OPTIONEEL:
config.production.yaml        # Production overrides
config.template.yaml          # Template (safe voor GitHub)
```

---

## 📦 **DEPENDENCIES (Bot werkt NIET zonder deze)**

```
✅ KRITIEK:
requirements/
└── requirements.txt          # Python packages
    ├── pandas==2.0.3
    ├── numpy==1.24.4
    ├── onnxruntime==1.16.3  # ⚠️ KRITIEK voor ML!
    ├── ta==0.11.0           # Technical analysis
    ├── ccxt==4.3.98         # Exchange API
    ├── krakenex==2.1.0      # Kraken
    ├── loguru==0.7.2        # Logging
    └── ... (zie bestand)

✅ PYTHON VENV:
venv/                         # Virtual environment
└── (gegenereerd met pip install -r requirements.txt)
```

---

## 🤖 **ML MODELS & ARTIFACTS (Voor ML trading)**

```
⚠️ ZEER BELANGRIJK (als ML enabled):
storage/artifacts/
├── latest.txt                # Points to current model
└── multi_coin_YYYYMMDD_HHMM/
    ├── model.onnx            # ⚠️ KRITIEK - ONNX model
    ├── scaler.pkl            # Feature scaler
    ├── thresholds.yaml       # Decision thresholds
    ├── featureset.json       # Feature definitions
    └── metadata.json         # Model metadata

📝 OPMERKING:
- Models NIET in Git (te groot)
- Upload via rsync/scp van Mac
- Bot werkt zonder (fallback naar non-ML strategies)
```

---

## 🗄️ **DATA & CACHE (Runtime gegenereerd)**

```
🔄 RUNTIME (Auto-gegenereerd):
storage/
├── artifacts/                # ML models (zie boven)
├── logs/                    # Application logs
│   └── *.log
└── cache/                   # Data cache (optioneel)

logs/                        # Systemd logs
└── trading-bot.log

📝 OPMERKING:
- Niet in Git (runtime data)
- Wordt automatisch aangemaakt
- Kan safely deleted worden (regenerates)
```

---

## 🔧 **SYSTEMD SERVICE FILES (Voor auto-start)**

```
✅ BELANGRIJK:
/etc/systemd/system/
├── tradingbot-pi.service    # Main bot service
└── inference.service        # Optioneel: Separate inference

🏠 LOCAL COPY:
tradingbot-pi.service        # Template in repo
inference.service            # Template in repo

SETUP:
sudo cp tradingbot-pi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable tradingbot-pi
sudo systemctl start tradingbot-pi
```

---

## 📚 **DOCUMENTATION (Voor jou, niet voor bot functie)**

```
ℹ️ HULPVOL (maar niet nodig voor draaien):
README.md                    # Project overview
IMPROVEMENTS_SUMMARY.md      # Alle verbeteringen
GITHUB_UPLOAD_GUIDE.md       # Upload security guide
QUICK_UPLOAD_CHECKLIST.md    # Quick reference
FILES_TO_UPLOAD.md           # File classification
PI_ESSENTIAL_FILES.md        # Dit document
check_upload_safety.sh       # Safety checker

📝 OPMERKING:
- Handig voor onderhoud
- Niet nodig voor bot functie
- Wel uploaden naar GitHub (documentatie)
```

---

## ❌ **NIET NODIG OP PI (Training only)**

```
❌ VERWIJDER/NEGEER:
├── __pycache__/            # Python cache (auto-generated)
├── *.pyc                   # Compiled Python
├── .DS_Store               # macOS files
├── .idea/                  # IDE settings
├── .vscode/                # VS Code settings
├── tests/                  # Unit tests (run on Mac)
├── notebooks/              # Jupyter notebooks
├── data/raw/               # Raw training data
└── models/training/        # Training checkpoints

📝 DEZE ZIJN ALLEEN OP MAC NODIG:
src/apps/ml_workflow/       # Training scripts
├── train_model.py
├── multi_coin_trainer.py
└── export_model.py

requirements-trainer.txt     # Training dependencies
├── xgboost                 # ❌ Niet op Pi
├── lightgbm               # ❌ Niet op Pi
├── matplotlib             # ❌ Niet op Pi
└── optuna                 # ❌ Niet op Pi
```

---

## 📊 **DIRECTORY STRUCTURE (Complete overzicht)**

```
trading-bot-pi-clean/
│
├── 🚀 KRITIEK - Source Code
│   └── src/
│       ├── apps/runner/              ⭐ ENTRY POINT
│       ├── core/                     ⭐ Core logic
│       ├── execution/                ⭐ Trading
│       ├── risk/                     ⭐ Risk management
│       ├── data/                     ⭐ Data management
│       ├── features/                 ⭐ Feature engineering
│       ├── strategies/               ⭐ Trading strategies
│       ├── serving/                  ⭐ ML inference
│       ├── filters/                  ⭐ Market filters
│       └── utils/                    ⭐ Utilities
│
├── ⚙️ KRITIEK - Configuration
│   ├── config.yaml                   ⭐ Main config
│   ├── config.template.yaml          📖 Template
│   └── .env                          🔒 Secrets (local only)
│
├── 📦 KRITIEK - Dependencies
│   ├── requirements/
│   │   └── requirements.txt          ⭐ Python packages
│   └── venv/                         ⭐ Virtual environment
│
├── 🤖 BELANGRIJK - ML Models (als enabled)
│   └── storage/
│       └── artifacts/                ⭐ ONNX models
│           ├── latest.txt
│           └── multi_coin_*/
│
├── 🔧 BELANGRIJK - Service files
│   ├── tradingbot-pi.service         ⭐ Systemd service
│   └── inference.service             ⚙️ Optioneel
│
├── 📚 HULPVOL - Documentation
│   ├── README.md
│   ├── IMPROVEMENTS_SUMMARY.md
│   ├── GITHUB_UPLOAD_GUIDE.md
│   └── PI_ESSENTIAL_FILES.md         📖 Dit document
│
└── 🗑️ RUNTIME - Gegenereerd
    ├── logs/                         🔄 Auto-generated
    ├── __pycache__/                  🔄 Python cache
    └── storage/cache/                🔄 Data cache
```

---

## 🎯 **MINIMALE SETUP (Absolute minimum om te draaien)**

```bash
MINIMAAL VEREIST:
/home/stephang/trading-bot-pi-clean/
├── src/                              # Alle source code
├── config.yaml                       # Configuration
├── .env                              # API credentials
├── requirements/requirements.txt     # Dependencies
├── venv/                             # Python packages installed
└── storage/
    ├── .gitkeep                      # Directory structure
    └── artifacts/                    # ML models (optioneel)
        └── latest.txt                # Points to model

PLUS systemd:
/etc/systemd/system/tradingbot-pi.service

TOTAAL: ~50MB (zonder venv)
         ~250MB (met venv)
```

---

## 🚀 **DEPLOYMENT CHECKLIST**

```
PRE-DEPLOYMENT (Op Mac):
[ ] Train models (Mac only)
[ ] Export to ONNX (Mac only)
[ ] Test models locally
[ ] Create artifacts tar.gz
[ ] Update requirements.txt

PI DEPLOYMENT:
[ ] Clone/pull repo naar Pi
[ ] Copy config.yaml (zonder secrets!)
[ ] Setup .env met credentials
[ ] Create venv: python3 -m venv venv
[ ] Install deps: venv/bin/pip install -r requirements/requirements.txt
[ ] Upload ML models: rsync artifacts/ pi:storage/artifacts/
[ ] Test startup: venv/bin/python3 src/apps/runner/main_v2_with_ml.py
[ ] Setup systemd service
[ ] Enable & start service

VERIFICATION:
[ ] Check logs: journalctl -u tradingbot-pi -f
[ ] Verify no errors
[ ] Check memory: free -h (should be < 300MB)
[ ] Monitor for 1 hour
```

---

## 💾 **DISK SPACE REQUIREMENTS**

```
BREAKDOWN:
Source code:        ~5MB      (src/, config)
Documentation:      ~1MB      (*.md files)
Python venv:        ~250MB    (dependencies)
ML models:          ~50MB     (per model bundle)
Logs (1 week):      ~10MB     (depends on verbosity)
Data cache:         ~20MB     (12 symbols × 200 bars)

TOTAAL: ~336MB (met 1 model)
        ~400MB (met 2 models + logs)

SD CARD RECOMMENDATION:
- Minimum: 8GB (maar gebruik 16GB+)
- Recommended: 32GB (room voor meerdere models)
- Class 10 of beter (snellere I/O)
```

---

## 🔄 **UPDATE PROCEDURE**

### **Code Updates (van GitHub):**
```bash
cd /home/stephang/trading-bot-pi-clean
git pull origin main
venv/bin/pip install -r requirements/requirements.txt
sudo systemctl restart tradingbot-pi
```

### **Model Updates (van Mac):**
```bash
# Op Mac:
tar -czf model_bundle.tar.gz storage/artifacts/multi_coin_YYYYMMDD_HHMM/
scp model_bundle.tar.gz pi@raspberrypi:~/

# Op Pi:
cd ~/trading-bot-pi-clean
tar -xzf ~/model_bundle.tar.gz
echo "multi_coin_YYYYMMDD_HHMM" > storage/artifacts/latest.txt
# Bot detecteert automatisch nieuwe model (binnen 1 min)
```

### **Config Updates:**
```bash
# Edit config
nano config.yaml

# Restart bot
sudo systemctl restart tradingbot-pi

# Verify
journalctl -u tradingbot-pi -f
```

---

## ⚠️ **KRITIEKE WAARSCHUWINGEN**

### **NEVER DELETE:**
```
❌ NEVER DELETE:
src/                    # Source code
config.yaml             # Configuration
.env                    # Credentials
venv/                   # Python environment
storage/artifacts/      # ML models
```

### **SAFE TO DELETE:**
```
✅ SAFE TO DELETE (regenerates):
__pycache__/           # Python cache
logs/                  # Logs (maak backup eerst)
storage/cache/         # Data cache
*.pyc                  # Compiled Python
.DS_Store             # macOS files
```

### **BACKUP BEFORE DELETE:**
```
⚠️ BACKUP FIRST:
storage/artifacts/     # ML models (hard to recreate)
config.yaml           # Your settings
.env                  # Your API keys
logs/                 # Trading history
```

---

## 🔍 **FILE SIZE REFERENCE**

```
GROTE BESTANDEN (> 1MB):
venv/                          ~250MB  (Python packages)
storage/artifacts/*.onnx       ~40MB   (per ONNX model)
logs/trading-bot.log           ~2MB    (per week)

MIDDELGROTE (100KB - 1MB):
src/strategies/*.py            ~150KB  (total)
src/apps/runner/*.py          ~200KB  (total)

KLEINE (< 100KB):
config.yaml                    ~5KB
.env                          ~1KB
*.md files                    ~50KB   (each)
```

---

## 📞 **TROUBLESHOOTING**

### **Bot start niet:**
```bash
# Check welke bestanden ontbreken:
ls -la src/apps/runner/main_v2_with_ml.py  # Moet bestaan
ls -la config.yaml                          # Moet bestaan
ls -la .env                                 # Moet bestaan
ls -la venv/bin/python3                     # Moet bestaan

# Check permissions:
chmod +x src/apps/runner/main_v2_with_ml.py

# Check dependencies:
venv/bin/python3 -c "import onnxruntime; print('OK')"
```

### **ML niet werkend:**
```bash
# Check models aanwezig:
ls -la storage/artifacts/latest.txt         # Moet bestaan
cat storage/artifacts/latest.txt            # Toon model path
ls -la storage/artifacts/$(cat storage/artifacts/latest.txt)/model.onnx

# Als niet aanwezig:
# Upload from Mac via rsync
```

### **Disk vol:**
```bash
# Check disk usage:
df -h

# Clear logs:
sudo journalctl --vacuum-time=7d
rm -rf logs/*.log

# Clear cache:
rm -rf storage/cache/*

# Remove old models:
cd storage/artifacts
ls -lt  # Toon sorted by date
# Delete old model directories (keep latest 2-3)
```

---

## ✅ **QUICK REFERENCE**

```
🚀 START BOT:
sudo systemctl start tradingbot-pi

📊 CHECK STATUS:
sudo systemctl status tradingbot-pi

📋 VIEW LOGS:
journalctl -u tradingbot-pi -f

🛑 STOP BOT:
sudo systemctl stop tradingbot-pi

🔄 RESTART BOT:
sudo systemctl restart tradingbot-pi

🔍 CHECK MEMORY:
free -h

💾 CHECK DISK:
df -h

📦 UPDATE CODE:
git pull && pip install -r requirements/requirements.txt && sudo systemctl restart tradingbot-pi
```

---

**Last Updated:** 2025-10-05  
**Version:** 2.0.0  
**Platform:** Raspberry Pi (ARM64)  
**Status:** ✅ COMPLETE & TESTED


