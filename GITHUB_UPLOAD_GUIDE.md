# 📤 GitHub Upload Guide voor Trading Bot Pi

## ⚠️ BELANGRIJK - LEES DIT EERST!

Deze guide helpt je om de trading bot **veilig** naar GitHub te uploaden zonder gevoelige informatie te lekken.

---

## 🚫 **WAT ABSOLUUT NIET UPLOADEN**

### **1. SECRETS & CREDENTIALS (KRITIEK!)**

❌ **NOOIT committen:**
```
secrets.yaml          # Plain text secrets
secrets.enc           # Encrypted secrets
encryption.key        # Encryption key
*.env                 # Environment files met API keys
config.production.yaml # Production config met credentials
config.local.yaml     # Local config met credentials
```

⚠️ **Waarom?**: Deze bevatten:
- Kraken API keys
- Telegram bot tokens
- Database credentials
- Encryption keys

💡 **Wat als je per ongeluk wel committed?**
```bash
# ONMIDDELLIJK:
# 1. Revoke alle API keys op Kraken
# 2. Genereer nieuwe keys
# 3. Verwijder uit git history:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.yaml" \
  --prune-empty --tag-name-filter cat -- --all
```

---

### **2. TRADING DATA & LOGS**

❌ **Te groot en/of gevoelig:**
```
storage/artifacts/    # ML models (100+ MB)
storage/models/       # Trained models
storage/logs/         # Trading logs (kunnen gevoelig zijn)
storage/trades/       # Trade history (financiële data!)
data/                 # Historical price data (GB's)
*.onnx               # ONNX models (groot)
*.parquet            # Price data
*.csv                # Export files
logs/                # Application logs
```

⚠️ **Waarom?**: 
- Te groot voor GitHub (>100MB limit)
- Bevat mogelijk gevoelige trading patterns
- Kan reverse-engineered worden

💡 **Alternatief**: 
- Models: Use Git LFS of externe storage (S3, GCS)
- Logs: Lokaal houden of externe logging service

---

### **3. VIRTUAL ENVIRONMENTS**

❌ **Niet committen:**
```
venv/                # Python virtual environment (100+ MB)
.venv/
ENV/
```

⚠️ **Waarom?**: 
- Enorm groot (500MB+)
- Platform-specific binaries
- Kan opnieuw gebouwd worden met `requirements.txt`

---

### **4. SYSTEM & IDE FILES**

❌ **Niet committen:**
```
__pycache__/         # Python cache
*.pyc
.DS_Store            # macOS
.idea/               # PyCharm
.vscode/             # VS Code settings (kunnen secrets bevatten)
```

---

## ✅ **WAT WEL UPLOADEN**

### **Essentiële Bestanden**

```
✅ src/                          # Alle source code
✅ requirements/requirements.txt  # Dependencies
✅ config.template.yaml          # Config template (GEEN secrets!)
✅ README.md                     # Documentatie
✅ .gitignore                    # Git ignore file
✅ tradingbot-pi.service         # Systemd service file
✅ inference.service             # Inference service file
✅ scripts/monitoring/           # Monitoring scripts
✅ storage/.gitkeep              # Directory structure
✅ logs/.gitkeep                 # Directory structure
```

---

## 🔒 **VEILIGHEIDSCHECK VOOR UPLOAD**

### **Stap 1: Verwijder gevoelige bestanden**

```bash
cd /home/stephang/trading-bot-pi-clean

# Check of gevoelige bestanden bestaan
find . -name "secrets.yaml" -o -name "secrets.enc" -o -name "*.env" -o -name "encryption.key"

# Als ze bestaan, verwijder ze (worden al genegeerd door .gitignore):
rm -f secrets.yaml secrets.enc encryption.key .env
```

### **Stap 2: Check config files**

```bash
# VERIFIEER dat config.yaml GEEN echte credentials bevat:
grep -i "api_key\|api_secret\|token\|password" config.yaml

# Als je credentials vindt, verwijder config.yaml:
rm config.yaml
```

### **Stap 3: Scan voor credentials**

```bash
# Installeer gitleaks (optional maar aanbevolen):
# https://github.com/gitleaks/gitleaks

# Of gebruik grep voor basis check:
grep -r -i "AKIA\|sk_\|api_key.*=" --include="*.py" --include="*.yaml" src/

# Als er matches zijn, FIX ZE VOOR JE UPLOADT!
```

### **Stap 4: Test .gitignore**

```bash
# Check wat Git ZAL tracken:
git status --ignored

# Verifieer dat deze NIET worden getrackt:
# - venv/
# - secrets.yaml
# - config.production.yaml
# - storage/artifacts/
# - *.log
```

---

## 📦 **UPLOAD PROCESS**

### **Optie A: Nieuwe Repository (Aanbevolen)**

```bash
cd /home/stephang/trading-bot-pi-clean

# 1. Initialiseer Git (als nog niet gedaan)
git init

# 2. Add .gitignore
git add .gitignore
git commit -m "Add .gitignore voor security"

# 3. Add essentiële bestanden (in stages)
git add README.md config.template.yaml requirements/
git commit -m "Add documentation and requirements"

git add src/
git commit -m "Add source code"

git add tradingbot-pi.service inference.service scripts/
git commit -m "Add service files and scripts"

git add storage/.gitkeep logs/.gitkeep
git commit -m "Add directory structure"

# 4. Maak private repo op GitHub
# Via web interface: github.com/new
# - Repository name: trading-bot-pi
# - Description: Cryptocurrency trading bot for Raspberry Pi with ML inference
# - Visibility: **PRIVATE** (belangrijk!)
# - DO NOT add README, .gitignore, or license (we hebben ze al)

# 5. Link en push
git remote add origin git@github.com:JOUW_USERNAME/trading-bot-pi.git
git branch -M main
git push -u origin main
```

### **Optie B: Bestaande Repo Updaten**

```bash
cd /home/stephang/trading-bot-pi-clean

# 1. Check huidige status
git status

# 2. Stage changes
git add -A

# 3. Review wat je gaat committen (BELANGRIJK!)
git status
git diff --staged

# 4. Als alles OK is, commit en push
git commit -m "Update trading bot code"
git push
```

---

## 🔐 **ENVIRONMENT VARIABLES SETUP**

In plaats van config files met secrets, gebruik environment variables:

### **Op Raspberry Pi**

```bash
# 1. Maak .env file (deze wordt NIET gecommit)
cat > .env << 'EOF'
# Kraken API
KRAKEN_API_KEY=your_key_here
KRAKEN_API_SECRET=your_secret_here
KRAKEN_SANDBOX=true

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Encryption key (genereer met: python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
TRADING_BOT_ENCRYPTION_KEY=your_key_here

# Environment
TRADING_BOT_ENV=production
EOF

# 2. Secure de file
chmod 600 .env

# 3. Load in systemd service (edit tradingbot-pi.service):
# [Service]
# EnvironmentFile=/home/stephang/trading-bot-pi-clean/.env
```

### **GitHub Secrets (voor CI/CD)**

Als je GitHub Actions wilt gebruiken:

1. Ga naar repo → Settings → Secrets and variables → Actions
2. Add secrets:
   - `KRAKEN_API_KEY`
   - `KRAKEN_API_SECRET`
   - `TELEGRAM_BOT_TOKEN`

---

## ✅ **POST-UPLOAD CHECKLIST**

Na uploaden, verifieer:

```bash
# 1. Clone je repo in een temp directory
cd /tmp
git clone https://github.com/JOUW_USERNAME/trading-bot-pi.git test-clone

# 2. Check dat GEEN secrets aanwezig zijn
cd test-clone
find . -name "secrets.yaml" -o -name "*.env" -o -name "encryption.key"
# Output moet LEEG zijn!

grep -r "AKIA\|sk_live\|api_key.*=" --include="*.py" --include="*.yaml" .
# Output moet LEEG zijn (of alleen comments/templates)!

# 3. Check dat essentiële bestanden WEL aanwezig zijn
ls -la src/ requirements/ config.template.yaml
# Moet allemaal bestaan

# 4. Als alles OK is:
cd ..
rm -rf test-clone
```

---

## 🚨 **OH SHIT, IK HEB EEN SECRET GECOMMIT!**

### **Immediate Actions**

```bash
# 1. REVOKE de credential ONMIDDELLIJK!
#    - Kraken: ga naar Settings → API → Delete key
#    - Telegram: praat met @BotFather → /revoke

# 2. Genereer NIEUWE credentials

# 3. Verwijder uit Git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/secret/file" \
  --prune-empty --tag-name-filter cat -- --all

# 4. Force push (als private repo en je bent de enige gebruiker)
git push origin --force --all

# 5. Roteer ALLE credentials die in de repo zaten

# 6. Overweeg repo te deleten en opnieuw te maken als het een public repo was
```

---

## 📚 **BEST PRACTICES**

### **1. Repository Structure**

```
trading-bot-pi/
├── src/                    # Source code (commit)
├── requirements/           # Dependencies (commit)
├── config.template.yaml    # Template (commit)
├── config.yaml            # Real config (DON'T commit)
├── .env                   # Secrets (DON'T commit)
├── .gitignore             # Git ignore (commit)
├── README.md              # Docs (commit)
├── storage/               # Runtime data (DON'T commit)
│   ├── .gitkeep          # Keep structure (commit)
│   ├── artifacts/        # Models (DON'T commit)
│   └── logs/             # Logs (DON'T commit)
└── venv/                  # Virtual env (DON'T commit)
```

### **2. Branching Strategy**

```bash
main          # Stable, production-ready code
├── develop   # Development branch
└── feature/  # Feature branches
```

### **3. Commit Messages**

```bash
# Good:
git commit -m "feat: Add ML overlay manager with shadow mode"
git commit -m "fix: Fix race condition in model reload"
git commit -m "docs: Update deployment guide"

# Bad:
git commit -m "update"
git commit -m "fix stuff"
git commit -m "asdf"
```

### **4. Tags for Releases**

```bash
# Tag stable versions
git tag -a v1.0.0 -m "Release v1.0.0 - Initial stable Pi version"
git push origin v1.0.0
```

---

## 🔍 **AUTOMATED SECURITY SCANNING**

### **Setup Pre-commit Hooks**

```bash
# 1. Install pre-commit
pip install pre-commit

# 2. Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10240']  # 10MB limit
      - id: detect-private-key
      - id: trailing-whitespace
EOF

# 3. Install hooks
pre-commit install

# Now every commit will be scanned automatically!
```

---

## 📞 **SUPPORT & RESOURCES**

- **GitHub Security**: https://docs.github.com/en/code-security
- **Gitleaks**: https://github.com/gitleaks/gitleaks
- **Git Filter-Branch**: https://git-scm.com/docs/git-filter-branch
- **Git LFS**: https://git-lfs.github.com/ (voor grote files)

---

## ✨ **SUMMARY CHECKLIST**

Gebruik deze checklist voor de upload:

- [ ] `.gitignore` is correct geconfigureerd
- [ ] Geen `secrets.yaml`, `secrets.enc`, of `.env` files in repo
- [ ] Geen echte API keys in config files
- [ ] `config.template.yaml` bevat alleen placeholder values
- [ ] Virtual environment (`venv/`) is uitgesloten
- [ ] Grote bestanden (models, data) zijn uitgesloten
- [ ] Repository is **PRIVATE** (tenzij je het expliciet public wilt)
- [ ] README.md is up-to-date
- [ ] Service files bevatten geen secrets
- [ ] Pre-commit hooks zijn geïnstalleerd (optional)
- [ ] Je hebt de repo gecloned en getest dat er geen secrets in zitten
- [ ] Environment variables zijn gedocumenteerd
- [ ] Deployment guide is compleet

**Als ALLES afgevinkt is: SAFE TO PUSH! 🚀**




