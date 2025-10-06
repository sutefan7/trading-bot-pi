# ⚡ Quick GitHub Upload Checklist

**Voor snelle reference - lees GITHUB_UPLOAD_GUIDE.md voor details**

---

## 🚫 **NEVER COMMIT** (Top Priority)

```bash
# Deze bestanden ABSOLUUT NIET uploaden:
❌ secrets.yaml
❌ secrets.enc  
❌ encryption.key
❌ .env
❌ config.production.yaml
❌ config.local.yaml
❌ Bestanden met "AKIA", "sk_live", echte API keys
```

---

## ⚡ **5-Minute Upload Process**

```bash
# 1. Ga naar je directory
cd /home/stephang/trading-bot-pi-clean

# 2. Verwijder gevoelige files (als ze bestaan)
rm -f secrets.yaml secrets.enc encryption.key .env config.production.yaml

# 3. Check of er geen credentials in config.yaml zitten
grep -i "api_key\|api_secret\|token\|password" config.yaml
# Als je ECHTE credentials ziet → rm config.yaml

# 4. Quick scan voor secrets
grep -r "AKIA\|sk_\|api_key.*=.*[a-zA-Z0-9]" --include="*.py" --include="*.yaml" src/ config*.yaml 2>/dev/null
# Als er matches zijn → FIX ZE!

# 5. Initialiseer Git
git init
git add .gitignore
git commit -m "Add .gitignore"

# 6. Add files in stages (zodat je controle hebt)
git add README.md config.template.yaml requirements/
git commit -m "Add docs and requirements"

git add src/
git commit -m "Add source code"

# 7. Check wat Git WIL tracken (laatste check!)
git status

# 8. Maak PRIVATE repo op GitHub (via web)
# https://github.com/new
# - Name: trading-bot-pi
# - Visibility: PRIVATE ⭐
# - NO README/gitignore (we hebben ze al)

# 9. Link en push
git remote add origin git@github.com:YOUR_USERNAME/trading-bot-pi.git
git branch -M main
git push -u origin main

# 10. VERIFY (clone en check)
cd /tmp
git clone YOUR_REPO_URL test
cd test
find . -name "secrets.*" -o -name ".env" -o -name "encryption.key"
# Moet LEEG zijn!

# 11. Cleanup
cd .. && rm -rf test
```

---

## ✅ **Post-Upload Verification**

```bash
# Clone je repo en check:
[ ] Geen secrets.yaml
[ ] Geen .env files
[ ] Geen encryption.key
[ ] config.yaml bestaat NIET (alleen template)
[ ] venv/ bestaat NIET
[ ] storage/artifacts/ bestaat NIET (wel .gitkeep)
[ ] src/ bestaat WEL
[ ] README.md bestaat WEL
[ ] requirements/ bestaat WEL
```

---

## 🚨 **Emergency: Secret Committed**

```bash
# 1. REVOKE credential op exchange/service ONMIDDELLIJK!

# 2. Verwijder uit Git:
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch FILENAME" \
  --prune-empty --tag-name-filter cat -- --all

# 3. Force push
git push origin --force --all

# 4. Genereer NIEUWE credentials

# 5. Update je lokale .env met nieuwe credentials
```

---

## 📦 **Essential Files to Keep**

```
✅ Keep:
├── src/                     # All source code
├── requirements/            # Dependencies
├── config.template.yaml     # Template only
├── .gitignore              # Git ignore
├── README.md               # Documentation
├── *.service               # Systemd files
├── scripts/                # Helper scripts
└── storage/.gitkeep        # Directory structure

❌ Remove/Ignore:
├── venv/                   # Virtual env
├── secrets.yaml            # Secrets
├── .env                    # Environment
├── config.yaml             # Real config
├── config.production.yaml  # Production config
├── storage/artifacts/      # Models
├── storage/logs/           # Logs
├── data/                   # Historical data
└── *.log                   # Log files
```

---

## 🔐 **Security Scores**

### ⚠️ **STOP - Do NOT upload if ANY of these exist:**

```bash
# Run these checks:
find . -name "*.env" -o -name "secrets.yaml" -o -name "encryption.key"
# Output MUST be empty!

grep -r "api_key.*=.*[A-Za-z0-9]{20}" . --include="*.yaml"
# Output MUST be empty or only show template/comments!

grep -r "AKIA" . --include="*.py" --include="*.yaml"
# Output MUST be empty!
```

### ✅ **Safe to upload if:**

- [ ] All above checks are empty
- [ ] `.gitignore` exists and is comprehensive
- [ ] `config.template.yaml` has NO real credentials
- [ ] Repository will be **PRIVATE**
- [ ] No `venv/` directory
- [ ] No large files (>10MB) except via Git LFS

---

## 📞 **Need Help?**

**Read full guide:** `GITHUB_UPLOAD_GUIDE.md`

**Common issues:**
- "I see my API key in commits" → See Emergency section above
- "Large files rejected" → Use Git LFS or add to .gitignore
- "Config has secrets" → Use config.template.yaml and .env instead

---

**Last updated:** 2025-10-05




