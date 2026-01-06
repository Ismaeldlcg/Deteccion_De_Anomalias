# 🚀 Quick Start - Pipeline de Mantenimiento

## 1️⃣ Primeros Pasos Locales (5 min)

```bash
# Clonar repo
git clone https://github.com/tu-usuario/animal-anomaly-detection.git
cd animal-anomaly-detection

# Setup automático
bash setup.sh

# Activar entorno
source venv/bin/activate

# Verificar instalación
pytest tests/ -v
```

## 2️⃣ Configurar GitHub (5 min)

```bash
# Instalar GitHub CLI si no lo tienes
brew install gh  # macOS
apt install gh   # Linux
choco install gh # Windows

# Autenticarse
gh auth login

# Setup de secrets
bash github_setup.sh
```

### Secrets a Configurar

1. **DISCORD_WEBHOOK** (opcional)
   - Para notificaciones en Discord
   - Obtén en: Server Settings → Webhooks

2. **PYPI_API_TOKEN** (opcional)
   - Para publicar en PyPI
   - Obtén en: https://pypi.org/account/

## 3️⃣ Configuraciones de Rama (GitHub Web)

1. Settings → Branches → main
   - ✅ Require status checks to pass
   - ✅ Require reviews before merging

## 📊 Workflows Automáticos

### ✅ Al hacer PUSH
```
Push code → Tests → Security → Build → Deploy
```

### ✅ Al hacer PULL REQUEST
```
PR created → Tests → Linting → Code Review
```

### ⏰ Scheduled (Semanal)
```
Monday 0:00 UTC → Security scan → Dependency check
```

## 🎯 Comandos Más Usados

```bash
# Desarrollo diario
make run                  # Iniciar servidor
make test                 # Ejecutar tests
make lint                 # Verificar código
make format              # Formatear código

# Antes de hacer commit
make pre-commit          # Ejecutar hooks

# Para debugging
make docker-run          # Con Docker
make coverage            # Ver cobertura

# Documentación
make docs                # Generar docs
make docs-serve         # Servir en 8000
```

## 🐳 Quick Docker Commands

```bash
# Desarrollo
docker-compose up -d

# Ver logs
docker-compose logs -f web

# Ejecutar tests
docker-compose run test

# Shell en contenedor
docker-compose exec web bash

# Detener
docker-compose down
```

## 📈 Monitorear en GitHub

### Actions Tab
```
Repository → Actions → Ver workflows en tiempo real
```

### Security Tab
```
Repository → Security → Dependabot alerts, code scanning
```

### Releases
```
Repository → Releases → Descargar artifacts
```

## ⚙️ Personalizar Workflows

### Cambiar rama principal
```yaml
# En .github/workflows/*.yml
on:
  push:
    branches: [ main, develop ]  # ← Cambiar aquí
```

### Agregar notificaciones
```yaml
# En notify.yml
- name: Notify Slack
  uses: slackapi/slack-github-action@v1
  with:
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Agregar custom step
```yaml
- name: Mi paso personalizado
  run: |
    echo "Haciendo algo custom"
    python custom_script.py
```

## 🔍 Debugging

### Tests fallan
```bash
pytest tests/ -vv --tb=long
pytest tests/test_app.py::TestFlaskRoutes::test_index_route -v
```

### Linting issues
```bash
flake8 .
black --check .
```

### Type checking
```bash
mypy .
```

### Docker issues
```bash
docker-compose logs -f
docker-compose down -v
docker-compose up --build
```

## 📚 Estructura de Carpetas

```
project/
├── .github/workflows/          # GitHub Actions
│   ├── test.yml               # Tests & Quality
│   ├── security.yml           # Security checks
│   ├── deploy.yml             # Build & Deploy
│   ├── docs.yml               # Documentation
│   └── notify.yml             # Notifications
├── tests/                      # Test files
│   ├── test_app.py
│   └── test_feature_extraction.py
├── .pre-commit-config.yaml     # Pre-commit hooks
├── pyproject.toml              # Tool config
├── requirements.txt            # Dependencies
├── Dockerfile                  # Container
├── docker-compose.yml          # Local setup
├── Makefile                    # Commands
├── setup.sh                    # Init script
└── README_PIPELINE.md          # Docs completas
```

## ✅ Checklist para Nuevo Repo

- [ ] Push código a GitHub
- [ ] Ir a Settings → Actions → General → Allow all actions
- [ ] Agregar secrets (DISCORD_WEBHOOK, PYPI_API_TOKEN)
- [ ] Ir a Settings → Branches → Requerir checks
- [ ] Crear rama develop
- [ ] Hacer primer commit
- [ ] Ver workflows en Actions tab

## 🆘 Help & Support

### Errores Comunes

**"ModuleNotFoundError"**
```bash
pip install -r requirements.txt --force-reinstall
```

**"Port already in use"**
```bash
docker-compose down
# o cambiar puerto en docker-compose.yml
```

**"Tests failing locally but passing in CI"**
```bash
# Usar mismo Python version
python --version
# Reinstalar dependencias
pip install -r requirements.txt --force-reinstall
```

**"Pre-commit hook failed"**
```bash
pre-commit run --all-files
# o skipear para commit específico
git commit --no-verify
```

## 📖 Recursos

- [Documentación Completa](./README_PIPELINE.md)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Docker Docs](https://docs.docker.com/)
- [pytest Docs](https://docs.pytest.org/)

## 💡 Pro Tips

1. **Local pre-commit antes de push**
   ```bash
   make pre-commit
   ```

2. **Tests con coverage**
   ```bash
   make coverage
   # Abre htmlcov/index.html
   ```

3. **Formato automático**
   ```bash
   make format
   ```

4. **Ver qué hace cada comando**
   ```bash
   make help
   ```

5. **Workflows en paralelo**
   - Test, Security, Build corren simultáneamente
   - Deploy espera a que todos pasen

---

**¿Dudas?** Revisa `README_PIPELINE.md` para docs completas
