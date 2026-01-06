# 🔧 GUÍA PRÁCTICA: COMANDOS GIT Y CÓMO REUBICAR EL CÓDIGO

## 📥 PASO 1: CLONAR EL REPOSITORIO

### Comando básico:
```bash
git clone https://github.com/tu-usuario/animal-anomaly-detection.git
```

**¿Qué hace?**
- Descarga TODO el repositorio remoto a tu máquina
- Crea una carpeta con el nombre del proyecto
- Descarga todo el historial de cambios
- Configura automáticamente el "remote origin"

### Entrar a la carpeta:
```bash
cd animal-anomaly-detection
```

**¿Qué hace?**
- Navega a la carpeta del proyecto descargada

---

## 📂 PASO 2: VER LA ESTRUCTURA

### Ver qué archivos hay:
```bash
ls -la
```

**¿Qué hace?**
- Lista todos los archivos y carpetas
- Muestra permisos, tamaño, fecha de modificación
- Incluye archivos ocultos (empiezan con punto)

### Ver estructura visual:
```bash
tree
```

O si no tienes tree:
```bash
find . -type d | head -20
```

**¿Qué hace?**
- Muestra árbol de directorios
- Útil para entender la organización del proyecto

---

## 🔍 PASO 3: ENTENDER CADA ARCHIVO/CARPETA

### `.github/workflows/` - GitHub Actions Automáticos

**¿Qué es?**
- Carpeta especial que GitHub reconoce automáticamente
- Contiene archivos YAML (.yml) que definen trabajos automáticos

**Archivos dentro:**

#### `test.yml` - Pruebas automáticas
```yaml
name: Tests & Code Quality

on:
  push:
    branches: [ main, develop ]  # Se ejecuta cuando haces push
  pull_request:
    branches: [ main, develop ]  # Se ejecuta cuando haces PR
```
**¿Qué hace?**
- Ejecuta tests automáticamente cada vez que subes código
- Verifica que el código sea bueno con linting
- Genera reportes de cobertura
- Si falla, bloquea el merge en PR

**Flujo:** Push → Tests corren → Si pasan, OK / Si fallan, aviso

#### `security.yml` - Análisis de seguridad
```yaml
- name: Bandit security scan
  run: |
    bandit -r . -f json -o bandit-report.json || true
```
**¿Qué hace?**
- Escanea el código para encontrar vulnerabilidades
- Revisa si las dependencias tienen problemas
- Busca claves privadas accidentalmente subidas

**Flujo:** Código sube → Busca problemas de seguridad → Reporta si hay

#### `deploy.yml` - Construcción y distribución
```yaml
- name: Build and push Docker image
  uses: docker/build-push-action@v4
```
**¿Qué hace?**
- Crea una imagen Docker automáticamente
- La sube a GitHub Container Registry
- Si haces un tag (v1.0.0), también publica en PyPI
- Genera "releases" descargables

**Flujo:** Git tag → Build Docker → Push a registro → Release en GitHub

#### `docs.yml` - Documentación automática
```yaml
- name: Build documentation
  run: |
    cd docs
    sphinx-build -W -b html -d _build/doctrees . _build/html
```
**¿Qué hace?**
- Genera documentación HTML con Sphinx
- La sube automáticamente a GitHub Pages
- Disponible en: `https://tu-usuario.github.io/tu-repo`

**Flujo:** Push → Genera docs → Publica en web

#### `notify.yml` - Notificaciones
```yaml
- name: Notify on failure via Discord
  uses: sarisia/actions-status-discord@v1
```
**¿Qué hace?**
- Envía notificaciones a Discord cuando falla algo
- Crea issues automáticos en GitHub
- Comenta en PRs con resultados

**Flujo:** Workflow falla → Mensaje Discord + Issue creado

---

### `tests/` - Carpeta de pruebas

#### `test_feature_extraction.py` - Tests del módulo de features
```python
class TestFeatureExtractor:
    """Tests para FeatureExtractor class"""
    
    def test_extractor_initialization(self, extractor):
        """Test que el extractor se inicializa bien"""
        assert extractor.device == 'cpu'
        assert extractor.model_name == 'resnet50'
```

**¿Qué hace?**
- Prueba que la extracción de features funciona
- Verifica que los modelos carguen correctamente
- Comprueba que los archivos se guardan y cargan bien
- Corre automáticamente en cada push

**Cómo funcionan:**
1. Define un escenario (fixture)
2. Ejecuta código a probar
3. Verifica que el resultado sea correcto (assert)

#### `test_app.py` - Tests de la API Flask
```python
def test_index_route(self, client):
    """Test que la ruta principal funciona"""
    response = client.get('/')
    assert response.status_code == 200  # Debe retornar OK
```

**¿Qué hace?**
- Prueba cada endpoint de la API
- Verifica que retornan el resultado correcto
- Comprueba manejo de errores
- Valida que CORS esté configurado

---

### `pyproject.toml` - Configuración centralizada

```toml
[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']
```

**¿Qué hace?**
- Define cómo Black formatea el código (máximo 100 caracteres)
- Black automáticamente indenta y formatea TODO igual

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**¿Qué hace?**
- Dice a pytest dónde buscar tests
- Cuando corres `pytest`, busca archivos que empiezan con `test_`

```toml
[tool.mypy]
python_version = "3.8"
ignore_missing_imports = true
```

**¿Qué hace?**
- Configura type checking
- Verifica que tipos de datos uses correctamente
- Ejemplo: `def suma(a: int) -> int:` → mypy verifica que `a` sea int

---

### `.pre-commit-config.yaml` - Validación antes de subir

```yaml
- repo: https://github.com/psf/black
  rev: 23.1.0
  hooks:
    - id: black
```

**¿Qué hace?**
- Antes de hacer `git commit`, ejecuta automáticamente:
  1. Black (formatea el código)
  2. isort (ordena los imports)
  3. flake8 (verifica errores)
  4. mypy (type checking)
  5. bandit (seguridad)

**Si algo falla:**
- El commit se cancela
- Te muestra qué está mal
- Tienes que arreglarlo y intentar de nuevo

**Ventaja:** No subes código malo por accidente

---

### `requirements.txt` - Dependencias del proyecto

```
Flask==2.3.3              # Framework web
numpy>=1.21.0             # Números y arrays
torch>=1.10.0             # PyTorch (deep learning)
pytest>=7.0.0             # Testing
black>=23.0.0             # Formatting
```

**¿Qué hace?**
- Lista todas las librerías que necesitas
- El `==` significa "exactamente esta versión"
- El `>=` significa "esta versión o más reciente"

**Cómo instalar:**
```bash
pip install -r requirements.txt
```

---

### `Dockerfile` - Instrucciones para Docker

```dockerfile
FROM python:3.10-slim as base
WORKDIR /app
RUN apt-get install -y --no-install-recommends build-essential
```

**¿Qué hace línea por línea?**
1. `FROM` - Parte de una imagen base (Python 3.10)
2. `WORKDIR` - Crea carpeta `/app` en el contenedor
3. `RUN` - Ejecuta comandos (instala dependencias)

```dockerfile
FROM base as production
COPY app.py .
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

**¿Qué hace?**
- Copia tu código al contenedor
- Define qué comando ejecutar cuando inicia (gunicorn - servidor web)

**Para qué sirve Docker?**
- Tu app funciona igual en cualquier máquina
- No importa si tienes Windows, Mac o Linux
- Todo encerrado en un "contenedor"

---

### `docker-compose.yml` - Orquesta múltiples servicios

```yaml
services:
  web:
    build:
      context: .
      dockerfile: Dockerfile
      target: development
    ports:
      - "5000:5000"
```

**¿Qué hace?**
- `build` - Construye Docker image desde Dockerfile
- `ports` - Mapea puerto 5000 del contenedor al 5000 de tu PC
- `target: development` - Usa el stage "development" del Dockerfile

```yaml
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: anomaly
      POSTGRES_PASSWORD: anomaly_secure
```

**¿Qué hace?**
- Levanta una base de datos PostgreSQL
- Define usuario y contraseña
- La app puede conectarse a `db:5432`

**Comando para levantar todo:**
```bash
docker-compose up -d
```

**¿Qué hace?**
- `-d` = detached (corre en background)
- Levanta ambos servicios (web + db)
- Accedes en `http://localhost:5000`

---

### `Makefile` - Atajos de comandos

```makefile
.PHONY: test lint format

test:
	@echo "Running test suite..."
	pytest tests/ -v --cov

lint:
	@echo "Running linting checks..."
	flake8 . --max-line-length=100

format:
	@echo "Formatting code..."
	black . --line-length=100
```

**¿Qué hace?**
- Define comandos cortos
- `make test` → ejecuta `pytest tests/ -v --cov`
- `make lint` → ejecuta `flake8 . --max-line-length=100`

**Ventaja:** No tienes que acordarte de comandos largos

```bash
make test        # En lugar de: pytest tests/ -v --cov
make format      # En lugar de: black . --line-length=100 && isort .
```

---

### `setup.sh` - Script de configuración automática

```bash
#!/bin/bash
echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt
```

**¿Qué hace línea por línea?**
1. `#!/bin/bash` - Indica que es un script bash
2. `python3 -m venv venv` - Crea entorno virtual
3. `source venv/bin/activate` - Activa el entorno
4. `pip install -r requirements.txt` - Instala dependencias

**Cómo correr:**
```bash
bash setup.sh
```

**Ventaja:** Una línea y listo, en lugar de hacer todo manualmente

---

### `.gitignore` - Archivos que NO se suben

```
__pycache__/
*.pyc
.env
venv/
*.pkl
```

**¿Qué hace?**
- Dice a Git qué archivos IGNORAR
- `__pycache__/` - Cache de Python (basura)
- `.env` - Archivos con contraseñas/secrets (NUNCA subir)
- `venv/` - Entorno virtual (se descarga, no se sube)
- `*.pkl` - Archivos de datos grandes

**Importancia:** Si no ignoras `.env`, subes contraseñas a GitHub

---

## ⬆️ PASO 4: CÓMO REUBICAR EL CÓDIGO (REUBICAR = SUBIR)

### Opción A: Crear repo nuevo y subir TODO

```bash
# 1. Crear repo en GitHub (vía web, vacío)

# 2. Clonar el nuevo repo vacío
git clone https://github.com/tu-usuario/nuevo-repo.git
cd nuevo-repo

# 3. Copiar TODOS los archivos descargados
cp -r ../animal-anomaly-detection/* .

# 4. Ver qué cambió
git status
```

**¿Qué hace `git status`?**
- Muestra archivos nuevos (verde)
- Muestra archivos modificados (rojo)
- Te dice exactamente qué va a subir

### Paso a paso para subir:

```bash
# 1. Agregar TODO
git add .
```
**¿Qué hace?**
- Prepara TODOS los cambios para subir
- El punto (.) significa "todo"

```bash
# 2. Ver qué va a subir
git status
```

```bash
# 3. Crear "commit" (paquete de cambios)
git commit -m "Add complete CI/CD pipeline"
```

**¿Qué hace?**
- Empaqueta los cambios
- `-m` = mensaje (qué es lo que subes)
- El mensaje debe ser descriptivo

**Ejemplos de buenos mensajes:**
```
git commit -m "Add GitHub Actions workflows"
git commit -m "Add unit tests for feature extraction"
git commit -m "Configure Docker and docker-compose"
```

```bash
# 4. Subir a GitHub
git push origin main
```

**¿Qué hace?**
- `origin` = servidor remoto (GitHub)
- `main` = rama principal
- Sube tus cambios al servidor

### Ver que subió correctamente:
```bash
git log --oneline
```

**¿Qué hace?**
- Muestra últimos commits
- Cada línea = un commit
- Ejemplo:
  ```
  a1b2c3d Add complete CI/CD pipeline
  f4e5d6c Initial commit
  ```

---

## 🔄 OPCIÓN B: PULL REQUESTS (Cambios graduales)

Útil si trabajas en equipo:

```bash
# 1. Crear rama nueva para tus cambios
git checkout -b add-ci-cd-pipeline
```

**¿Qué hace?**
- Crea rama nueva basada en main
- Ahora estás en esa rama

```bash
# 2. Copiar archivos a esta rama
cp -r /ruta/a/archivos/* .

# 3. Verificar cambios
git status
git diff  # Ver exactamente qué cambió
```

```bash
# 4. Agregar cambios
git add .

# 5. Commit
git commit -m "Add CI/CD pipeline configuration"

# 6. Subir rama
git push origin add-ci-cd-pipeline
```

```bash
# 7. Crear Pull Request en GitHub (vía web)
```

**¿Qué hace un PR?**
- Propone cambios a la rama principal
- Permite que otros revisen antes de aceptar
- Ejecuta tests automáticamente
- Si todo está bien, haces "merge" (fusiona)

---

## 🔐 PASO 5: CONFIGURAR SECRETS (CONTRASEÑAS/TOKENS)

GitHub tiene variables secretas para cosas como:
- Tokens de PyPI
- Webhooks de Discord
- Claves de API

### Vía command line (más rápido):

```bash
# 1. Instalar GitHub CLI si no lo tienes
# macOS: brew install gh
# Linux: apt install gh
# Windows: choco install gh

# 2. Autenticarte
gh auth login
# Sigue los pasos que te pide

# 3. Agregar un secret
gh secret set DISCORD_WEBHOOK --body "https://discord.com/api/webhooks/..."
```

**¿Qué hace?**
- Guarda el secret en GitHub
- Lo usa automáticamente en los workflows
- No se ve en el código público

### Vía web (GitHub.com):

1. Ve a tu repo
2. Settings → Secrets and variables → Actions
3. "New repository secret"
4. Name: `DISCORD_WEBHOOK`
5. Value: `https://discord.com/api/webhooks/...`
6. Click "Add secret"

---

## 📊 RESUMEN DE FLUJO COMPLETO

```
1. CLONAR
   git clone https://github.com/...
   cd carpeta

2. VER CAMBIOS
   git status
   git diff

3. PREPARAR CAMBIOS
   git add .
   git commit -m "descripción"

4. SUBIR
   git push origin main

5. VER RESULTADO
   GitHub Actions ejecuta automáticamente:
   ├── Tests (test.yml)
   ├── Seguridad (security.yml)
   ├── Build Docker (deploy.yml)
   ├── Documentación (docs.yml)
   └── Notificaciones (notify.yml)

6. SI TODO OK
   ✅ Tests pasan
   ✅ Documentación generada
   ✅ Docker image disponible
```

---

## 🎯 COMANDOS MÁS USADOS

```bash
# CLONAR PROYECTO
git clone https://github.com/usuario/repo.git

# VER ESTADO
git status

# VER CAMBIOS ESPECÍFICOS
git diff archivo.py

# AGREGAR CAMBIOS
git add .              # Todo
git add archivo.py     # Solo un archivo

# CREAR COMMIT
git commit -m "Mensaje descriptivo"

# SUBIR CAMBIOS
git push origin main   # A rama main
git push origin rama   # A otra rama

# VER HISTORIAL
git log --oneline      # Resumen
git log                # Detallado

# CREAR RAMA
git checkout -b nueva-rama

# CAMBIAR RAMA
git checkout main

# MERGE (FUSIONAR)
git merge otra-rama

# DESCARTAR CAMBIOS
git restore archivo.py         # Un archivo
git reset --hard origin/main   # Todo (CUIDADO)
```

---

## ⚠️ TIPS IMPORTANTES

### NO hagas esto:
```bash
# ❌ NO subas archivos grandes
git add *.pkl
git add *.tar.gz

# ❌ NO subas secretos
git add .env
git add config_con_contraseñas.py

# ❌ NO fuerces push (solo si sabes qué haces)
git push -f origin main
```

### SÍ haz esto:
```bash
# ✅ Verifica antes de subir
git status
git diff

# ✅ Usa mensajes descriptivos
git commit -m "Add feature X that does Y"

# ✅ Haz commits pequeños y frecuentes
git commit -m "Add tests for module X"
git commit -m "Configure Docker"
git commit -m "Add documentation"

# ✅ Sube con frecuencia
git push origin main
# En lugar de: todo junto al final
```

---

## 🆘 PROBLEMAS COMUNES

### "fatal: destination path 'folder' already exists"
```bash
# La carpeta ya existe, entra a ella:
cd carpeta
# O elimínala y clona de nuevo:
rm -rf carpeta
git clone https://github.com/...
```

### "Permission denied" en push
```bash
# Verifica que tengas permisos:
gh auth login
# O usa SSH:
git remote set-url origin git@github.com:usuario/repo.git
```

### "Your branch is ahead of 'origin/main'"
```bash
# Tienes commits que no subiste:
git push origin main
```

### "merge conflict"
```bash
# Alguien más cambió el mismo archivo:
git status
# Edita el archivo (verás <<<< >>>> ====)
git add archivo.py
git commit -m "Resolve merge conflict"
git push origin main
```

---

Created with ❤️ for practical Git learning
