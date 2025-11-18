# 📥 Descargar Dataset en el Notebook

## Opción 1: Descarga Automática desde Notebook (RECOMENDADO)

Copia esta celda al **principio del notebook** (antes de cargar datos):

```python
# =============================================================================
# DESCARGAR DATASET MOVIELENS 100K (Ejecutar solo una vez)
# =============================================================================

import os
import urllib.request
import zipfile
from pathlib import Path

print("=" * 70)
print("📥 DESCARGANDO DATASET MOVIELENS 100K")
print("=" * 70)

# Crear directorios
NOTEBOOK_DIR = Path.cwd()
PROJECT_DIR = NOTEBOOK_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw' / 'ml-100k'

# Crear carpeta si no existe
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Verificar si dataset ya existe
if (DATA_DIR / 'u.data').exists():
    print(f"✅ Dataset ya existe en: {DATA_DIR}")
else:
    print(f"📥 Descargando a: {DATA_DIR}")

    # URL del dataset
    URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    ZIP_FILE = DATA_DIR.parent / "ml-100k.zip"

    try:
        # Descargar
        print("⏳ Descargando ml-100k.zip (~5MB)...")
        urllib.request.urlretrieve(URL, ZIP_FILE)
        print("✅ Descarga completada")

        # Extraer
        print("⏳ Extrayendo archivos...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR.parent)

        # Mover archivos
        ml100k_dir = DATA_DIR.parent / 'ml-100k'
        if ml100k_dir.exists():
            for file in ml100k_dir.glob('*'):
                file.rename(DATA_DIR / file.name)
            ml100k_dir.rmdir()

        # Limpiar ZIP
        ZIP_FILE.unlink()

        print(f"✅ Dataset descargado y extraído correctamente")
        print(f"📁 Ubicación: {DATA_DIR}")

    except Exception as e:
        print(f"❌ Error descargando: {e}")
        print("💡 Intenta descargarlo manualmente desde:")
        print("   http://files.grouplens.org/datasets/movielens/ml-100k.zip")

print()
```

---

## Opción 2: Usando `!bash` (Comando del Sistema)

Si solo necesitas ejecutar el script Python, usa `!` para ejecutar comandos:

```python
# En una celda del notebook, ejecuta:
!cd .. && python download_movielens.py
```

O así:

```python
import os
os.system('python download_movielens.py')
```

---

## Opción 3: Descargar Manualmente

Si nada funciona, descarga manualmente:

1. **Descarga el archivo:**
   - Ve a: http://files.grouplens.org/datasets/movielens/ml-100k.zip
   - Descarga el ZIP

2. **Extrae el ZIP:**
   - Extrae a: `amazone/data/raw/ml-100k/`
   - Debe haber archivos como: `u.data`, `u.item`, `u.user`, etc.

3. **Verifica la estructura:**
   ```
   amazone/
   └── data/
       └── raw/
           └── ml-100k/
               ├── u.data
               ├── u.item
               ├── u.user
               ├── u.genre
               └── ...
   ```

---

## ✅ Verificar que el Dataset Existe

Ejecuta esta celda para verificar:

```python
from pathlib import Path

NOTEBOOK_DIR = Path.cwd()
PROJECT_DIR = NOTEBOOK_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw' / 'ml-100k'

# Verificar archivos
required_files = ['u.data', 'u.item', 'u.user']

print("📂 Verificando dataset...")
for file in required_files:
    file_path = DATA_DIR / file
    exists = "✅" if file_path.exists() else "❌"
    print(f"{exists} {file}: {file_path}")

if (DATA_DIR / 'u.data').exists():
    print("\n✅ Dataset listo para usar!")
else:
    print("\n❌ Dataset no encontrado. Por favor descárgalo primero.")
```

---

## 📋 Pasos Recomendados

1. **Copia la celda de "Opción 1"** al notebook
2. **Ejecuta la celda** (tarda ~30-60 segundos)
3. **Verifica con la celda de verificación**
4. **Continúa con el resto del notebook**

---

## 💡 Si el Descarga es Lenta

Si Internet es lento, puedes:

- Descargar en otra máquina y transferir
- Usar un cliente de descarga más rápido
- Usar una VPN si hay restricciones geográficas
- Usar un espejo de datos (si está disponible)

---

## 🆘 Solución de Problemas

### "No se puede descargar"
- Verifica tu conexión a Internet
- Intenta con un VPN
- Descarga manualmente desde el navegador

### "Permiso denegado"
- Asegúrate que tienes permisos en la carpeta `amazone/`
- Intenta crear la carpeta manualmente primero

### "Archivo corrupto"
- Descarga de nuevo
- Elimina la carpeta `ml-100k/` y vuelve a intentar

---

## ✨ Nota Final

El notebook está **100% autocontendido**. Una vez que ejecutes la celda de descarga una sola vez, el dataset estará disponible para todas las futuras ejecuciones.

**¡Así que no necesitas Python instalado globalmente!** 🎉
