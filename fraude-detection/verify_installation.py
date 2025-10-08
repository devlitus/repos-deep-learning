"""
Script de verificación de instalación del proyecto de Detección de Fraude
Verifica que todos los módulos y dependencias estén correctamente configurados
"""
import sys
from pathlib import Path

# Agregar raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

print("="*70)
print("  VERIFICACIÓN DE INSTALACIÓN - PROYECTO DETECCIÓN DE FRAUDE")
print("="*70)

# Lista de verificaciones
checks = []

# 1. Verificar estructura de directorios
print("\n1. Verificando estructura de directorios...")
required_dirs = [
    'config',
    'data/raw',
    'data/processed',
    'models',
    'reports/figures',
    'src/data',
    'src/models',
    'src/visualization',
    'notebooks'
]

for dir_path in required_dirs:
    full_path = project_root / dir_path
    exists = full_path.exists()
    status = "[OK]" if exists else "[X]"
    print(f"   {status} {dir_path}")
    checks.append(("Directorio " + dir_path, exists))

# 2. Verificar archivos principales
print("\n2. Verificando archivos principales...")
required_files = [
    'main.py',
    'requirements.txt',
    'config/config.py',
    'src/data/load.py',
    'src/data/preprocess.py',
    'src/models/train.py',
    'src/models/evaluate.py',
    'src/visualization/plots.py'
]

for file_path in required_files:
    full_path = project_root / file_path
    exists = full_path.exists()
    status = "[OK]" if exists else "[X]"
    print(f"   {status} {file_path}")
    checks.append(("Archivo " + file_path, exists))

# 3. Verificar importaciones básicas
print("\n3. Verificando importaciones de módulos...")
modules_to_test = [
    ('config.config', 'Configuración del proyecto'),
    ('src.data.load', 'Módulo de carga de datos'),
    ('src.models.train', 'Módulo de entrenamiento'),
]

for module_name, description in modules_to_test:
    try:
        __import__(module_name)
        print(f"   [OK] {description} ({module_name})")
        checks.append((description, True))
    except Exception as e:
        print(f"   [X] {description} ({module_name}): {str(e)[:50]}")
        checks.append((description, False))

# 4. Verificar dependencias críticas
print("\n4. Verificando dependencias Python...")
dependencies = [
    ('pandas', 'Manipulación de datos'),
    ('numpy', 'Cálculos numéricos'),
    ('sklearn', 'Machine Learning (scikit-learn)'),
    ('imblearn', 'Balanceo de clases (imbalanced-learn)'),
    ('matplotlib', 'Visualización'),
    ('seaborn', 'Visualización estadística'),
    ('joblib', 'Serialización de modelos')
]

for package, description in dependencies:
    try:
        __import__(package)
        print(f"   [OK] {description} ({package})")
        checks.append((f"Dependencia {package}", True))
    except ImportError:
        print(f"   [X] {description} ({package}) - NO INSTALADO")
        checks.append((f"Dependencia {package}", False))

# 5. Verificar dataset
print("\n5. Verificando dataset...")
from config.config import RAW_DATA_FILE
dataset_exists = RAW_DATA_FILE.exists()
status = "[OK]" if dataset_exists else "[X]"
print(f"   {status} Dataset creditcard.csv en {RAW_DATA_FILE}")
if not dataset_exists:
    print(f"      -> Descargar de: https://www.kaggle.com/mlg-ulb/creditcardfraud")
checks.append(("Dataset creditcard.csv", dataset_exists))

# Resumen final
print("\n" + "="*70)
print("  RESUMEN DE VERIFICACIÓN")
print("="*70)

total_checks = len(checks)
passed_checks = sum(1 for _, passed in checks if passed)
failed_checks = total_checks - passed_checks

print(f"\nTotal de verificaciones: {total_checks}")
print(f"[OK] Exitosas: {passed_checks}")
print(f"[X] Fallidas: {failed_checks}")

if failed_checks == 0:
    print("\n*** EXCELENTE! Todos los componentes estan correctamente instalados.")
    print("   El proyecto esta listo para ejecutarse con: python main.py")
else:
    print(f"\n*** ATENCION: {failed_checks} verificaciones fallaron.")
    print("   Acciones recomendadas:")
    if any(not passed for (name, passed) in checks if 'Dependencia' in name):
        print("   1. Instalar dependencias: pip install -r requirements.txt")
    if not dataset_exists:
        print("   2. Descargar el dataset de Kaggle y colocarlo en data/raw/")
    print("   3. Revisar los mensajes de error arriba para mas detalles")

print("\n" + "="*70)

sys.exit(0 if failed_checks == 0 else 1)
