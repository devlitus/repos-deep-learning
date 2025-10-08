"""
Configuración centralizada del proyecto de Detección de Fraude
"""
import os
from pathlib import Path

# ============================================
# RUTAS DEL PROYECTO
# ============================================
# Raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Rutas de modelos
MODELS_DIR = BASE_DIR / "models"

# Rutas de reportes
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Crear directorios si no existen
for directory in [PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================
# ARCHIVOS
# ============================================
RAW_DATA_FILE = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "creditcard_processed.csv"

# ============================================
# PARÁMETROS DEL MODELO
# ============================================
# Semilla aleatoria para reproducibilidad
RANDOM_STATE = 42

# División de datos
TEST_SIZE = 0.2  # 20% para test, 80% para train
VALIDATION_SIZE = 0.2  # 20% del train para validación

# Parámetros de balanceo
SAMPLING_STRATEGY = 0.5  # Ratio de balanceo para SMOTE

# ============================================
# PARÁMETROS DE VISUALIZACIÓN
# ============================================
FIGURE_SIZE = (12, 6)
STYLE = 'whitegrid'

# ============================================
# COLUMNAS DEL DATASET
# ============================================
TARGET_COLUMN = 'Class'
TIME_COLUMN = 'Time'
AMOUNT_COLUMN = 'Amount'

# Nombres de clases
CLASS_NAMES = {
    0: 'Legítima',
    1: 'Fraude'
}