# config.py
import os

# Rutas absolutas siguiendo el patrón del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# Archivos de modelo
MODEL_FILE = os.path.join(MODELS_DIR, 'titanic_random_forest.pkl')

# Features y target para Titanic
# Importante: Estas son las features que el modelo espera (en orden)
FEATURES = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'family_size', 'is_alone']
TARGET = 'survived'

# Parámetros de división train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Parámetros del modelo Random Forest
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_STATE
}