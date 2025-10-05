# config.py
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Archivos
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, 'casas.csv')
MODEL_FILE = os.path.join(MODELS_DIR, 'modelo_casas.pkl')

# Parámetros del modelo
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURES = ['tamano_m2', 'habitaciones', 'banos', 'edad_anos', 'distancia_centro_km']
TARGET = 'precio'