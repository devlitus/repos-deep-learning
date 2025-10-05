# config.py
import os

# Rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Archivos - Dataset simple
RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, 'casas.csv')

# Archivos - Dataset Kaggle
KAGGLE_TRAIN_FILE = os.path.join(DATA_RAW_DIR, 'train.csv')
KAGGLE_TEST_FILE = os.path.join(DATA_RAW_DIR, 'test.csv')

MODEL_FILE = os.path.join(MODELS_DIR, 'modelo_casas.pkl')
KAGGLE_MODEL_FILE = os.path.join(MODELS_DIR, 'modelo_kaggle.pkl')

# Parámetros del modelo
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Features dataset simple
FEATURES = ['tamano_m2', 'habitaciones', 'banos', 'edad_anos', 'distancia_centro_km']
TARGET = 'precio'

# Features dataset Kaggle (top 10 por correlación)
KAGGLE_FEATURES = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'GarageArea',
    'TotalBsmtSF',
    'FullBath',
    'TotRmsAbvGrd',
    'YearBuilt',
    'YearRemodAdd',
    'Fireplaces'
]
KAGGLE_TARGET = 'SalePrice'