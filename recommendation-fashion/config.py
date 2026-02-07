"""
Configuración centralizada del proyecto - Amazon Fashion Recommendations
Define todas las rutas, parámetros y configuraciones globales

Siguiendo el patrón de fraude-detection con pathlib.Path
"""

import sys
import io
from pathlib import Path

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# =====================================
# RUTAS BASE DEL PROYECTO
# =====================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
WEB_DIR = BASE_DIR / 'web'
SRC_DIR = BASE_DIR / 'src'

# Crear directorios si no existen
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =====================================
# ARCHIVOS DE DATOS
# =====================================

# Dataset raw (descargado)
DATASET_FILE = DATA_RAW_DIR / 'fashion_reviews.json'
DATASET_GZIP_FILE = DATA_RAW_DIR / 'fashion_reviews.json.gz'

# Datos procesados
PROCESSED_INTERACTIONS_FILE = DATA_PROCESSED_DIR / 'interactions.pkl'
PROCESSED_USER_ITEM_MATRIX = DATA_PROCESSED_DIR / 'user_item_matrix.pkl'
PROCESSED_SIMILARITY_MATRIX_USER = DATA_PROCESSED_DIR / 'user_similarity_matrix.pkl'
PROCESSED_SIMILARITY_MATRIX_ITEM = DATA_PROCESSED_DIR / 'item_similarity_matrix.pkl'
PROCESSED_TRAIN_TEST = DATA_PROCESSED_DIR / 'train_test_split.pkl'

# =====================================
# ARCHIVOS DE MODELOS
# =====================================

# Modelos de collaborative filtering
SVD_MODEL_FILE = MODELS_DIR / 'svd_model.pkl'
USER_SIMILARITY_FILE = MODELS_DIR / 'user_similarity.pkl'
ITEM_SIMILARITY_FILE = MODELS_DIR / 'item_similarity.pkl'
HYBRID_MODEL_FILE = MODELS_DIR / 'hybrid_model.pkl'

# Matrices sparse (para datasets grandes)
USER_ITEM_SPARSE_MATRIX = MODELS_DIR / 'user_item_sparse.npz'

# =====================================
# ARCHIVOS DE REPORTES
# =====================================

# Análisis exploratorio
REPORT_RATINGS_DIST = REPORTS_DIR / 'rating_distribution.png'
REPORT_TOP_USERS = REPORTS_DIR / 'top_users.png'
REPORT_TOP_PRODUCTS = REPORTS_DIR / 'top_products.png'
REPORT_DISTRIBUTION_HISTOGRAM = REPORTS_DIR / 'distribution_histogram.png'

# Collaborative Filtering
REPORT_USER_SIMILARITY_DIST = REPORTS_DIR / 'user_similarity_distribution.png'
REPORT_ITEM_SIMILARITY_DIST = REPORTS_DIR / 'item_similarity_distribution.png'
REPORT_USER_BASED_PREDICTIONS = REPORTS_DIR / 'user_based_predictions.png'
REPORT_ITEM_BASED_PREDICTIONS = REPORTS_DIR / 'item_based_predictions.png'

# SVD y Matrix Factorization
REPORT_SVD_SINGULAR_VALUES = REPORTS_DIR / 'svd_singular_values.png'
REPORT_SVD_PREDICTIONS = REPORTS_DIR / 'svd_predictions.png'
REPORT_SVD_LATENT_SPACE = REPORTS_DIR / 'svd_latent_space.png'
REPORT_SVD_HEATMAP = REPORTS_DIR / 'svd_heatmap_comparison.png'

# Sistema Híbrido
REPORT_HYBRID_PREDICTIONS = REPORTS_DIR / 'hybrid_predictions.png'
REPORT_HYBRID_COMPARISON = REPORTS_DIR / 'hybrid_comparison.png'

# Deep Learning (para futuras implementaciones)
REPORT_DL_TRAINING_CURVES = REPORTS_DIR / 'deep_learning_training_curves.png'
REPORT_DL_PREDICTIONS = REPORTS_DIR / 'deep_learning_predictions.png'

# EDA adicional
REPORT_EDA_USERS_DIST = REPORTS_DIR / 'eda_users_distribution.png'
REPORT_EDA_PRODUCTS_DIST = REPORTS_DIR / 'eda_products_distribution.png'
REPORT_EDA_RATINGS_DIST = REPORTS_DIR / 'eda_ratings_distribution.png'

# Métricas y evaluación
METRICS_FILE = REPORTS_DIR / 'metrics.json'
EVALUATION_FILE = REPORTS_DIR / 'evaluation.csv'
USER_BASED_CF_EVALUATION = REPORTS_DIR / 'user_based_cf_evaluation.png'

# =====================================
# PARÁMETROS DE DATOS
# =====================================

# Columnas del dataset
COL_USER_ID = 'reviewerID'
COL_PRODUCT_ID = 'asin'
COL_RATING = 'overall'
COL_REVIEW_TEXT = 'reviewText'
COL_SUMMARY = 'summary'
COL_TIMESTAMP = 'unixReviewTime'
COL_VERIFIED = 'verified'

# Filtrado de datos (para reducir sparsity)
MIN_USER_RATINGS = 5  # Usuarios con al menos N reviews
MIN_PRODUCT_RATINGS = 5  # Productos con al menos N reviews
MIN_RATING_SCORE = 1.0  # Rating mínimo válido
MAX_RATING_SCORE = 5.0  # Rating máximo válido

# Muestreo para análisis rápido (None = usar todos los datos)
SAMPLE_SIZE = None  # Útil para pruebas rápidas: 10000, 50000, etc.
SAMPLE_RANDOM_STATE = 42

# División de datos
TEST_SIZE = 0.2  # Proporción para conjunto de prueba (20%)
VALIDATION_SIZE = 0.1  # Proporción para validación (10%)
RANDOM_STATE = 42  # Para reproducibilidad

# =====================================
# PARÁMETROS DE COLLABORATIVE FILTERING
# =====================================

# User-Based Collaborative Filtering
USER_CF_K_NEIGHBORS = 20  # Número de usuarios similares para recomendar
USER_CF_SIMILARITY_METRIC = 'cosine'  # 'cosine', 'pearson', 'jaccard'
USER_CF_MIN_SUPPORT = 3  # Mínimo de items en común para calcular similitud

# Item-Based Collaborative Filtering
ITEM_CF_K_NEIGHBORS = 20  # Número de productos similares
ITEM_CF_SIMILARITY_METRIC = 'cosine'
ITEM_CF_MIN_SUPPORT = 3  # Mínimo de usuarios en común

# =====================================
# PARÁMETROS DE SVD (Matrix Factorization)
# =====================================

SVD_K_FACTORS = 50  # Número de factores latentes (dimensiones ocultas)
SVD_MIN_RATING = 1.0  # Rating mínimo en el dataset
SVD_MAX_RATING = 5.0  # Rating máximo en el dataset
SVD_DAMPING = 5  # Para dampening en predicciones (suavizado)
SVD_REGULARIZATION = 0.02  # Regularización para evitar overfitting

# =====================================
# PARÁMETROS DE SISTEMA HÍBRIDO
# =====================================

# Pesos de combinación (deben sumar 1.0)
HYBRID_WEIGHT_USER_CF = 0.3
HYBRID_WEIGHT_ITEM_CF = 0.3
HYBRID_WEIGHT_SVD = 0.4

# Estrategia de combinación: 'weighted', 'voting', 'cascade'
HYBRID_STRATEGY = 'weighted'

# =====================================
# PARÁMETROS DE EVALUACIÓN
# =====================================

# Top-N recommendations
TOP_N = 10
TOP_N_OPTIONS = [5, 10, 20, 50]  # Para análisis de sensibilidad
METRICS_AT_K = [5, 10, 20]

# Métricas a calcular
METRICS = [
    'RMSE',       # Root Mean Squared Error
    'MAE',        # Mean Absolute Error
    'Precision',  # Precisión en Top-N
    'Recall',     # Recall en Top-N
    'F1',         # F1-Score
    'Coverage',   # Cobertura del catálogo
    'Diversity'   # Diversidad de recomendaciones
]

# =====================================
# PARÁMETROS DE DEEP LEARNING
# =====================================

# Neural Collaborative Filtering (NCF)
DL_EMBEDDING_SIZE = 64  # Dimensión de embeddings de usuarios/productos
DL_HIDDEN_LAYERS = [128, 64, 32]  # Capas ocultas del MLP
DL_DROPOUT_RATE = 0.2  # Dropout para regularización
DL_LEARNING_RATE = 0.001  # Learning rate inicial
DL_BATCH_SIZE = 256  # Tamaño de batch para entrenamiento
DL_EPOCHS = 10  # Número de épocas (con early stopping)
DL_EARLY_STOPPING_PATIENCE = 5  # Paciencia para early stopping
DL_WEIGHT_DECAY = 1e-5  # Regularización L2
DL_GRADIENT_CLIP = 1.0  # Gradient clipping

# Deep Hybrid Recommender
DEEP_HYBRID_MODEL_FILE = MODELS_DIR / 'deep_hybrid_model.pth'
DEEP_HYBRID_ATTENTION_EPOCHS = 10  # Épocas para fine-tuning de atención
DEEP_HYBRID_ATTENTION_LR = 0.0001  # Learning rate para attention (más bajo)

# =====================================
# PARÁMETROS DE VISUALIZACIÓN
# =====================================

# Tamaños de figura
PLOT_FIGSIZE = (12, 6)
PLOT_FIGSIZE_LARGE = (16, 8)
PLOT_FIGSIZE_SQUARE = (10, 10)
PLOT_DPI = 100

# Estilo de gráficos
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_PALETTE = 'husl'
PLOT_CONTEXT = 'notebook'  # 'paper', 'notebook', 'talk', 'poster'

# Colores personalizados
COLOR_PRIMARY = '#2E86AB'
COLOR_SECONDARY = '#A23B72'
COLOR_ACCENT = '#F18F01'
COLOR_SUCCESS = '#06A77D'
COLOR_WARNING = '#F77F00'
COLOR_DANGER = '#D62828'

# =====================================
# PARÁMETROS DE WEB APP (Streamlit)
# =====================================

STREAMLIT_PAGE_CONFIG = {
    'page_title': '👕 Fashion Recommendations',
    'page_icon': '👕',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Configuración de cache
STREAMLIT_CACHE_TTL = 3600  # 1 hora en segundos

# =====================================
# PARÁMETROS DE LOGGING Y DEBUG
# =====================================

DEBUG_MODE = False
VERBOSE = True
LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
SHOW_PROGRESS_BARS = True

# =====================================
# URLS Y REFERENCIAS
# =====================================

# Dataset sources
HUGGING_FACE_DATASET = "Kuaipai/amazon_fashion_reviews"
# URL actualizada del dataset de Amazon Fashion (2014)
AMAZON_REVIEWS_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_v2/categoryFilesSmall/AMAZON_FASHION.json.gz"
# Alternativa: "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Clothing_Shoes_and_Jewelry_5.json.gz"

# Referencias
DATASET_REFERENCE = "Justifying recommendations using distantly-labeled reviews and fined-grained aspects (Jianmo Ni, EMNLP 2019)"
PROJECT_REPO = "https://github.com/tu-usuario/repos-deep-learning"

# =====================================
# CONSTANTES
# =====================================

# Categorías de productos (para análisis futuro)
FASHION_CATEGORIES = [
    'Clothing',
    'Shoes',
    'Jewelry',
    'Accessories',
    'Watches'
]

# Rangos de ratings
RATING_SCALE = {
    'min': 1.0,
    'max': 5.0,
    'labels': ['Muy malo', 'Malo', 'Regular', 'Bueno', 'Excelente']
}

# =====================================
# VALIDACIÓN DE CONFIGURACIÓN
# =====================================

def validate_config():
    """Valida que la configuración sea correcta"""
    issues = []
    warnings = []

    # Verificar directorios críticos
    if not BASE_DIR.exists():
        issues.append(f"❌ BASE_DIR no existe: {BASE_DIR}")

    if not DATA_DIR.exists():
        issues.append(f"❌ DATA_DIR no existe: {DATA_DIR}")

    # Verificar dataset
    if not DATASET_FILE.exists():
        warnings.append(f"⚠️  Dataset no encontrado: {DATASET_FILE}")
        warnings.append("   Ejecuta: python src/download_fashion.py")

    # Verificar pesos del sistema híbrido
    total_weight = HYBRID_WEIGHT_USER_CF + HYBRID_WEIGHT_ITEM_CF + HYBRID_WEIGHT_SVD
    if abs(total_weight - 1.0) > 0.01:
        issues.append(f"❌ Los pesos del sistema híbrido deben sumar 1.0 (suma actual: {total_weight})")

    # Verificar rangos válidos
    if MIN_RATING_SCORE < 1.0 or MAX_RATING_SCORE > 5.0:
        issues.append("❌ Rango de ratings inválido (debe estar entre 1.0 y 5.0)")

    if TEST_SIZE + VALIDATION_SIZE >= 1.0:
        issues.append("❌ TEST_SIZE + VALIDATION_SIZE debe ser menor que 1.0")

    return len(issues) == 0, issues, warnings

def print_config_summary():
    """Imprime un resumen de la configuración"""
    print("\n" + "=" * 70)
    print("  📋 CONFIGURACIÓN DEL PROYECTO")
    print("=" * 70)

    print("\n📁 Directorios:")
    print(f"  - Base: {BASE_DIR}")
    print(f"  - Datos raw: {DATA_RAW_DIR}")
    print(f"  - Datos procesados: {DATA_PROCESSED_DIR}")
    print(f"  - Modelos: {MODELS_DIR}")
    print(f"  - Reportes: {REPORTS_DIR}")

    print("\n📊 Parámetros de datos:")
    print(f"  - Min user ratings: {MIN_USER_RATINGS}")
    print(f"  - Min product ratings: {MIN_PRODUCT_RATINGS}")
    print(f"  - Test size: {TEST_SIZE * 100:.0f}%")
    print(f"  - Random state: {RANDOM_STATE}")

    print("\n🤝 Collaborative Filtering:")
    print(f"  - User-based neighbors: {USER_CF_K_NEIGHBORS}")
    print(f"  - Item-based neighbors: {ITEM_CF_K_NEIGHBORS}")
    print(f"  - Similarity metric: {USER_CF_SIMILARITY_METRIC}")

    print("\n📐 SVD:")
    print(f"  - Factores latentes: {SVD_K_FACTORS}")
    print(f"  - Regularización: {SVD_REGULARIZATION}")

    print("\n🎯 Sistema Híbrido:")
    print(f"  - Peso User-CF: {HYBRID_WEIGHT_USER_CF}")
    print(f"  - Peso Item-CF: {HYBRID_WEIGHT_ITEM_CF}")
    print(f"  - Peso SVD: {HYBRID_WEIGHT_SVD}")
    print(f"  - Estrategia: {HYBRID_STRATEGY}")

    print("\n📈 Evaluación:")
    print(f"  - Top-N: {TOP_N}")
    print(f"  - Métricas: {', '.join(METRICS)}")

    print("\n" + "=" * 70)

if __name__ == '__main__':
    print("🔧 Verificando configuración...")

    success, issues, warnings = validate_config()

    if issues:
        print("\n❌ Problemas encontrados:")
        for issue in issues:
            print(f"  {issue}")

    if warnings:
        print("\n⚠️  Advertencias:")
        for warning in warnings:
            print(f"  {warning}")

    if success:
        print("\n✅ Configuración válida")
        print_config_summary()
    else:
        print("\n❌ Configuración inválida. Corrige los errores antes de continuar.")
        exit(1)
