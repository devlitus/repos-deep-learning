"""
Configuración central del proyecto de Análisis de Sentimientos
================================================================

Este archivo contiene todas las rutas absolutas y parámetros de configuración
siguiendo el patrón modular del repositorio.
"""

from pathlib import Path

# ============================================================================
# RUTAS ABSOLUTAS DEL PROYECTO
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
REPORTS_DIR = BASE_DIR / 'reports'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'
WEB_DIR = BASE_DIR / 'web'

# ============================================================================
# PARÁMETROS DE DATOS
# ============================================================================

# Usaremos el dataset IMDB de reviews de películas (incluido en Keras)
# 50,000 reviews: 25,000 para entrenamiento, 25,000 para prueba
DATASET_NAME = 'imdb'
NUM_WORDS = 10000  # Vocabulario: top 10,000 palabras más frecuentes
MAX_SEQUENCE_LENGTH = 200  # Longitud máxima de cada review (en palabras)

# Columnas para datasets personalizados (si usamos CSV)
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'sentiment'  # 0 = negativo, 1 = positivo

# ============================================================================
# PREPROCESAMIENTO DE TEXTO (Conceptos nuevos para ti)
# ============================================================================

# STOPWORDS: Palabras sin significado semántico ("el", "la", "de", "a")
# Las eliminaremos porque no aportan información sobre el sentimiento
REMOVE_STOPWORDS = True
LANGUAGE = 'english'  # Para stopwords

# STEMMING vs LEMMATIZATION: Reducir palabras a su raíz
# "corriendo", "corrió", "corre" → "corr" (stemming) o "correr" (lemmatization)
USE_STEMMING = False  # Porter Stemmer (más agresivo)
USE_LEMMATIZATION = True  # Más preciso, mantiene palabras reales

# Caracteres a remover (puntuación, números, etc.)
REMOVE_PUNCTUATION = True
REMOVE_NUMBERS = True
REMOVE_HTML_TAGS = True
CONVERT_TO_LOWERCASE = True

# ============================================================================
# FEATURE EXTRACTION (Convertir texto a números)
# ============================================================================

# TF-IDF: Term Frequency - Inverse Document Frequency
# Mide qué tan importante es una palabra en un documento
# Ejemplo: "excelente" aparece mucho en este review pero poco en otros → alto TF-IDF
TFIDF_MAX_FEATURES = 5000  # Top 5,000 palabras más relevantes

# Word Embeddings: Representación vectorial de palabras
# Palabras similares tienen vectores similares
# "excelente" y "bueno" estarán cerca en el espacio vectorial
EMBEDDING_DIM = 100  # Dimensión del vector para cada palabra
USE_PRETRAINED_EMBEDDINGS = False  # GloVe o Word2Vec pre-entrenados

# ============================================================================
# DIVISIÓN DE DATOS
# ============================================================================
TEST_SIZE = 0.2  # 20% para prueba
VALIDATION_SIZE = 0.1  # 10% para validación
RANDOM_STATE = 42  # Reproducibilidad

# ============================================================================
# MODELOS CLÁSICOS DE ML (Scikit-learn)
# ============================================================================

# NAIVE BAYES: Probabilístico, muy rápido para clasificación de texto
# Asume independencia entre palabras (naive = ingenuo)
NAIVE_BAYES_ALPHA = 1.0  # Suavizado de Laplace

# SUPPORT VECTOR MACHINE (SVM): Encuentra hiperplano óptimo
SVM_C = 1.0  # Parámetro de regularización
SVM_KERNEL = 'linear'  # 'linear' funciona bien con TF-IDF

# ============================================================================
# MODELO DEEP LEARNING (LSTM para texto)
# ============================================================================

# LSTM: Long Short-Term Memory
# Similar a prediccion-temperatura, pero con secuencias de palabras
LSTM_UNITS_1 = 128  # Primera capa LSTM
LSTM_UNITS_2 = 64   # Segunda capa LSTM
DROPOUT_RATE = 0.5  # Prevenir overfitting (50% de neuronas desactivadas)
RECURRENT_DROPOUT = 0.2  # Dropout en conexiones recurrentes

DENSE_UNITS = 64  # Capa densa antes de la salida
ACTIVATION_HIDDEN = 'relu'
ACTIVATION_OUTPUT = 'sigmoid'  # Clasificación binaria (0 o 1)

OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'  # Para clasificación binaria
METRICS = ['accuracy', 'precision', 'recall']

BATCH_SIZE = 64
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3  # Detener si no mejora en 3 épocas

# ============================================================================
# RUTAS DE MODELOS GUARDADOS
# ============================================================================
MODEL_TFIDF_SVM = MODELS_DIR / 'tfidf_svm_model.pkl'
MODEL_TFIDF_NAIVE_BAYES = MODELS_DIR / 'tfidf_nb_model.pkl'
MODEL_LSTM = MODELS_DIR / 'lstm_sentiment_model.keras'
TOKENIZER_FILE = MODELS_DIR / 'tokenizer.pkl'  # Keras Tokenizer guardado
TFIDF_VECTORIZER_FILE = MODELS_DIR / 'tfidf_vectorizer.pkl'

# ============================================================================
# VISUALIZACIONES
# ============================================================================
WORDCLOUD_WIDTH = 800
WORDCLOUD_HEIGHT = 400
WORDCLOUD_BACKGROUND = 'white'

# Colores para gráficos
COLOR_POSITIVE = '#2ecc71'  # Verde
COLOR_NEGATIVE = '#e74c3c'  # Rojo

# ============================================================================
# CONFIGURACIÓN DE REPORTES
# ============================================================================
REPORT_METRICS_FILE = REPORTS_DIR / 'model_metrics.json'
REPORT_CONFUSION_MATRIX = REPORTS_DIR / 'confusion_matrix.png'
REPORT_WORDCLOUD_POSITIVE = REPORTS_DIR / 'wordcloud_positive.png'
REPORT_WORDCLOUD_NEGATIVE = REPORTS_DIR / 'wordcloud_negative.png'
REPORT_TRAINING_HISTORY = REPORTS_DIR / 'training_history.png'

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

print("✅ Configuración cargada exitosamente")
print(f"📁 Directorio base: {BASE_DIR}")
print(f"🗂️  Dataset: {DATASET_NAME.upper()}")
print(f"🧠 Vocabulario: top {NUM_WORDS} palabras")
print(f"📏 Longitud máxima de secuencia: {MAX_SEQUENCE_LENGTH} palabras")
