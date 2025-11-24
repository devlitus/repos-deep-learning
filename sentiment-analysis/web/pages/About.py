"""
About - Información del Proyecto
=================================

Información sobre el proyecto de análisis de sentimientos.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import config

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="About - Sentiment Analysis",
    page_icon="ℹ️",
    layout="wide"
)

# ============================================================================
# HEADER
# ============================================================================

st.title("ℹ️ Acerca del Proyecto")
st.markdown("---")

# ============================================================================
# DESCRIPCIÓN GENERAL
# ============================================================================

st.header("🎬 Sentiment Analysis - IMDB Reviews")

st.markdown("""
Este proyecto implementa un sistema completo de **análisis de sentimientos** usando
técnicas de **Deep Learning** y **Natural Language Processing (NLP)**.

El objetivo es clasificar automáticamente reseñas de películas como **positivas** o **negativas**
analizando el texto de las mismas.
""")

# ============================================================================
# CARACTERÍSTICAS
# ============================================================================

st.markdown("---")
st.header("✨ Características Principales")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🧠 Deep Learning")
    st.markdown("""
    - Modelo LSTM (Long Short-Term Memory)
    - Arquitectura optimizada (87% accuracy)
    - Embeddings de 64 dimensiones
    - Entrenado con 50,000 reviews
    """)

with col2:
    st.subheader("📊 Métricas Completas")
    st.markdown("""
    - Accuracy: 87.1%
    - AUC-ROC: 93.3%
    - Precision: 86.8%
    - Recall: 87.1%
    """)

with col3:
    st.subheader("🚀 Interfaz Interactiva")
    st.markdown("""
    - Predicciones en tiempo real
    - Análisis de confianza
    - Visualizaciones intuitivas
    - Ejemplos predefinidos
    """)

# ============================================================================
# TECNOLOGÍAS
# ============================================================================

st.markdown("---")
st.header("🛠️ Stack Tecnológico")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Machine Learning")
    st.markdown("""
    - **TensorFlow/Keras**: Framework de Deep Learning
    - **Scikit-learn**: Preprocesamiento y métricas
    - **NLTK**: Procesamiento de lenguaje natural
    - **NumPy**: Computación numérica
    - **Pandas**: Manipulación de datos
    """)

    st.subheader("Modelos Implementados")
    st.markdown("""
    1. **LSTM Neural Network** (principal)
    2. **SVM con TF-IDF**
    3. **Naive Bayes con TF-IDF**
    """)

with col2:
    st.subheader("Web & Visualización")
    st.markdown("""
    - **Streamlit**: Framework web interactivo
    - **Matplotlib**: Gráficos estáticos
    - **Seaborn**: Visualizaciones estadísticas
    - **Plotly**: Gráficos interactivos
    """)

    st.subheader("Desarrollo")
    st.markdown("""
    - **Python 3.10+**
    - **Jupyter Notebooks**: Experimentación
    - **Git**: Control de versiones
    - **VS Code**: IDE
    """)

# ============================================================================
# DATASET
# ============================================================================

st.markdown("---")
st.header("📁 Dataset: IMDB Movie Reviews")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    El **IMDB Dataset** es uno de los datasets más populares para análisis de sentimientos:

    - **50,000 reseñas** de películas de IMDB
    - **Balanceado**: 25,000 positivas + 25,000 negativas
    - **División**: 50% entrenamiento, 50% test
    - **Idioma**: Inglés
    - **Longitud promedio**: 239 palabras por reseña

    ### ¿Por qué IMDB?

    - Ampliamente usado en investigación académica
    - Reseñas auténticas de usuarios reales
    - Sentimientos claramente marcados (≥7 stars = positivo, ≤4 stars = negativo)
    - Vocabulario rico y variado
    - Permite comparar con otros trabajos de NLP
    """)

with col2:
    st.info("""
    **📊 Estadísticas**

    - 25,000 train
    - 25,000 test
    - 10,000 vocabulario
    - 300 palabras max
    - 64 dim embeddings
    """)

    st.success("""
    **✅ Calidad**

    - Reseñas reales
    - Sin spam/bots
    - Sentimientos claros
    - Texto limpio
    """)

# ============================================================================
# PIPELINE
# ============================================================================

st.markdown("---")
st.header("🔄 Pipeline de Procesamiento")

st.markdown("""
```
1. CARGA DE DATOS
   ├── Dataset IMDB (Keras)
   └── 50,000 reviews con etiquetas

2. PREPROCESAMIENTO
   ├── Lowercase conversion
   ├── Tokenización
   ├── Eliminación de stopwords
   ├── Lemmatización
   └── Padding/Truncation (300 palabras)

3. FEATURE EXTRACTION
   ├── Word Embeddings (64 dim)
   └── Secuencias numéricas

4. ENTRENAMIENTO
   ├── LSTM (64 units)
   ├── Dropout (30% + 20%)
   ├── Learning rate: 0.0001
   ├── Early stopping (patience: 5)
   └── 15 épocas máximo

5. EVALUACIÓN
   ├── Test set (25,000 reviews)
   ├── Métricas: Accuracy, Precision, Recall, AUC
   └── Visualizaciones

6. PREDICCIÓN
   ├── Input: Texto nuevo
   ├── Preprocesamiento automático
   ├── Inferencia con modelo entrenado
   └── Output: Sentimiento + Confianza
```
""")

# ============================================================================
# ARQUITECTURA LSTM
# ============================================================================

st.markdown("---")
st.header("🏗️ Arquitectura del Modelo LSTM")

tab1, tab2, tab3 = st.tabs(["Diagrama", "Explicación", "Código"])

with tab1:
    st.markdown("""
    ```
    Input: "This movie is great!"
          ↓
    ┌─────────────────────────┐
    │  Tokenization           │  → [23, 145, 56, 789, 2]
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │  Embedding (64 dim)     │  → Vectores densos
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │  LSTM (64 units)        │  → Aprende secuencias
    │  + Recurrent Dropout    │
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │  Dropout (30%)          │  → Previene overfitting
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │  Dense (32, ReLU)       │  → Representación densa
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │  Dropout (20%)          │
    └─────────────────────────┘
          ↓
    ┌─────────────────────────┐
    │  Output (1, Sigmoid)    │  → Probabilidad [0, 1]
    └─────────────────────────┘
          ↓
    Output: 0.92 (Positive 92%)
    ```
    """)

with tab2:
    st.markdown("""
    ### ¿Qué es una LSTM?

    **LSTM (Long Short-Term Memory)** es un tipo especial de red neuronal recurrente (RNN)
    diseñada para aprender dependencias a largo plazo en secuencias.

    #### ¿Por qué LSTM para texto?

    1. **Memoria Contextual**: Recuerda información de palabras anteriores
       - "not good" vs "very good" → el "not" cambia el significado

    2. **Orden Importa**: Captura la secuencia de palabras
       - "dog bites man" ≠ "man bites dog"

    3. **Relaciones Largas**: Conecta palabras lejanas en el texto
       - "The movie was... [200 palabras]... absolutely terrible"

    #### Componentes Clave:

    - **Embedding Layer**: Convierte palabras en vectores densos
      - Similar a "coordenadas" en un espacio de 64 dimensiones
      - Palabras similares tienen vectores similares

    - **LSTM Cell**: Tiene 3 "puertas" que controlan el flujo de información
      - **Forget Gate**: Qué olvidar del contexto anterior
      - **Input Gate**: Qué nueva información agregar
      - **Output Gate**: Qué información pasar a la siguiente capa

    - **Dropout**: Apaga aleatoriamente neuronas durante entrenamiento
      - Previene que el modelo "memorice" los datos

    - **Dense Layer**: Capa totalmente conectada para clasificación final
    """)

with tab3:
    st.code("""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# Crear modelo
model = Sequential([
    # 1. Embedding: palabras → vectores
    Embedding(
        input_dim=10000,      # Vocabulario
        output_dim=64,        # Dimensión vectores
        input_length=300      # Longitud secuencia
    ),

    # 2. LSTM: procesa secuencia
    LSTM(
        units=64,
        recurrent_dropout=0.2
    ),

    # 3. Regularización
    Dropout(0.3),

    # 4. Capa intermedia
    Dense(32, activation='relu'),
    Dropout(0.2),

    # 5. Output: probabilidad
    Dense(1, activation='sigmoid')
])

# Compilar
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
    """, language="python")

# ============================================================================
# CÓMO USAR
# ============================================================================

st.markdown("---")
st.header("🚀 Cómo Usar esta Aplicación")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Página Principal")
    st.markdown("""
    - Escribe una reseña de película en el cuadro de texto
    - O selecciona un ejemplo del sidebar
    - Haz clic en "🔍 Analizar"
    - Observa el resultado y la confianza del modelo
    """)

    st.subheader("3️⃣ Esta Página (About)")
    st.markdown("""
    - Aprende sobre el proyecto
    - Entiende la tecnología usada
    - Descubre cómo funciona el modelo
    """)

with col2:
    st.subheader("2️⃣ Métricas")
    st.markdown("""
    - Revisa el rendimiento del modelo
    - Visualiza el histórico de entrenamiento
    - Compara con otros modelos
    - Analiza ejemplos de predicciones
    """)

    st.subheader("💡 Tips")
    st.markdown("""
    - Reseñas más largas = mejores predicciones
    - Escribe en inglés (dataset IMDB)
    - Sé específico sobre la película
    - Expresa claramente tu opinión
    """)

# ============================================================================
# ESTRUCTURA DEL PROYECTO
# ============================================================================

st.markdown("---")
st.header("📂 Estructura del Proyecto")

st.code("""
sentiment-analysis/
├── data/
│   ├── raw/                    # Dataset IMDB original
│   └── processed/              # Datos preprocesados
├── src/                        # Código fuente
│   ├── data_loader.py          # Carga de datos
│   ├── text_preprocessing.py   # Limpieza de texto
│   ├── model.py                # Modelos clásicos
│   ├── deep_model.py           # LSTM
│   ├── predictor.py            # Predicciones
│   └── visualizations.py       # Gráficas
├── models/                     # Modelos entrenados (.keras)
├── reports/                    # Visualizaciones y métricas
├── notebooks/                  # 6 notebooks educativos
├── web/                        # Esta aplicación Streamlit
│   ├── app.py                  # Página principal
│   └── pages/                  # Páginas adicionales
├── config.py                   # Configuración central
├── main.py                     # Pipeline completo
└── requirements.txt            # Dependencias
""", language="text")

# ============================================================================
# EJECUCIÓN LOCAL
# ============================================================================

st.markdown("---")
st.header("💻 Ejecutar Localmente")

tab1, tab2, tab3 = st.tabs(["Instalación", "Entrenamiento", "Web App"])

with tab1:
    st.markdown("### 📥 Instalación")
    st.code("""
# Clonar repositorio
git clone <repo-url>
cd repos-deep-learning/sentiment-analysis

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\\Scripts\\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar recursos NLTK
python install_resources.py
    """, language="bash")

with tab2:
    st.markdown("### 🏋️ Entrenar Modelo")
    st.code("""
# Ejecutar pipeline completo
python main.py

# Esto ejecutará:
# 1. Carga de datos IMDB
# 2. Preprocesamiento
# 3. Entrenamiento LSTM (~10-15 min)
# 4. Evaluación
# 5. Guardado de modelo y reportes
    """, language="bash")

with tab3:
    st.markdown("### 🌐 Lanzar Web App")
    st.code("""
# Desde el directorio sentiment-analysis/
streamlit run web/app.py

# La app se abrirá en:
# http://localhost:8501
    """, language="bash")

# ============================================================================
# REFERENCIAS
# ============================================================================

st.markdown("---")
st.header("📚 Referencias y Recursos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Papers & Artículos")
    st.markdown("""
    - [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
    - [IMDB Dataset Paper](https://ai.stanford.edu/~amaas/data/sentiment/)
    - [Word Embeddings](https://arxiv.org/abs/1301.3781)
    """)

with col2:
    st.subheader("Tecnologías")
    st.markdown("""
    - [TensorFlow](https://www.tensorflow.org/)
    - [Keras Documentation](https://keras.io/)
    - [Streamlit](https://streamlit.io/)
    - [NLTK](https://www.nltk.org/)
    """)

# ============================================================================
# CONTACTO Y CONTRIBUCIÓN
# ============================================================================

st.markdown("---")
st.header("🤝 Contribuir")

st.info("""
Este proyecto es parte de un repositorio educativo de Machine Learning.

**Mejoras sugeridas:**
- Soporte para múltiples idiomas (español, francés, etc.)
- Análisis de sentimientos multi-clase (muy negativo → muy positivo)
- Explicabilidad (¿qué palabras influyen más?)
- Fine-tuning con BERT o GPT
- API REST para integración
""")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Sentiment Analysis Project</strong></p>
    <p>Desarrollado con ❤️ usando TensorFlow, Keras y Streamlit</p>
    <p style="font-size: 0.9rem;">Parte del repositorio educativo de Machine Learning</p>
</div>
""", unsafe_allow_html=True)
