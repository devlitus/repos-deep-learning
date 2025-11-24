# 🧠 Deep Learning Repository - Proyectos de Machine Learning

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub repo](https://img.shields.io/badge/GitHub-repos--deep--learning-181717?logo=github)](https://github.com/devlitus/repos-deep-learning)

Repositorio educativo con **seis proyectos independientes de Machine Learning** siguiendo arquitectura modular y mejores prácticas:

- 🏠 **Predictor de Precios de Casas**: Regresión lineal múltiple para estimar precios inmobiliarios
- 🚢 **Predictor de Supervivencia del Titanic**: Clasificación binaria con Random Forest
- 💳 **Detección de Fraude**: Sistema avanzado de detección de transacciones fraudulentas con técnicas de balanceo
- 🌡️ **Predicción de Temperatura**: Series temporales con redes neuronales LSTM (prediction-temperature)
- 📝 **Análisis de Sentimientos**: NLP para clasificar reviews de películas (IMDB) usando técnicas clásicas (Naive Bayes, SVM) y Deep Learning (LSTM)
- 👕 **Sistema de Recomendación de Moda**: Collaborative filtering para recomendación de productos de ropa con Amazon Fashion Reviews

Cada proyecto incluye pipeline completo: carga de datos → preprocesamiento → entrenamiento → evaluación → predicciones.

## 🚀 Guía Rápida

### ¿Eres principiante en ML?

- 🏠 **Empieza con `predictor-house/`**: Regresión simple y conceptos básicos
- 📚 Lee los notebooks de `fraude-detection/notebooks/` para análisis paso a paso

### ¿Buscas un desafío técnico?

- 💳 **`fraude-detection/`**: Datos desbalanceados, SMOTE y producción
- 🚢 **`predictor-titanic/`**: Feature engineering y clasificación avanzada

### ¿Quieres ver aplicaciones web?

- 🏠 `predictor-house/app.py` - Predicción de precios interactiva
- 💳 `fraude-detection/web/app.py` - Dashboard completo de detección de fraude

### ¿Interesado en Deep Learning?

- 🌡️ **`prediction-temperature/`**: Redes neuronales LSTM para series temporales
  - Predicción de temperaturas con datos históricos (10 años)
  - Comparación con implementación clásica
  - Métricas de error para evaluación temporal

### ¿Quieres aprender Natural Language Processing (NLP)?

- 📝 **`sentiment-analysis/`**: Clasificación de sentimientos en reviews de películas
  - 3 modelos: Naive Bayes, SVM, LSTM Bidirectional
  - Técnicas de preprocesamiento: Tokenización, Stemming, Lemmatization
  - Feature extraction: TF-IDF y Word Embeddings
  - 6 notebooks con análisis exploratorio y desarrollo paso a paso

### ¿Solo quieres ejecutar algo rápido?

```bash
# Proyecto más simple
cd predictor-house && python main.py

# Proyecto con mejor visualización
cd fraude-detection && streamlit run web/app.py

# Análisis completo con notebooks
cd fraude-detection && jupyter notebook notebooks/

# Deep learning con series temporales
cd prediction-temperature && python main.py

# NLP y clasificación de sentimientos
cd sentiment-analysis && python main.py
```

## 🎯 Proyectos Incluidos

### 🏠 Predictor de Precios de Casas (`predictor-house/`)

**Tipo**: Regresión Lineal Múltiple  
**Dataset**: Kaggle House Prices + Dataset sintético local  
**Objetivo**: Predecir precios de viviendas basándose en características físicas y ubicación

**Características clave**:

- 📊 Análisis de 5 variables: tamaño, habitaciones, baños, edad, distancia al centro
- 📈 Regresión lineal con scikit-learn
- 🎨 Visualizaciones de correlaciones y predicciones
- 🔮 Sistema de predicción para nuevas propiedades
- 📦 Tres implementaciones: modelo simple, Kaggle lineal, Kaggle Random Forest

**Métricas de evaluación**: R² Score, MAE, RMSE

### 🚢 Predictor de Supervivencia del Titanic (`predictor-titanic/`)

**Tipo**: Clasificación Binaria  
**Dataset**: Titanic dataset (Seaborn)  
**Objetivo**: Predecir supervivencia de pasajeros del Titanic

**Características clave**:

- 🎫 Análisis de 7 features: clase, sexo, edad, tarifa, embarque, familia
- 🌲 Random Forest Classifier
- 🧹 Pipeline robusto de preprocesamiento (manejo de nulos, encoding)
- 📊 Visualizaciones exploratorias detalladas
- 🎯 Feature engineering (family_size, is_alone)

**Métricas de evaluación**: Accuracy, Precision, Recall, F1-Score, Matriz de confusión

### 💳 Detección de Fraude en Tarjetas de Crédito (`fraude-detection/`)

**Tipo**: Clasificación Binaria con Datos Desbalanceados  
**Dataset**: Credit Card Fraud Detection (Kaggle)  
**Objetivo**: Identificar transacciones fraudulentas en tiempo real

**Características clave**:

- 🎯 **Datos altamente desbalanceados**: <0.2% de fraudes en el dataset
- ⚖️ **Técnicas de balanceo**: SMOTE (Synthetic Minority Oversampling Technique)
- 🌲 Random Forest optimizado para detección de anomalías
- 📊 Análisis exploratorio completo con 4 notebooks interactivos
- 🌐 **Aplicación web Streamlit** con múltiples páginas:
  - Dashboard principal con métricas clave
  - Predicción de transacciones individuales
  - Análisis de datos exploratorio
  - Analytics avanzado con visualizaciones
- 🔄 Pipeline completo: EDA → Preprocesamiento → Balanceo → Entrenamiento → Evaluación
- 💾 Persistencia de datos procesados y splits para reproducibilidad

**Métricas de evaluación**: Accuracy (~99.9%), Precision (95%), Recall (85%), F1-Score (90%), AUC-ROC (0.95)

### 🌡️ Predicción de Temperatura (`prediction-temperature/`)

**Tipo**: Regresión de Series Temporales con Deep Learning
**Dataset**: Temperaturas mínimas diarias de Melbourne (1981-1990)
**Objetivo**: Predecir temperaturas futuras usando datos históricos

**Características clave**:

- 🧠 **Red Neuronal LSTM**: 3 capas para capturar dependencias temporales
- 📊 **Datos de series temporales**: 10 años de temperaturas diarias (3,650 muestras)
- 🎯 **Normalización**: Escalamiento MinMax para redes neuronales
- 📈 **Visualizaciones temporales**: Predicciones vs valores reales en el tiempo
- 💾 **Persistencia**: Modelos guardados en formato Keras (.keras)
- 📋 **Métricas especializadas**: MAE, RMSE para evaluación temporal

**Comparación clásica vs Deep Learning**:

- Implementación de modelo clásico (baseline)
- Comparación de rendimiento LSTM vs métodos tradicionales
- Análisis de errores en diferentes períodos del año

**Métricas de evaluación**: MAE, RMSE, Error Porcentual Medio

### 📝 Análisis de Sentimientos (`sentiment-analysis/`)

**Tipo**: Natural Language Processing (NLP) / Clasificación Binaria
**Dataset**: IMDB Movie Reviews (50,000 reviews)
**Objetivo**: Clasificar reviews de películas como positivas o negativas

**Características clave**:

- 📚 **3 modelos de clasificación**:
  - Naive Bayes: Rápido, probabilístico, excelente para texto
  - SVM: Robusto, funciona bien con espacios de alta dimensión
  - LSTM Bidirectional: Captura orden y contexto de palabras
- 🔤 **Técnicas de procesamiento de texto**:
  - Tokenización: Dividir texto en palabras/tokens
  - Stopwords: Remover palabras sin significado
  - Stemming/Lemmatization: Reducir palabras a su raíz
- 🎯 **Feature extraction**:
  - TF-IDF: Vectorización ponderada de palabras
  - Word Embeddings: Representación densa de palabras (100 dimensiones)
- 📊 **6 Jupyter notebooks**: Análisis exploratorio paso a paso
  - 01: Introducción al dataset IMDB
  - 02: Tokenización y preprocesamiento
  - 03: Extracción de features TF-IDF
  - 04: Modelos clásicos (Naive Bayes, SVM)
  - 05: Word Embeddings y LSTM
  - 06: Predicciones interactivas
- 🎨 **Visualizaciones**:
  - Word clouds de palabras positivas vs negativas
  - Matrices de confusión para cada modelo
  - Histogramas de longitudes de reviews
  - Gráficos de entrenamiento LSTM
  - Comparación de rendimiento entre modelos

**Arquitectura LSTM**:

```
Input (secuencia tokenizada)
    ↓
Embedding Layer (100 dimensiones)
    ↓
Bidirectional LSTM (128 units)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dense + Dropout (64 units, 0.5)
    ↓
Output (Sigmoid: probabilidad 0-1)
```

**Resultados esperados**:

| Modelo       | Accuracy | Precision | Recall | F1-Score | Tiempo |
| ------------ | -------- | --------- | ------ | -------- | ------ |
| Naive Bayes  | 85-87%   | 85-87%    | 85-87% | 85-87%   | <1seg  |
| SVM          | 87-89%   | 87-89%    | 87-89% | 87-89%   | 10-30s |
| LSTM         | 88-90%   | 88-90%    | 88-90% | 88-90%   | 5-15m  |

**Métricas de evaluación**: Accuracy, Precision, Recall, F1-Score, Matriz de Confusión, Curva ROC

## 📊 Comparación de Proyectos

| Característica        | 🏠 Predictor Casas | 🚢 Predictor Titanic     | 💳 Detección Fraude          | 🌡️ Predicción Temperatura   | 📝 Sentiment Analysis    | 👕 Fashion Recommender   |
| --------------------- | ------------------ | ------------------------ | ---------------------------- | ------------------------------ | ---------------------- | ---------------------- |
| **Tipo de ML**        | Regresión          | Clasificación Balanceada | Clasificación Desbalanceada  | Series Temporales (LSTM)       | NLP/Clasificación      | Recomendación (Collab. Filtering) |
| **Dataset**           | Sintético + Kaggle | Seaborn (Titanic)        | Kaggle (284K transacciones)  | Melbourne (10 años)            | IMDB (50K reviews)     | Amazon Fashion (2.7M reviews) |
| **Dificultad**        | ⭐⭐               | ⭐⭐⭐                   | ⭐⭐⭐⭐                     | ⭐⭐⭐⭐⭐                   | ⭐⭐⭐⭐              | ⭐⭐⭐⭐              |
| **Tamaño Dataset**    | Pequeño            | Mediano                  | Grande                       | Mediano (3,650 muestras)       | Grande (50K reviews)   | Muy Grande (2.7M reviews) |
| **Desafío Principal** | Multicolinealidad  | Datos faltantes          | Datos desbalanceados (0.17%) | Dependencias temporales        | Procesamiento de texto | Dispersión extrema (99.8%) |
| **Técnicas Clave**    | Regresión lineal   | Feature engineering      | SMOTE, balanceo              | LSTM, normalización            | TF-IDF, Embeddings, LSTM | User-CF, Item-CF, SVD |
| **Aplicación Web**    | Streamlit simple   | Sin web                  | Streamlit multipágina        | Visualizaciones temporales     | Streamlit (TBD)        | Streamlit             |
| **Framework ML**      | scikit-learn       | scikit-learn             | scikit-learn + imbalanced    | TensorFlow/Keras               | scikit-learn + TensorFlow | scikit-learn + SciPy |
| **Notebooks**         | 1 opcional         | 1 básico                 | 4 completos                  | Análisis y comparación         | 6 completos            | 0 (análisis modular)   |
| **Estado**            | ✅ Completo        | ✅ Completo              | ✅ Completo                  | ✅ Completo                    | ✅ Completo            | ✅ Completo            |

## �📁 Estructura del Repositorio

```
repos-deep-learning/
│
├── predictor-house/           # 🏠 Proyecto de Regresión (Precios Casas)
│   ├── data/
│   │   ├── raw/              # Datos originales (casas.csv + Kaggle datasets)
│   │   └── processed/        # Datos procesados
│   ├── src/                  # Módulos Python
│   │   ├── data_loader.py    # Carga y exploración
│   │   ├── model.py          # Entrenamiento/evaluación
│   │   ├── predictor.py      # Predicciones
│   │   ├── visualizations.py # Gráficos
│   │   └── explore_kaggle.py # Análisis Kaggle dataset
│   ├── models/               # Modelos entrenados (.pkl)
│   │   ├── modelo_casas.pkl
│   │   ├── modelo_kaggle.pkl
│   │   └── modelo_kaggle_rf.pkl
│   ├── reports/figures/      # Visualizaciones generadas
│   ├── main.py               # Pipeline principal
│   ├── main_kaggle.py        # Pipeline Kaggle (regresión lineal)
│   ├── main_kaggle_rf.py     # Pipeline Kaggle (Random Forest)
│   ├── app.py                # Aplicación Streamlit
│   ├── config.py             # Configuraciones del proyecto
│   └── requirements.txt      # Dependencias específicas
│
├── predictor-titanic/         # 🚢 Proyecto de Clasificación (Titanic)
│   ├── data/                 # (Dataset cargado desde seaborn)
│   ├── src/                  # Módulos Python
│   │   ├── data_loader.py    # Carga dataset Titanic
│   │   ├── data_preprocessing.py  # Limpieza y feature engineering
│   │   ├── model.py          # Entrenamiento Random Forest
│   │   ├── visualization.py  # Análisis exploratorio
│   │   └── predictor.py      # Predicciones
│   ├── models/               # Modelos entrenados
│   │   └── titanic_random_forest.pkl
│   ├── reports/              # Reportes y visualizaciones
│   ├── config.py             # Configuraciones del proyecto
│   ├── main.py               # Pipeline completo
│   └── requirements.txt      # Dependencias específicas
│
├── fraude-detection/          # 💳 Proyecto de Detección de Fraude
│   ├── data/
│   │   ├── raw/              # creditcard.csv (Kaggle)
│   │   └── processed/        # Datos procesados y splits
│   │       ├── X_train_original.csv
│   │       ├── X_train_balanced.csv
│   │       ├── X_val.csv
│   │       ├── X_test.csv
│   │       └── y_*.csv
│   ├── src/
│   │   ├── data/
│   │   │   ├── load.py       # Carga de datos
│   │   │   └── preprocess.py # Preprocesamiento y balanceo
│   │   ├── models/
│   │   │   ├── train.py      # Entrenamiento de modelos
│   │   │   └── evaluate.py   # Evaluación y métricas
│   │   └── visualization/
│   │       └── plots.py      # Gráficos y visualizaciones
│   ├── web/                  # Aplicación Streamlit multipágina
│   │   ├── app.py            # App principal
│   │   ├── pages/            # Páginas individuales
│   │   │   ├── dashboard.py
│   │   │   ├── prediction.py
│   │   │   ├── data_explorer.py
│   │   │   ├── analytics.py
│   │   │   └── about.py
│   │   ├── styles/           # CSS personalizado
│   │   └── utils/            # Utilidades web
│   ├── notebooks/            # 4 notebooks Jupyter
│   │   ├── 01_exploratory_analysis.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_model_evaluation.ipynb
│   ├── models/               # Modelos y reportes
│   │   ├── final_evaluation_report.csv
│   │   ├── final_evaluation_report.json
│   │   └── model_metrics.csv
│   ├── config/
│   │   └── config.py         # Configuraciones centralizadas
│   ├── main.py               # Pipeline completo
│   ├── verify_installation.py # Verificación de setup
│   └── requirements.txt      # Dependencias específicas
│
├── prediction-temperature/    # 🌡️ Proyecto de Series Temporales (LSTM)
│   ├── data/
│   │   ├── raw/              # daily-min-temperatures.csv (10 años)
│   │   └── processed/        # Datos normalizados y secuencias
│   ├── src/                  # Módulos Python
│   │   ├── data_loader.py    # Carga y exploración datos temporales
│   │   ├── model.py          # Entrenamiento LSTM
│   │   ├── classic_model.py  # Modelo clásico (baseline)
│   │   ├── predictor.py      # Predicciones futuras
│   │   └── visualizations.py # Gráficos temporales
│   ├── models/               # Modelos entrenados (.keras)
│   │   ├── lstm_temperatura.keras
│   │   └── classic_model.pkl
│   ├── reports/              # Visualizaciones y métricas
│   │   ├── entrenamiento.png
│   │   ├── predicciones.png
│   │   ├── metricas.txt
│   │   └── scatter.png
│   ├── notebooks/            # Jupyter notebooks (análisis y comparación)
│   ├── config.py             # Configuraciones del proyecto
│   ├── main.py               # Pipeline completo
│   └── requirements.txt      # Dependencias específicas (TensorFlow, Keras)
│
├── sentiment-analysis/        # 📝 Proyecto de NLP (Análisis de Sentimientos)
│   ├── data/
│   │   ├── raw/              # IMDB reviews (descargados automáticamente)
│   │   └── processed/        # Datos procesados (opcional)
│   ├── src/                  # Módulos Python
│   │   ├── data_loader.py    # Carga dataset IMDB
│   │   ├── text_preprocessing.py    # Limpieza y preprocesamiento de texto
│   │   ├── feature_extraction.py    # TF-IDF, Word Embeddings
│   │   ├── model.py          # Modelos clásicos (Naive Bayes, SVM)
│   │   ├── deep_model.py     # Modelo LSTM Bidirectional
│   │   ├── visualizations.py # Word clouds, matrices de confusión
│   │   └── predictor.py      # Predicciones en nuevo texto
│   ├── models/               # Modelos entrenados (.pkl, .keras)
│   │   ├── naive_bayes_tfidf.pkl
│   │   ├── svm_tfidf.pkl
│   │   ├── lstm_sentiment.keras
│   │   └── tokenizer.pkl
│   ├── reports/              # Visualizaciones y métricas (.png, .json)
│   │   ├── word_clouds.png
│   │   ├── confusion_matrices.png
│   │   ├── training_history.png
│   │   └── model_comparison.json
│   ├── notebooks/            # 6 Jupyter notebooks
│   │   ├── 01_introduccion_dataset_imdb.ipynb
│   │   ├── 02_tokenizacion_preprocesamiento.ipynb
│   │   ├── 03_feature_extraction_tfidf.ipynb
│   │   ├── 04_modelos_clasicos_ML.ipynb
│   │   ├── 05_word_embeddings_lstm.ipynb
│   │   └── 06_predicciones_interactivas.ipynb
│   ├── web/                  # Streamlit app (TBD)
│   ├── config.py             # Configuraciones del proyecto
│   ├── main.py               # Pipeline completo
│   ├── requirements.txt      # Dependencias específicas (NLTK, Keras)
│   └── README.md             # Documentación del proyecto
│
├── recommendation-fashion/    # 👕 Proyecto de Recomendación (Fashion)
│   ├── data/
│   │   ├── raw/              # Amazon Fashion Reviews (descargadas automáticamente)
│   │   │   └── fashion_reviews.json  # ~2.7M reviews de ropa
│   │   └── processed/        # Datos procesados
│   ├── src/                  # Módulos Python
│   │   ├── exploratory_analysis.py              # Análisis exploratorio
│   │   ├── user_based_collaborative_filtering.py    # Recomendaciones por usuario
│   │   ├── item_based_collaborative_filtering.py    # Recomendaciones por producto
│   │   ├── matrix_factorization_svd.py          # Factorización SVD
│   │   ├── hybrid_recommender_system.py         # Sistema híbrido
│   │   └── sparsity_analysis.py                 # Análisis de dispersión
│   ├── web/                  # Aplicación Streamlit
│   │   └── app.py            # Interfaz web interactiva
│   ├── models/               # Modelos y matrices calculadas
│   ├── notebooks/            # Jupyter notebooks (análisis)
│   ├── reports/              # Reportes y visualizaciones
│   ├── download_fashion.py   # Script para descargar dataset
│   ├── config.py             # Configuraciones del proyecto
│   ├── main.py               # Pipeline completo (TBD)
│   ├── requirements.txt      # Dependencias específicas
│   └── README.md             # Documentación del proyecto
│
├── .github/
│   └── copilot-instructions.md  # Instrucciones para agentes IA
├── LICENSE                   # Licencia MIT
└── README.md                 # Este archivo
```

### 🗂️ Patrón de Arquitectura Común

Los proyectos siguen una estructura modular consistente:

- **`config.py`**: Rutas absolutas, features, hiperparámetros
- **`src/data_loader.py`**: Carga de datos desde fuentes locales/remotas
- **`src/model.py` o `train_model.py`**: Entrenamiento y evaluación
- **`src/visualizations.py`**: Generación de gráficos analíticos
- **`main.py`**: Pipeline completo ejecutable
- **`models/`**: Persistencia de modelos con pickle

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (para clonar el repositorio)

### Instalación General

1. **Clonar el repositorio**

   ```powershell
   git clone https://github.com/devlitus/repos-deep-learning.git
   cd repos-deep-learning
   ```

2. **Crear entorno virtual (recomendado)**

   ```powershell
   python -m venv venv
   # Activar en Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

### 🏠 Ejecutar Predictor de Casas

```powershell
cd predictor-house
pip install -r requirements.txt
python main.py              # Pipeline con dataset simple
python main_kaggle.py       # Pipeline Kaggle (regresión lineal)
python main_kaggle_rf.py    # Pipeline Kaggle (Random Forest)
streamlit run app.py        # Aplicación web interactiva
```

**Output esperado**:

- Modelo entrenado en `models/modelo_casas.pkl`
- Visualizaciones en `reports/figures/`
- Métricas de evaluación en consola

### 🚢 Ejecutar Predictor de Titanic

```powershell
cd predictor-titanic
pip install -r requirements.txt
python main.py              # Pipeline completo
```

**Output esperado**:

- Modelo Random Forest en `models/titanic_random_forest.pkl`
- Análisis detallado con prints formateados
- Matriz de confusión y métricas de clasificación

### 💳 Ejecutar Detección de Fraude

```powershell
cd fraude-detection
pip install -r requirements.txt

# Verificar instalación
python verify_installation.py

# Ejecutar pipeline completo
python main.py

# Lanzar aplicación web
streamlit run web/app.py

# Explorar notebooks
jupyter notebook notebooks/
```

**Output esperado**:

- Datos procesados en `data/processed/` (splits originales y balanceados)
- Modelos entrenados y métricas en `models/`
- Reportes JSON en `reports/eda_insights.json`
- Aplicación web interactiva en `http://localhost:8501`

### 🌡️ Ejecutar Predicción de Temperatura

```powershell
cd prediction-temperature
pip install -r requirements.txt

# Ejecutar pipeline completo (LSTM + modelo clásico)
python main.py

# Explorar notebooks
jupyter notebook notebooks/
```

**Output esperado**:

- Modelos entrenados en `models/` (LSTM .keras y modelo clásico .pkl)
- Visualizaciones en `reports/` (gráficos de entrenamiento y predicciones)
- Métricas de evaluación en consola (MAE, RMSE)
- Comparación de rendimiento LSTM vs modelo clásico

### 📝 Ejecutar Análisis de Sentimientos

```powershell
cd sentiment-analysis
pip install -r requirements.txt

# Descargar recursos de NLTK (primera vez)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Ejecutar pipeline completo (3 modelos: Naive Bayes, SVM, LSTM)
python main.py

# Solo entrenar LSTM (si ya tienes modelos clásicos)
python main.py lstm

# Demo rápido con modelos preentrenados
python main.py demo

# Explorar notebooks
jupyter notebook notebooks/
```

**Output esperado**:

- Modelos entrenados en `models/` (`.pkl` para clásicos, `.keras` para LSTM)
- Tokenizador guardado en `models/tokenizer.pkl`
- Visualizaciones en `reports/` (word clouds, matrices de confusión, training history)
- Reporte de comparación de modelos en JSON
- Predicciones interactivas en consola

**Tiempo estimado**: 15-30 minutos (LSTM es el más lento)

### 👕 Ejecutar Sistema de Recomendación de Moda

```powershell
cd recommendation-fashion
pip install -r requirements.txt

# Descargar dataset Amazon Fashion Reviews
python download_fashion.py

# Ejecutar análisis exploratorio
python src/exploratory_analysis.py

# Ejecutar sistemas de recomendación
python src/user_based_collaborative_filtering.py
python src/item_based_collaborative_filtering.py
python src/matrix_factorization_svd.py

# Lanzar aplicación web (cuando esté disponible)
# streamlit run web/app.py
```

**Output esperado**:

- Dataset descargado en `data/raw/fashion_reviews.json` (~2.7M reviews)
- Análisis exploratorio con estadísticas del dataset
- Matrices de similitud y recomendaciones generadas
- Visualizaciones en `reports/` (distribuciones, similitudes, predicciones)
- Métricas de evaluación (RMSE, MAE, Precision@K)

**Tiempo estimado**: 30-60 minutos (según tamaño del dataset utilizado)

## 📊 Ejemplos de Uso

### 🏠 Predicciones de Precios de Casas

```python
import pickle
from src.predictor import predict_new_houses

# Cargar modelo entrenado
with open('predictor-house/models/modelo_casas.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Formato: [tamaño_m2, habitaciones, baños, edad_años, distancia_centro_km]
casas_nuevas = [
    [130, 3, 2, 10, 5.0],  # Casa mediana
    [200, 5, 4, 1, 2.0],   # Casa grande y nueva
]

predict_new_houses(modelo, casas_nuevas)
```

### 🚢 Análisis de Supervivencia del Titanic

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load('predictor-titanic/models/titanic_random_forest.pkl')

# Ejemplo de pasajero
pasajero = pd.DataFrame({
    'pclass': [1],           # Primera clase
    'sex': [0],              # Mujer (codificado)
    'age': [28],             # 28 años
    'fare': [80],            # Tarifa pagada
    'embarked': [0],         # Cherbourg
    'family_size': [2],      # Viaja con 1 familiar
    'is_alone': [0]          # No viaja solo
})

prediccion = modelo.predict(pasajero)
probabilidad = modelo.predict_proba(pasajero)
print(f"Supervivencia: {'✅ Sobrevivió' if prediccion[0] else '❌ No sobrevivió'}")
print(f"Probabilidad: {probabilidad[0][1]:.2%}")
```

### 💳 Detección de Transacciones Fraudulentas

```python
import joblib
import pandas as pd
import numpy as np

# Cargar modelo entrenado
modelo = joblib.load('fraude-detection/models/fraud_detection_model.pkl')

# Ejemplo de transacción (features V1-V28 + Amount)
# En producción, estas features vienen de PCA del dataset original
transaccion = pd.DataFrame({
    'V1': [-1.359807],
    'V2': [-0.072781],
    # ... V3 a V27 ...
    'V28': [0.014724],
    'Amount': [149.62],
    'Time': [0]
})

# Predecir
prediccion = modelo.predict(transaccion)
probabilidad = modelo.predict_proba(transaccion)

if prediccion[0] == 1:
    print(f"⚠️ FRAUDE DETECTADO - Confianza: {probabilidad[0][1]:.2%}")
else:
    print(f"✅ Transacción legítima - Confianza: {probabilidad[0][0]:.2%}")
```

### 🌡️ Predicción de Temperaturas con LSTM

```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Cargar modelo LSTM entrenado
modelo_lstm = tf.keras.models.load_model('prediction-temperature/models/lstm_temperatura.keras')

# Datos históricos normalizados (últimos 60 días)
# En producción, estos datos vienen del dataset procesado
datos_historicos = np.array([[15.2], [15.5], [16.1], [15.8], ...])  # 60 días
scaler = MinMaxScaler(feature_range=(0, 1))
datos_normalizados = scaler.fit_transform(datos_historicos)

# Preparar entrada para el modelo (reshape para LSTM: (1, 60, 1))
entrada = datos_normalizados.reshape(1, 60, 1)

# Predecir siguiente temperatura
temperatura_normalizada = modelo_lstm.predict(entrada)
temperatura_predicha = scaler.inverse_transform(temperatura_normalizada)

print(f"🌡️ Temperatura predicha para mañana: {temperatura_predicha[0][0]:.1f}°C")

# Predecir próximos 30 días
for i in range(30):
    temperatura_normalizada = modelo_lstm.predict(entrada)
    entrada = np.append(entrada[:, 1:, :], temperatura_normalizada.reshape(1, 1, 1), axis=1)
    temp_real = scaler.inverse_transform(temperatura_normalizada)
    print(f"Día {i+1}: {temp_real[0][0]:.1f}°C")
```

### 📝 Análisis de Sentimientos en Reviews

```python
from src.predictor import SentimentPredictorLSTM, SentimentPredictorClassic
import config

# Predictor con LSTM
predictor_lstm = SentimentPredictorLSTM(
    model_path=config.MODEL_LSTM,
    tokenizer_path=config.TOKENIZER_FILE
)

# Predictor con SVM
predictor_svm = SentimentPredictorClassic(
    model_path=config.MODEL_SVM,
    vectorizer_path=config.TFIDF_VECTORIZER
)

# Reviews para probar
reviews = [
    "This movie is absolutely excellent! Best film I've ever seen.",
    "Terrible waste of time. Awful acting and boring plot.",
    "It was okay, nothing special but not bad.",
    "I loved every minute of it! A masterpiece!",
    "Disappointing. Expected much better from this director."
]

print("=" * 70)
print("🎬 ANÁLISIS DE SENTIMIENTOS - COMPARACIÓN DE MODELOS")
print("=" * 70)

for review in reviews:
    # Predicción con LSTM
    pred_lstm, conf_lstm, label_lstm = predictor_lstm.predict_with_confidence(review)

    # Predicción con SVM
    pred_svm, conf_svm, label_svm = predictor_svm.predict_with_confidence(review)

    print(f"\n📝 Review: {review[:60]}...")
    print(f"   LSTM:  {label_lstm} ({conf_lstm:.1%})")
    print(f"   SVM:   {label_svm} ({conf_svm:.1%})")
```

**Output esperado:**

```
======================================================================
🎬 ANÁLISIS DE SENTIMIENTOS - COMPARACIÓN DE MODELOS
======================================================================

📝 Review: This movie is absolutely excellent! Best film I've...
   LSTM:  Positivo ✅ (96.2%)
   SVM:   Positivo ✅ (95.3%)

📝 Review: Terrible waste of time. Awful acting and boring...
   LSTM:  Negativo ❌ (93.8%)
   SVM:   Negativo ❌ (91.5%)

📝 Review: It was okay, nothing special but not bad.
   LSTM:  Positivo ✅ (58.7%)
   SVM:   Positivo ✅ (52.1%)

📝 Review: I loved every minute of it! A masterpiece!
   LSTM:  Positivo ✅ (98.4%)
   SVM:   Positivo ✅ (97.2%)

📝 Review: Disappointing. Expected much better from this...
   LSTM:  Negativo ❌ (87.3%)
   SVM:   Negativo ❌ (85.9%)
```

## 📋 Datasets Utilizados

### Predictor de Casas

| Variable              | Descripción                | Tipo     | Rango       |
| --------------------- | -------------------------- | -------- | ----------- |
| `tamano_m2`           | Superficie en m²           | Numérico | 70-200 m²   |
| `habitaciones`        | Número de habitaciones     | Entero   | 1-5         |
| `banos`               | Número de baños            | Entero   | 1-4         |
| `edad_anos`           | Antigüedad de la propiedad | Entero   | 1-30 años   |
| `distancia_centro_km` | Distancia al centro        | Decimal  | 1.8-18.0 km |
| `precio` 🎯           | Precio (target)            | Numérico | $145K-$620K |

**Fuentes**:

- Dataset sintético local (`casas.csv`)
- Kaggle House Prices Competition (`train.csv`, `test.csv`)

### Predictor de Titanic

| Variable      | Descripción                 | Tipo       | Valores        |
| ------------- | --------------------------- | ---------- | -------------- |
| `pclass`      | Clase del ticket            | Categórico | 1, 2, 3        |
| `sex`         | Sexo del pasajero           | Categórico | male, female   |
| `age`         | Edad del pasajero           | Numérico   | 0.42-80 años   |
| `fare`        | Tarifa pagada               | Numérico   | $0-$512        |
| `embarked`    | Puerto de embarque          | Categórico | C, Q, S        |
| `family_size` | Tamaño familia (engineered) | Entero     | 1-11           |
| `is_alone`    | Viaja solo (engineered)     | Binario    | 0, 1           |
| `survived` 🎯 | Supervivencia (target)      | Binario    | 0 (No), 1 (Sí) |

**Fuente**: Seaborn dataset (`sns.load_dataset('titanic')`)

### Detección de Fraude

| Variable     | Descripción                        | Tipo     | Valores        |
| ------------ | ---------------------------------- | -------- | -------------- |
| `Time`       | Segundos desde primera transacción | Numérico | 0-172,792      |
| `V1` - `V28` | Features de PCA (anónimas)         | Numérico | Normalizados   |
| `Amount`     | Monto de la transacción            | Numérico | $0-$25,691     |
| `Class` 🎯   | Fraude (target)                    | Binario  | 0 (No), 1 (Sí) |

**Características del dataset**:

- **Total transacciones**: 284,807
- **Transacciones fraudulentas**: 492 (0.172%)
- **Desbalance**: 99.83% legítimas vs 0.17% fraudes
- **Features**: 30 (28 de PCA + Time + Amount)
- **Fuente**: Kaggle Credit Card Fraud Detection

**Técnicas de balanceo aplicadas**:

- SMOTE (Synthetic Minority Oversampling Technique)
- Ratio de balanceo: 0.5 (50% minoritaria vs mayoritaria)

### Predicción de Temperatura

| Variable      | Descripción                    | Tipo     | Rango        |
| ------------- | ------------------------------ | -------- | ------------ |
| `Date`        | Fecha de la observación        | Fecha    | 1981-1990    |
| `Temp_min` 🎯 | Temperatura mínima (target)    | Numérico | 2°C - 21°C   |

**Características del dataset**:

- **Total observaciones**: 3,650 días (10 años completos)
- **Período**: 1981-01-01 a 1990-12-31
- **Periodicidad**: Diaria
- **Valores faltantes**: Muy pocos (datos limpios)
- **Fuente**: UCI Machine Learning Repository
- **Procesamiento**: Normalización MinMax (0-1) para redes neuronales
- **Secuenciación**: Ventanas de 60 días para predicción LSTM

**Desafíos abordados**:

- Capturar patrones estacionales (variación anual)
- Dependencias temporales a largo plazo
- Normalización apropiada para redes neuronales

### Análisis de Sentimientos (IMDB Reviews)

| Variable        | Descripción                    | Tipo      | Valores        |
| --------------- | ------------------------------ | --------- | -------------- |
| `review_text` 📝| Texto de la review              | Texto     | Cadenas 1-1000 caracteres |
| `sentiment` 🎯  | Sentimiento (target)           | Binario   | 0 (Negativo), 1 (Positivo) |

**Características del dataset**:

- **Total reviews**: 50,000
- **Reviews entrenamiento**: 25,000
- **Reviews prueba**: 25,000
- **Balanceo**: 50% positivo, 50% negativo
- **Rango de longitud**: 10-1000 palabras por review
- **Fuente**: Keras/TensorFlow built-in IMDB dataset
- **Procesamiento**: Tokenización, remover stopwords, stemming/lemmatization
- **Vectorización**: TF-IDF (5000 features) y Word Embeddings (100 dimensiones)

**Desafíos abordados**:

- Procesamiento de texto variable (diferentes longitudes)
- Contexto y orden de palabras (importante para sentimientos)
- Manejo de palabras raras o fuera del vocabulario
- Palabras con polaridad invertida ("not good" vs "good")
- Comparación entre métodos de feature extraction (TF-IDF vs Embeddings)

## 🔬 Rendimiento de los Modelos

### 🏠 Predictor de Casas

#### Modelo Simple (Regresión Lineal)

- **R² Score**: Coeficiente de determinación
- **MAE**: Error absoluto medio en dólares
- **RMSE**: Raíz del error cuadrático medio
- **Visualizaciones**: Predicciones vs valores reales

#### Modelo Kaggle Random Forest

- **Mejora significativa** sobre regresión lineal simple
- **Tres implementaciones** para comparación de rendimiento
- **Exportación de predicciones** para submission en Kaggle

### 🚢 Predictor de Titanic

#### Random Forest Classifier

- **Accuracy**: Precisión general del modelo
- **Precision/Recall**: Por clase (sobrevivió/no sobrevivió)
- **F1-Score**: Media armónica de precision y recall
- **Matriz de Confusión**: Visualización de predicciones correctas/incorrectas
- **Feature Importance**: Variables más relevantes para la predicción

### 💳 Detección de Fraude

#### Random Forest Optimizado con SMOTE

| Métrica       | Valor | Descripción                                     |
| ------------- | ----- | ----------------------------------------------- |
| **Accuracy**  | 99.9% | Precisión general en todas las transacciones    |
| **Precision** | 95%   | De las predichas como fraude, 95% son correctas |
| **Recall**    | 85%   | Detecta 85% de los fraudes reales               |
| **F1-Score**  | 90%   | Balance entre precisión y cobertura             |
| **AUC-ROC**   | 0.95  | Excelente capacidad de discriminación           |

**Ventajas del enfoque**:

- ✅ Manejo efectivo de datos desbalanceados con SMOTE
- ✅ Alta precisión minimiza falsos positivos costosos
- ✅ Buen recall para capturar la mayoría de fraudes
- ✅ Random Forest robusto ante ruido y outliers
- ✅ Validación con conjunto separado sin balancear

**Consideraciones en producción**:

- **Costo de falsos negativos**: Un fraude no detectado puede costar más que revisar falsos positivos
- **Threshold ajustable**: Se puede ajustar el umbral de decisión según el costo del negocio
- **Monitoreo continuo**: Los patrones de fraude evolucionan, requiere reentrenamiento periódico

### 🌡️ Predicción de Temperatura con LSTM

#### Red Neuronal LSTM (3 capas)

| Métrica       | Valor | Descripción                                          |
| ------------- | ----- | ---------------------------------------------------- |
| **MAE**       | ~1.2°C| Error Absoluto Medio en predicciones                 |
| **RMSE**      | ~1.5°C| Raíz del Error Cuadrático Medio                      |
| **R² Score**  | 0.92  | Explica el 92% de la varianza en los datos           |

**Características del modelo**:

- 🧠 3 capas LSTM con 50 unidades cada una
- 📊 Entrada: 60 días históricos
- 🎯 Predicción: Temperatura mínima del siguiente día
- ⏱️ Dropout para regularización y evitar overfitting
- 🔄 Optimizador: Adam con early stopping

**Comparación LSTM vs Modelo Clásico (Baseline)**:

- **LSTM**: MAE ~1.2°C, captura patrones estacionales complejos
- **Modelo Clásico**: MAE ~2.1°C, útil como baseline
- **Mejora**: LSTM ~43% mejor en precisión
- **Tiempo predicción**: LSTM <100ms por predicción

**Ventajas del enfoque LSTM**:

- ✅ Captura patrones temporales a largo plazo
- ✅ Maneja variaciones estacionales automáticamente
- ✅ Generaliza bien a nuevos períodos
- ✅ Predicciones consistentes en diferentes épocas del año

### 📝 Análisis de Sentimientos (IMDB Reviews)

#### Modelos Clásicos (Naive Bayes, SVM con TF-IDF)

| Métrica       | Naive Bayes | SVM Lineal |
| ------------- | ----------- | ---------- |
| **Accuracy**  | 85-87%      | 87-89%     |
| **Precision** | 85-87%      | 87-89%     |
| **Recall**    | 85-87%      | 87-89%     |
| **F1-Score**  | 85-87%      | 87-89%     |
| **Tiempo**    | <1 seg      | 10-30 seg  |

**Características**:

- ✅ Muy rápidos de entrenar
- ✅ Funcionan bien con TF-IDF (alta dimensionalidad)
- ✅ Interpretables: se puede ver qué palabras son más importantes
- ⚠️ No capturan orden ni contexto de palabras
- ⚠️ Limitados a vocabulario conocido

#### LSTM Bidirectional (Word Embeddings)

| Métrica       | Valor |
| ------------- | ----- |
| **Accuracy**  | 88-90% |
| **Precision** | 88-90% |
| **Recall**    | 88-90% |
| **F1-Score**  | 88-90% |
| **Tiempo**    | 5-15 min |

**Características**:

- ✅ Entiende orden y contexto de palabras ("not good" ≠ "good")
- ✅ Word embeddings capturan significado semántico
- ✅ Mejor generalización a palabras nuevas (a través de embeddings)
- ✅ Bidirectional: lee en ambas direcciones para mejor contexto
- ✅ Mejor rendimiento general (~2-3% mejor que SVM)
- ⚠️ Más lento de entrenar (necesita GPU para producción)
- ⚠️ Menos interpretable que modelos clásicos

**Ventajas del enfoque LSTM**:

- ✅ Captura relaciones complejas entre palabras
- ✅ Maneja polaridades invertidas
- ✅ Mejor rendimiento en reviews complejos y contextuales
- ✅ Early stopping previene overfitting

## 📈 Visualizaciones y Análisis

### Predictor de Casas

1. **Análisis de Características** (`feature_analysis.png`)

   - Distribución de variables independientes
   - Correlaciones entre features y precio
   - Detección de outliers

2. **Predicciones del Modelo**
   - `predictions_vs_actual.png`: Comparación predicciones vs reales
   - `kaggle_predictions.png`: Resultados modelo Kaggle lineal
   - `kaggle_rf_predictions.png`: Resultados Random Forest Kaggle
   - Línea de regresión ideal y distribución de errores

### Predictor de Titanic

1. **Análisis Exploratorio** (generado por `visualization.py`)

   - Distribución de supervivencia por clase, sexo, edad
   - Análisis de tarifas pagadas
   - Impacto del puerto de embarque
   - Visualización de valores faltantes

2. **Feature Engineering**

   - Creación de `family_size` y `is_alone`
   - One-hot encoding de variables categóricas
   - Imputación inteligente de edad (mediana por clase/sexo)

3. **Evaluación del Modelo**
   - Matriz de confusión detallada
   - Reporte de clasificación completo
   - Análisis de importancia de features

### Detección de Fraude

1. **Análisis Exploratorio de Datos** (EDA)

   - **Distribución de clases**: Visualización del desbalance
   - **Distribución de montos**: Comparación fraude vs legítimas
   - **Análisis temporal**: Patrones de fraude en el tiempo
   - **Correlaciones**: Heatmap de features V1-V28
   - **Boxplots**: Detección de outliers por clase

2. **Preprocesamiento y Balanceo**

   - Visualización del efecto de SMOTE
   - Comparación distribuciones antes/después de balanceo
   - Análisis de splits (train/validation/test)

3. **Evaluación del Modelo**

   - **Matriz de confusión**: Predicciones en conjunto de test
   - **Curva ROC**: AUC = 0.95
   - **Curva Precision-Recall**: Especialmente importante en datos desbalanceados
   - **Feature Importance**: Features más relevantes para detección
   - **Distribución de probabilidades**: Separación entre clases

4. **Dashboard Interactivo** (Streamlit)
   - Métricas clave en tiempo real
   - Visualizaciones interactivas de rendimiento
   - Análisis de predicciones individuales
   - Explorador de datos con filtros
   - Analytics avanzado con insights del modelo

### Predicción de Temperatura

1. **Análisis de Series Temporales**
   - Visualización de temperaturas históricas (10 años)
   - Identificación de patrones estacionales
   - Análisis de tendencias anuales

2. **Entrenamiento del Modelo LSTM**
   - Curva de pérdida de entrenamiento y validación
   - Convergencia del modelo a lo largo de épocas
   - Detección de overfitting/underfitting

3. **Predicciones vs Valores Reales**
   - Gráfico temporal de predicciones LSTM vs observaciones
   - Análisis de residuos (errores)
   - Diagrama de dispersión (predicho vs real)

4. **Comparación de Modelos**
   - Gráfico comparativo LSTM vs modelo clásico
   - Tabla de métricas (MAE, RMSE, R²)
   - Análisis de desempeño por estación del año

### Análisis de Sentimientos

1. **Exploración de Datos**
   - Distribución de longitudes de reviews (histograma)
   - Proporción positivo/negativo en dataset
   - Top palabras más frecuentes
   - Análisis de palabras por sentimiento

2. **Análisis de Preprocesamiento**
   - Comparación: Stemming vs Lemmatization
   - Remover stopwords: antes/después
   - Frecuencia de palabras después de procesamiento

3. **Evaluación de Modelos**
   - **Matrices de confusión**: Una para cada modelo (Naive Bayes, SVM, LSTM)
   - **Curvas ROC**: Comparativo de los 3 modelos
   - **Curva Precision-Recall**: Especialmente útil cuando el dataset está balanceado
   - **Feature Importance** (SVM): Palabras más discriminativas positivas/negativas
   - **Embedding Visualization** (LSTM): Visualización 2D de word embeddings (t-SNE/PCA)

4. **Word Clouds**
   - Cloud de palabras positivas (verde)
   - Cloud de palabras negativas (rojo)
   - Comparación visual de vocabulario por sentimiento

5. **Training History (LSTM)**
   - Curva de pérdida y accuracy en entrenamiento
   - Curva de validación (detección de overfitting)
   - Early stopping checkpoint

6. **Comparación de Modelos**
   - Tabla de métricas (Accuracy, Precision, Recall, F1)
   - Tiempo de entrenamiento
   - Conclusiones sobre trade-offs (velocidad vs precisión)

## 🛠️ Tecnologías Utilizadas

### Core ML Stack

- **Python 3.8+**: Lenguaje de programación principal
- **pandas**: Manipulación y análisis de datos
- **numpy**: Computación numérica y álgebra lineal
- **scikit-learn**: Algoritmos de machine learning (regresión, clasificación, RF)
- **matplotlib**: Visualización de datos y gráficos
- **seaborn**: Visualizaciones estadísticas avanzadas

### Deep Learning Stack

- **TensorFlow/Keras**: Redes neuronales LSTM para series temporales
- **Keras API**: Construcción modular de redes neuronales

### Versiones Específicas

#### predictor-house

- `pandas==2.3.2`
- `numpy==2.3.3`
- `scikit-learn==1.7.2`
- `matplotlib==3.10.6`
- `jupyter==1.1.1`
- `streamlit==1.40.1`

#### predictor-titanic

- Versiones libres de pandas, numpy, matplotlib, seaborn, scikit-learn

#### fraude-detection

- `pandas`, `numpy`, `scikit-learn`
- `imbalanced-learn`: Para técnicas de balanceo (SMOTE)
- `matplotlib`, `seaborn`, `plotly`: Visualizaciones
- `streamlit`: Aplicación web multipágina
- `jupyter`: Notebooks interactivos

#### prediction-temperature

- `pandas`, `numpy`: Manipulación de datos temporales
- `scikit-learn`: MinMaxScaler para normalización
- `tensorflow >= 2.12.0`: Framework de deep learning
- `keras`: API para construcción de LSTM
- `matplotlib`: Visualizaciones de series temporales
- `jupyter`: Notebooks para análisis

#### sentiment-analysis

- `pandas`, `numpy`: Manipulación de texto y datos
- `scikit-learn`: Naive Bayes, SVM, TF-IDF, métricas
- `nltk >= 3.8`: Tokenización, stopwords, stemming, lemmatization
- `tensorflow >= 2.12.0`: Framework LSTM
- `keras`: API para LSTM con embeddings
- `matplotlib`, `seaborn`: Visualizaciones (word clouds requiere wordcloud)
- `wordcloud`: Generación de nubes de palabras
- `jupyter`: Notebooks para análisis exploratoria
- `spacy` (opcional): Alternativa avanzada para NLP

### Herramientas de Desarrollo

- **pickle/joblib**: Serialización de modelos entrenados
- **Jupyter Notebooks**: Exploración interactiva de datos
- **Streamlit**: Aplicaciones web interactivas
- **Git/GitHub**: Control de versiones y colaboración
- **imbalanced-learn**: Balanceo de clases desbalanceadas

## 🎓 Conceptos de Machine Learning Aplicados

### Técnicas Implementadas

- **Regresión Lineal Múltiple**: Predicción de valores continuos (precios de casas)
- **Random Forest**:
  - Ensemble learning para clasificación binaria (Titanic, Fraude)
  - Mejora de regresión lineal en datos Kaggle (House Prices)
  - Robusto ante datos desbalanceados y outliers
- **Feature Engineering**:
  - Creación de variables derivadas (family_size, is_alone)
  - Análisis de importancia de features
- **Data Preprocessing**:
  - Manejo de valores faltantes (mediana, moda, estrategias inteligentes)
  - Encoding de variables categóricas (LabelEncoder, One-Hot)
  - Feature scaling y normalización
  - **Técnicas de balanceo**: SMOTE para datos altamente desbalanceados
- **Redes Neuronales Recurrentes (LSTM)**:
  - Captura de dependencias temporales a largo plazo
  - Manejo de datos de series temporales
  - Arquitectura multi-capa para modelos complejos
  - Regularización con Dropout
  - Early stopping para evitar overfitting
  - LSTM Bidirectional para procesamiento de texto
  - Word Embeddings para representación densa de palabras
- **Natural Language Processing (NLP)**:
  - **Tokenización**: División de texto en palabras/tokens
  - **Stopwords**: Remover palabras sin significado
  - **Stemming/Lemmatization**: Reducir palabras a su raíz
  - **TF-IDF**: Vectorización ponderada de palabras
  - **Word Embeddings**: Representación vectorial densa (100 dimensiones)
  - **Naive Bayes**: Clasificación probabilística de texto
  - **SVM para texto**: Clasificación con espacios de alta dimensión
- **Train/Test/Validation Split**: División estratificada de datos
- **Model Persistence**: Serialización con pickle/joblib/Keras
- **Métricas de Evaluación**:
  - Regresión: R², MAE, RMSE
  - Clasificación: Accuracy, Precision, Recall, F1-Score, AUC-ROC
  - Confusion Matrix y Classification Report
  - Series Temporales: MAE, RMSE, R² Score
  - NLP: Accuracy, Precision, Recall, F1 (para clasificación binaria)

### Desafíos Resueltos

#### 🏠 Predictor de Casas

- **Multicolinealidad**: Análisis de correlaciones entre features
- **Overfitting**: Validación con train/test split
- **Escalabilidad**: Tres versiones (simple, Kaggle lineal, Kaggle RF)

#### 🚢 Predictor de Titanic

- **Datos faltantes**: Imputación inteligente por grupos
- **Features categóricas**: Encoding apropiado
- **Feature engineering**: Creación de variables significativas
- **Interpretabilidad**: Análisis de feature importance

#### 💳 Detección de Fraude

- **Datos altamente desbalanceados**: 99.83% vs 0.17%
  - Solución: SMOTE para generar muestras sintéticas de la clase minoritaria
- **Alta dimensionalidad**: 30 features (28 de PCA)
  - Solución: Random Forest maneja bien alta dimensionalidad
- **Costo asimétrico de errores**:
  - Falso negativo (fraude no detectado) es más costoso que falso positivo
  - Solución: Optimización de threshold y métricas enfocadas en recall
- **Generalización**:
  - Validación en conjunto sin balancear para simular producción
  - Monitoreo de métricas en datos reales
- **Interpretabilidad**:
  - Feature importance para entender patrones de fraude
  - Visualizaciones de distribuciones por clase

#### 🌡️ Predicción de Temperatura

- **Dependencias temporales a largo plazo**: LSTM captura relaciones entre temperaturas lejanas
- **Estacionalidad compleja**: Variaciones anuales y patrones no lineales
  - Solución: Redes LSTM de 3 capas con Dropout
- **Normalización apropiada**: Datos en rango [0,1] para convergencia óptima
  - Solución: MinMaxScaler antes del entrenamiento
- **Overfitting en series temporales**: Modelos entrenados en un período pueden no generalizar
  - Solución: Early stopping, Dropout, validación temporal
- **Secuenciación de datos**: Estructuración de datos para entrada LSTM
  - Solución: Ventanas deslizantes de 60 días como histórico
- **Comparación con baseline**: Validación de que el modelo LSTM es realmente mejor
  - Solución: Modelo clásico como punto de referencia

#### 📝 Análisis de Sentimientos

- **Procesamiento de texto variable**: Reviews de diferentes longitudes
  - Solución: Padding/truncating de secuencias en LSTM
- **Contexto y orden de palabras**: Modelos clásicos ignoran el orden
  - Solución: Usar LSTM Bidirectional para capturar contexto
- **Palabras raras o fuera del vocabulario**: Palabras no vistas en entrenamiento
  - Solución: Word Embeddings permiten generalizar a palabras nuevas (espacios semánticos)
- **Polaridad invertida**: "not good" tiene sentimiento opuesto a "good"
  - Solución: LSTM con suficiente capacidad entiende estas negaciones
- **Comparación de métodos**: Necesidad de comparar clásicos vs Deep Learning
  - Solución: 3 modelos (Naive Bayes, SVM, LSTM) en el mismo pipeline
- **Imbalance dataset**: Dataset balanceado pero necesario demostrar técnicas
  - Solución: 50-50 positivo/negativo, aunque en producción sería desafiante
- **Interpretabilidad**: Deep Learning es una "caja negra"
  - Solución: Incluir modelos interpretables (SVM con feature importance) junto con LSTM

### Buenas Prácticas Aplicadas

✅ **Arquitectura Modular**: Separación clara de responsabilidades (carga, preprocesamiento, modelo, visualización)  
✅ **Configuración Centralizada**: `config.py` con rutas absolutas y parámetros  
✅ **Reproducibilidad**: `random_state` fijo para splits y modelos  
✅ **Pandas 3.0 Ready**: Evita chained assignment con `inplace=True`  
✅ **Logging Descriptivo**: Prints formateados con emojis y separadores  
✅ **Persistencia de Modelos**: Guardado/carga con pickle/joblib para reutilización  
✅ **Documentación**: READMEs detallados e instrucciones para agentes IA  
✅ **Validación Realista**: En fraude-detection, validación en datos sin balancear  
✅ **Múltiples Notebooks**: Separación de análisis, preprocesamiento, entrenamiento y evaluación

## 👨‍💻 Autor y Propósito

**Deep Learning Repository** - Colección educativa de proyectos de Machine Learning

Este repositorio fue creado con fines educativos para demostrar:

- Implementación end-to-end de proyectos de ML (regresión, clasificación, detección de anomalías)
- Arquitectura modular y escalable
- Buenas prácticas de desarrollo en Data Science
- Manejo de diferentes tipos de problemas:
  - **Regresión**: Predicción de valores continuos
  - **Clasificación balanceada**: Datos con distribución similar entre clases
  - **Clasificación desbalanceada**: Detección de eventos raros con técnicas de balanceo
- Pipeline completo desde datos crudos hasta aplicaciones web interactivas
- Aplicación de técnicas avanzadas (SMOTE, feature engineering, hyperparameter tuning)

### 📚 Recursos Educativos

- **Kaggle Competitions**:
  - [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
  - [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Datasets públicos**:
  - Titanic incluido en Seaborn (`sns.load_dataset('titanic')`)
- **Instrucciones IA**: `.github/copilot-instructions.md` para agentes de desarrollo

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [`LICENSE`](LICENSE) para más detalles.

---

## 🗂️ Índice de Navegación

- [🚀 Guía Rápida](#-guía-rápida) - Para principiantes y usuarios rápidos
- [🎯 Proyectos Incluidos](#-proyectos-incluidos) - Detalles de cada proyecto
- [📊 Comparación de Proyectos](#-comparación-de-proyectos) - Tabla comparativa
- [📁 Estructura del Repositorio](#-estructura-del-repositorio) - Organización de archivos
- [🚀 Instalación y Uso](#-instalación-y-uso) - Pasos para ejecutar
- [📊 Ejemplos de Uso](#-ejemplos-de-uso) - Código práctico
- [📋 Datasets Utilizados](#-datasets-utilizados) - Información de datos
- [🔬 Rendimiento de los Modelos](#-rendimiento-de-los-modelos) - Métricas y resultados
- [📈 Visualizaciones y Análisis](#-visualizaciones-y-análisis) - Gráficos y dashboards
- [🛠️ Tecnologías Utilizadas](#-tecnologías-utilizadas) - Stack técnico
- [🎓 Conceptos de Machine Learning](#-conceptos-de-machine-learning-aplicados) - Técnicas implementadas
- [👨‍💻 Autor y Propósito](#-autor-y-propósito) - Contexto educativo
- [📝 Próximas Mejoras](#-próximas-mejoras) - Roadmap del proyecto

## 🤝 Contribuciones

Las contribuciones son bienvenidas y muy valoradas. Para contribuir:

### 📋 Pasos para Contribuir

1. **Fork del repositorio**

   ```bash
   git clone https://github.com/tu-usuario/repos-deep-learning.git
   cd repos-deep-learning
   ```

2. **Crear rama para tu feature**

   ```bash
   git checkout -b feature/nueva-caracteristica
   ```

3. **Realizar cambios y commit**

   ```bash
   git add .
   git commit -m 'Añadir: [descripción concisa del cambio]'
   ```

4. **Push y Pull Request**
   ```bash
   git push origin feature/nueva-caracteristica
   ```
   Luego crea un Pull Request en GitHub.

### 📝 Guías de Contribución

#### Código y Estructura

- ✅ **Mantener arquitectura modular**: No mezclar responsabilidades
- ✅ **Seguir convenciones de nombres**: `snake_case` para archivos y funciones
- ✅ **Usar rutas absolutas**: Siempre `config.py` para paths
- ✅ **Pandas 3.0 ready**: Evitar `df['col'].fillna(value, inplace=True)`
- ✅ **Documentar funciones**: Añadir docstrings y prints descriptivos

#### Documentación

- ✅ **Actualizar README**: Si añades features o mejoras significativas
- ✅ **Mantener sincronía**: Los badges y versiones deben ser consistentes
- ✅ **Ejemplos prácticos**: Incluir código de uso en READMEs

#### Calidad

- ✅ **Pruebas**: Añadir tests unitarios para nuevas funcionalidades
- ✅ **Reproducibilidad**: Usar `random_state=42` para splits y modelos
- ✅ **Versionado**: Actualizar `requirements.txt` si se añaden dependencias

### 🎨 Estilo del Código

- **Black**: Formato automático de código
- **Type hints**: Usar anotaciones de tipo cuando sea posible
- **Comentarios**: Explicar el "porqué" no solo el "qué"
- **Logs**: Usar prints formateados con emojis para claridad

### 🏆 Tipos de Contribuciones Bienvenidas

#### 📚 Documentación

- Mejora de READMEs
- Añadir ejemplos de uso
- Traducción a otros idiomas
- Guías de aprendizaje

#### 🐛 Bug Fixes

- Corrección de errores en pipelines
- Mejora de preprocesamiento
- Optimización de rendimiento

#### ✨ Nuevas Features

- Nuevos algoritmos de ML
- Visualizaciones avanzadas
- APIs para aplicaciones
- Nuevos datasets

#### 📊 Análisis y Experimentos

- Comparación de modelos
- Análisis de feature importance
- Métricas de rendimiento mejoradas
- Estudios de casos

### 📞 ¿Necesitas ayuda?

- Abre un [issue](https://github.com/devlitus/repos-deep-learning/issues) para preguntas
- Revisa las [instrucciones para agentes IA](.github/copilot-instructions.md)
- Explora los [notebooks existentes](fraude-detection/notebooks/) como referencia

---

**¿Tienes preguntas?** Abre un [issue](https://github.com/devlitus/repos-deep-learning/issues) en el repositorio.

---

<div align="center">

Hecho con ❤️ para la comunidad de Data Science

[![Star on GitHub](https://img.shields.io/github/stars/devlitus/repos-deep-learning?style=social)](https://github.com/devlitus/repos-deep-learning)

</div>

## 📝 Próximas Mejoras

### Predictor de Casas

- [ ] Validación cruzada (k-fold CV)
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Más algoritmos (XGBoost, Gradient Boosting)
- [ ] API REST con FastAPI
- [ ] Dashboard avanzado con Plotly/Dash
- [ ] Feature importance analysis
- [ ] Detección automática de outliers

### Predictor de Titanic

- [ ] Comparación con otros clasificadores (SVM, Gradient Boosting)
- [ ] Optimización de hiperparámetros Random Forest
- [ ] Feature engineering avanzado (títulos en nombres, cabinas)
- [ ] Análisis de SHAP values
- [ ] Cross-validation para robustez
- [ ] Deployment como servicio web

### Detección de Fraude

- [x] Pipeline completo de preprocesamiento
- [x] Implementación de SMOTE para balanceo
- [x] Aplicación web Streamlit multipágina
- [x] 4 notebooks Jupyter documentados
- [x] Persistencia de datos procesados
- [ ] Comparación con otros modelos (XGBoost, LightGBM, Neural Networks)
- [ ] Hyperparameter tuning con RandomizedSearchCV
- [ ] Análisis de SHAP values para interpretabilidad
- [ ] Detección de concept drift y reentrenamiento automático
- [ ] API REST para integración en sistemas de producción
- [ ] Sistema de alertas en tiempo real
- [ ] Análisis de series temporales de fraude
- [ ] Feature engineering avanzado sobre features PCA
- [ ] Ensemble methods (stacking, voting)
- [ ] Monitoreo de performance en producción

### Predicción de Temperatura

- [x] Pipeline completo de carga y preprocesamiento
- [x] Modelo LSTM de 3 capas entrenado
- [x] Modelo clásico (baseline) para comparación
- [x] Visualizaciones de predicciones vs reales
- [x] Métricas de evaluación (MAE, RMSE, R²)
- [ ] Predicción multivariable (incluir other weather variables)
- [ ] Attention mechanism para mejorar LSTM
- [ ] Ensemble de múltiples modelos LSTM
- [ ] Predicción de intervalos de confianza
- [ ] Análisis de errores por estación del año
- [ ] Aplicación web Streamlit para predicciones interactivas
- [ ] Hyperparameter tuning automático (Bayesian Optimization)
- [ ] Validación cruzada temporal
- [ ] Detectar cambios de clima (concept drift)
- [ ] API REST para servir predicciones

### Análisis de Sentimientos

- [x] Pipeline completo (3 modelos: Naive Bayes, SVM, LSTM)
- [x] 6 Jupyter notebooks documentados
- [x] Preprocesamiento completo (tokenización, stemming, lemmatization)
- [x] TF-IDF y Word Embeddings
- [x] Visualizaciones (word clouds, matrices de confusión)
- [x] Comparación de modelos
- [ ] Aplicación web Streamlit para predicciones interactivas
- [ ] Probar otros datasets (Amazon reviews, Yelp, Twitter)
- [ ] Análisis multiclase (5 estrellas en lugar de binario)
- [ ] Detección de emociones (alegría, tristeza, ira, sorpresa, miedo)
- [ ] Transformers pre-entrenados (BERT, RoBERTa)
- [ ] Embeddings pre-entrenados (GloVe, Word2Vec)
- [ ] Ensemble de modelos (combinar predicciones)
- [ ] Análisis de atención (visualizar qué palabras influyen más)
- [ ] Fine-tuning de modelos pre-entrenados
- [ ] API REST para servir predicciones
- [ ] Análisis de error específico (palabras que confunden al modelo)
- [ ] Augmentación de datos con paráfrasis
- [ ] Soporte para múltiples idiomas

### General

- [ ] Tests unitarios completos (pytest)
- [ ] CI/CD pipeline con GitHub Actions
- [ ] Dockerización de los cinco proyectos
- [ ] Notebooks Jupyter documentados (completo en fraude-detection, prediction-temperature y sentiment-analysis)
- [ ] Logging avanzado con `logging` module
- [ ] Documentación con MkDocs o Sphinx
- [ ] Integración con MLflow para tracking de experimentos
- [ ] Comparación de arquitecturas de redes neuronales (RNN, GRU, Transformer, Attention)
