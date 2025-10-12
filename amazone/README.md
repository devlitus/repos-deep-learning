# 🎬 Sistema de Recomendación de Películas - Amazoné

Un sistema completo de recomendación de películas implementado con múltiples técnicas de Machine Learning sobre el dataset MovieLens 100K.

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema híbrido de recomendación de películas** que combina diferentes enfoques de collaborative filtering y factorización de matrices para proporcionar recomendaciones personalizadas. El sistema incluye:

- **User-Based Collaborative Filtering**: Recomendaciones basadas en usuarios similares
- **Item-Based Collaborative Filtering**: Recomendaciones basadas en películas similares
- **Matrix Factorization (SVD)**: Descomposición por valores singulares para predicciones
- **Hybrid Recommender System**: Combinación de múltiples técnicas
- **Interfaz Web Interactiva**: Aplicación Streamlit para exploración y recomendaciones

## 🏗️ Arquitectura del Proyecto

```
amazone/
├── data/                          # Datos del proyecto
│   ├── raw/                       # Dataset original MovieLens 100K
│   │   └── ml-100k/              # Archivos del dataset
│   └── processed/                 # Datos procesados
├── src/                           # Módulos principales
│   ├── exploratory_analysis.py    # Análisis exploratorio de datos
│   ├── user_based_collaborative_filtering.py  # Filtrado colaborativo basado en usuarios
│   ├── item_based_collaborative_filtering.py  # Filtrado colaborativo basado en ítems
│   ├── matrix_factorization_svd.py            # Factorización de matrices SVD
│   ├── hybrid_recommender_system.py           # Sistema híbrido
│   └── sparsity_analysis.py                   # Análisis de dispersión
├── web/                           # Aplicación web
│   └── app.py                     # Interfaz Streamlit
├── models/                        # Modelos entrenados
├── notebooks/                     # Jupyter notebooks para análisis
├── reports/                       # Reportes y visualizaciones
├── download_movielens.py          # Script para descargar datos
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## 🚀 Configuración Rápida

### 1. Clonar el Repositorio

```bash
git clone <repository-url>
cd amazone
```

### 2. Crear Entorno Virtual

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar Dataset

```bash
python download_movielens.py
```

Esto descargará y descomprimirá el dataset MovieLens 100K en `data/raw/ml-100k/`.

## 📊 Dataset: MovieLens 100K

- **Usuarios**: 943 usuarios
- **Películas**: 1,682 películas
- **Ratings**: 100,000 calificaciones (1-5)
- **Géneros**: 19 categorías de películas
- **Información adicional**: Datos demográficos de usuarios, metadatos de películas

### Archivos Principales

- `u.data`: Ratings de usuarios para películas
- `u.item`: Información detallada de películas
- `u.user`: Datos demográficos de usuarios
- `u.genre`: Lista de géneros cinematográficos

## 🛠️ Módulos del Sistema

### 1. Análisis Exploratorio (`exploratory_analysis.py`)

```python
# Ejecutar análisis exploratorio
python src/exploratory_analysis.py
```

**Funcionalidades:**

- Estadísticas descriptivas del dataset
- Distribución de ratings por usuario y película
- Análisis de géneros más populares
- Visualizaciones interactivas con matplotlib y seaborn

### 2. User-Based Collaborative Filtering

```python
# Ejecutar filtrado colaborativo basado en usuarios
python src/user_based_collaborative_filtering.py
```

**Características:**

- Cálculo de similitud entre usuarios (cosine similarity)
- Predicción de ratings para películas no vistas
- Recomendaciones top-N por usuario
- Métricas de evaluación (RMSE, MAE)

### 3. Item-Based Collaborative Filtering

```python
# Ejecutar filtrado colaborativo basado en ítems
python src/item_based_collaborative_filtering.py
```

**Características:**

- Similitud entre películas
- Recomendaciones basadas en películas similares
- Análisis de correlación de géneros

### 4. Matrix Factorization SVD

```python
# Ejecutar factorización de matrices
python src/matrix_factorization_svd.py
```

**Características:**

- Descomposición por valores singulares
- Reducción de dimensionalidad
- Predicción de ratings con SVD
- Comparación con métodos tradicionales

### 5. Sistema Híbrido

```python
# Ejecutar sistema híbrido
python src/hybrid_recommender_system.py
```

**Características:**

- Combinación de múltiples técnicas
- Ponderación de recomendaciones
- Mejora de precisión

## 🌐 Aplicación Web Interactiva

### Iniciar la Aplicación Web

```bash
streamlit run web/app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Funcionalidades de la App Web

- **Exploración de Datos**: Visualizaciones interactivas del dataset
- **Recomendaciones Personalizadas**: Ingresa tu ID de usuario para obtener recomendaciones
- **Análisis de Similitud**: Explora usuarios y películas similares
- **Métricas de Evaluación**: Compara rendimiento de diferentes algoritmos
- **Búsqueda de Películas**: Busca películas por título o género

## 📈 Algoritmos Implementados

### 1. Collaborative Filtering Basado en Usuarios

```python
# Similitud de coseno entre usuarios
user_similarity = cosine_similarity(user_item_matrix)

# Predicción de rating
predicted_rating = user_mean + (similarity_weights * (neighbor_ratings - neighbor_means)).sum()
```

### 2. Collaborative Filtering Basado en Ítems

```python
# Similitud entre películas
item_similarity = cosine_similarity(item_user_matrix.T)

# Predicción basada en películas similares
predicted_rating = (item_similarity * user_ratings).sum() / item_similarity.sum()
```

### 3. Matrix Factorization con SVD

```python
# Descomposición SVD
U, sigma, Vt = svds(ratings_matrix_filled, k=50)

# Reconstrucción de matriz
predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
```

## 📊 Métricas de Evaluación

- **RMSE (Root Mean Square Error)**: Error cuadrático medio
- **MAE (Mean Absolute Error)**: Error absoluto medio
- **Precision@K**: Precisión en las top-K recomendaciones
- **Recall@K**: Cobertura en las top-K recomendaciones
- **Coverage**: Porcentaje de películas recomendadas

## 🔧 Configuración y Parámetros

### Parámetros Principales

```python
# Número de vecinos para collaborative filtering
N_NEIGHBORS = 50

# Número de factores latentes para SVD
N_FACTORS = 50

# Número de recomendaciones
TOP_N = 10

# Umbral de similitud
SIMILARITY_THRESHOLD = 0.1
```

## 📋 Requisitos del Sistema

### Python Version

- Python 3.8+

### Dependencias Principales

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
scikit-learn>=1.0.0
scipy>=1.7.0
streamlit>=1.0.0
jupyter>=1.0.0
```

## 🚀 Ejecución Completa del Pipeline

### 1. Análisis Exploratorio Completo

```bash
# Ejecutar todos los módulos en orden
python src/exploratory_analysis.py
python src/user_based_collaborative_filtering.py
python src/item_based_collaborative_filtering.py
python src/matrix_factorization_svd.py
python src/hybrid_recommender_system.py
```

### 2. Ejecución con Jupyter Notebooks

```bash
jupyter notebook
# Abrir notebooks/ para análisis interactivo
```

## 📊 Resultados Esperados

### Métricas de Rendimiento Típicas

| Algoritmo            | RMSE  | MAE   | Precision@10 | Coverage |
| -------------------- | ----- | ----- | ------------ | -------- |
| User-Based CF        | ~0.95 | ~0.75 | ~0.25        | ~85%     |
| Item-Based CF        | ~0.92 | ~0.72 | ~0.28        | ~90%     |
| Matrix Factorization | ~0.88 | ~0.68 | ~0.32        | ~95%     |
| Hybrid System        | ~0.85 | ~0.65 | ~0.35        | ~98%     |

## 🎯 Casos de Uso

### 1. Sistema de Recomendación para Plataforma Streaming

- Recomendaciones personalizadas basadas en historial
- Descubrimiento de nuevo contenido

### 2. Análisis de Preferencias de Usuarios

- Segmentación de usuarios por preferencias
- Análisis de tendencias de consumo

### 3. Sistema de Recomendación Cold-Start

- Manejo de nuevos usuarios y películas
- Estrategias híbridas para mejorar cobertura

## 🔍 Análisis de Dispersión

El dataset MovieLens 100K tiene una dispersión del **93.7%**, lo que significa que solo el 6.3% de las posibles combinaciones usuario-película tienen ratings. Este alto nivel de dispersión hace que los algoritmos de collaborative filtering y factorización de matrices sean especialmente adecuados.

## 🚨 Limitaciones y Consideraciones

1. **Cold Start Problem**: Dificultad con nuevos usuarios/películas sin historial
2. **Data Sparsity**: Alta dispersión en la matriz de ratings
3. **Scalability**: Los algoritmos pueden ser computacionalmente intensivos con datasets grandes
4. **Popularity Bias**: Tendencia a recomendar películas populares

## 🔄 Mejoras Futuras

- [ ] Implementar Deep Learning (Neural Collaborative Filtering)
- [ ] Añadir contextual information (tiempo, dispositivo)
- [ ] Implementar online learning para actualizaciones en tiempo real
- [ ] Añadir sistema de evaluación A/B testing
- [ ] Integrar con APIs de bases de datos de películas (TMDB, OMDb)

## 📚 Referencias

1. [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
2. [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
3. [Matrix Factorization Techniques for Recommender Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

## 👥 Contribución

1. Fork del repositorio
2. Crear feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit de cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](../LICENSE) para detalles.
