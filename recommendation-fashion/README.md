# 👕 Sistema de Recomendación de Moda - Fashion Recommender

Un sistema completo de recomendación de productos de moda implementado con múltiples técnicas de Machine Learning sobre el dataset Amazon Fashion Reviews.

## 📋 Descripción del Proyecto

Este proyecto implementa un **sistema híbrido de recomendación de productos de ropa** que combina diferentes enfoques de collaborative filtering y factorización de matrices para proporcionar recomendaciones personalizadas de prendas de vestir, zapatos y accesorios. El sistema incluye:

- **User-Based Collaborative Filtering**: Recomendaciones basadas en usuarios con gustos similares
- **Item-Based Collaborative Filtering**: Recomendaciones basadas en productos similares
- **Matrix Factorization (SVD)**: Descomposición por valores singulares para predicciones de ratings
- **Hybrid Recommender System**: Combinación de múltiples técnicas para mejor precisión
- **Interfaz Web Interactiva**: Aplicación Streamlit para exploración y recomendaciones en tiempo real

## 🏗️ Arquitectura del Proyecto

```
recommendation-fashion/
├── data/                          # Datos del proyecto
│   ├── raw/                       # Dataset original Amazon Fashion Reviews
│   │   └── fashion_reviews.json   # Reviews de ropa (JSON, ~2.7M reviews)
│   └── processed/                 # Datos procesados
├── src/                           # Módulos principales
│   ├── exploratory_analysis.py    # Análisis exploratorio de datos
│   ├── user_based_collaborative_filtering.py  # Recomendaciones por usuario
│   ├── item_based_collaborative_filtering.py  # Recomendaciones por producto
│   ├── matrix_factorization_svd.py            # Factorización de matrices SVD
│   ├── hybrid_recommender_system.py           # Sistema híbrido combinado
│   └── sparsity_analysis.py                   # Análisis de dispersión
├── web/                           # Aplicación web
│   └── app.py                     # Interfaz Streamlit
├── models/                        # Modelos entrenados y matrices
├── notebooks/                     # Jupyter notebooks para análisis
├── reports/                       # Reportes y visualizaciones
├── download_fashion.py            # Script para descargar el dataset
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

## 🎯 Diferencias con el Proyecto Original (MovieLens)

| Aspecto | MovieLens (amazone) | Amazon Fashion |
|--------|-----|----------|
| **Dataset** | MovieLens 100K | Amazon Fashion Reviews |
| **Dominio** | Películas (movies) | Ropa y accesorios (fashion) |
| **Tamaño** | 100K ratings | ~2.7M reviews |
| **Usuarios** | 943 usuarios | ~800K usuarios |
| **Items** | 1,682 películas | ~180K productos |
| **Formato Datos** | TSV (tab-separated) | JSON (línea por línea) |
| **Estructura** | Archivos separados (u.data, u.item, u.user) | Un archivo JSON unificado |

## 🚀 Configuración Rápida

### 1. Clonar el Repositorio

```bash
cd recommendation-fashion
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
python download_fashion.py
```

Esto descargará el dataset Amazon Fashion Reviews (~2.7 millones de reviews) en `data/raw/fashion_reviews.json`.

**Nota**: El dataset es bastante grande (~500 MB comprimido). Asegúrate de tener suficiente espacio en disco.

## 📊 Dataset: Amazon Fashion Reviews

**Fuente**: [Jianmo Ni's Amazon Review Data](http://jmcauley.ucsd.edu/data/amazon/index.html)

- **Reviews Totales**: ~2.7 millones
- **Usuarios**: ~800,000 usuarios únicos
- **Productos**: ~180,000 productos de moda
- **Ratings**: 1-5 estrellas
- **Categorías**: Clothing, Shoes, Jewelry (Ropa, Zapatos, Joyas)

### Estructura de los Datos

Cada línea es un JSON con:
```json
{
  "reviewerID": "AID123...",           // ID del usuario
  "asin": "B00ABC123",                 // ID del producto
  "overall": 4.5,                      // Rating (1-5)
  "reviewText": "Great product!",      // Texto de la review
  "summary": "Excellent quality",      // Resumen
  "unixReviewTime": 1234567890         // Timestamp
}
```

## 🛠️ Módulos del Sistema

### 1. Análisis Exploratorio (`exploratory_analysis.py`)

```bash
python src/exploratory_analysis.py
```

**Funcionalidades:**
- Carga de datos JSON
- Estadísticas descriptivas del dataset
- Distribución de ratings (1-5 estrellas)
- Análisis de usuarios y productos más activos/populares
- Visualizaciones: distribuciones, histogramas, gráficos de barras
- Cálculo de dispersión de la matriz de ratings

### 2. User-Based Collaborative Filtering

```bash
python src/user_based_collaborative_filtering.py
```

**Características:**
- Cálculo de similitud entre usuarios (cosine similarity)
- Predicción de ratings basada en usuarios similares
- Recomendaciones top-N personalizadas
- Métricas: RMSE, MAE
- Visualizaciones de similitudes y predicciones

### 3. Item-Based Collaborative Filtering

```bash
python src/item_based_collaborative_filtering.py
```

**Características:**
- Similitud entre productos de moda
- Recomendaciones basadas en productos similares
- Análisis de co-compra y preferencias

### 4. Matrix Factorization SVD

```bash
python src/matrix_factorization_svd.py
```

**Características:**
- Descomposición por valores singulares
- Reducción de dimensionalidad (latent factors)
- Predicción de ratings con factores latentes
- Mejor escalabilidad con datasets grandes

### 5. Sistema Híbrido

```bash
python src/hybrid_recommender_system.py
```

**Características:**
- Combinación ponderada de técnicas
- Mejora de cobertura y precisión
- Manejo del cold-start problem

## 🌐 Aplicación Web Interactiva

### Iniciar la Aplicación Web

```bash
streamlit run web/app.py
```

La aplicación estará disponible en `http://localhost:8501`

### Funcionalidades de la App Web

- **Exploración de Datos**: Visualizaciones interactivas del dataset
- **Recomendaciones Personalizadas**: Ingresa ID de usuario para obtener recomendaciones
- **Búsqueda de Productos**: Busca productos por ID o características
- **Análisis de Similitud**: Explora usuarios y productos similares
- **Métricas de Evaluación**: Compara rendimiento de algoritmos
- **Estadísticas**: Análisis demográfico de usuarios y patrones de compra

## 📊 Resultados Esperados

### Métricas Típicas de Rendimiento

| Algoritmo | RMSE | MAE | Precision@10 | Coverage |
|-----------|------|-----|--------------|----------|
| User-Based CF | ~0.95 | ~0.75 | ~0.25 | ~85% |
| Item-Based CF | ~0.92 | ~0.72 | ~0.28 | ~90% |
| SVD (k=50) | ~0.88 | ~0.68 | ~0.32 | ~95% |
| Hybrid System | ~0.85 | ~0.65 | ~0.35 | ~98% |

## 🔄 Mejoras Futuras

- [ ] Implementar Deep Learning (Neural Collaborative Filtering)
- [ ] Integrar content-based filtering (características del producto)
- [ ] Análisis de texto (sentimiento en reviews)
- [ ] Contexto temporal (tendencias por temporada)
- [ ] Embeddings de productos (BERT para descriptions)
- [ ] API REST para servir recomendaciones
- [ ] Graph-based recommendations

## 📚 Referencias

1. [Amazon Review Data](http://jmcauley.ucsd.edu/data/amazon/index.html)
2. [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
3. [Matrix Factorization in Recommender Systems](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))

## 🔗 Relación con Otros Proyectos

Este proyecto es una **extensión educativa** del proyecto `amazone` (MovieLens):
- **amazone**: Recomendación de películas con MovieLens 100K
- **recommendation-fashion**: Recomendación de ropa con Amazon Fashion Reviews

Ambos demostralos mismos algoritmos aplicados a diferentes dominios.

## 📝 Notas Importantes

1. **Dataset**: El dataset se carga directamente en JSON sin archivos intermedios
2. **Memoria**: Considera usar sparse matrices para datasets muy grandes
3. **Rendimiento**: SVD es más eficiente que user-based CF con datos grandes
4. **Validación**: Se usa validación cruzada simple

## 👥 Contribución

1. Fork del repositorio
2. Crear feature branch
3. Commit de cambios
4. Push a la rama
5. Abrir Pull Request

## 📄 Licencia

Proyecto educativo - Libre para uso y modificación.

---

**¡Feliz recomendación de moda! 👕👖👠**
