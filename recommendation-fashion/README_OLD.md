# 🧠 Redes Neuronales para Recomendaciones - Amazon Fashion

**Proyecto educativo** para aprender cómo funcionan las redes neuronales en sistemas de recomendación usando datos reales de Amazon Fashion.

---

## 🎯 Objetivo

Aprender **Neural Collaborative Filtering (NCF)** - una red neuronal que predice qué rating daría un usuario a un producto.

## 🚀 Inicio Rápido (3 Pasos)

### 1️⃣ Instalar Dependencias
```bash
cd recommendation-fashion
pip install -r requirements.txt
```

### 2️⃣ Abrir el Tutorial Interactivo
```bash
jupyter notebook notebooks/01_neural_network_tutorial.ipynb
```

**O en VS Code**: Abrir el archivo `.ipynb` directamente.

### 3️⃣ Ejecutar Celda por Celda

El notebook te guía paso a paso a través de:
- 📊 Entender los datos
- 🔢 Preparar datos para la red
- 🧠 Arquitectura NCF
- 🏋️ Entrenar el modelo
- 🎯 Hacer predicciones
- 🔍 Explorar embeddings

⏱️ **Tiempo**: 15-20 minutos

---

## 📚 Documentación

- **[GUIA_APRENDIZAJE.md](GUIA_APRENDIZAJE.md)** ⭐ - Guía completa de conceptos
- **[notebooks/01_neural_network_tutorial.ipynb](notebooks/01_neural_network_tutorial.ipynb)** - Tutorial interactivo

---

## 🏗️ Estructura del Proyecto (Simplificada)

```
recommendation-fashion/
├── 📓 notebooks/
│   └── 01_neural_network_tutorial.ipynb    ⭐ TUTORIAL PRINCIPAL
│
├── 📊 data/raw/
│   └── fashion_reviews.json                Dataset (5000 reviews)
│
├── 💾 models/
│   └── ncf_model.pth                       Modelo entrenado
│
├── ⚙️ config.py                            Configuración
├── 🚀 train_ncf_only.py                    Script de entrenamiento
├── 📚 GUIA_APRENDIZAJE.md                  Guía de conceptos
└── 📦 requirements.txt                     Dependencias
```

### 🗑️ Archivos Opcionales (puedes ignorar)

Los siguientes archivos son modelos tradicionales (no deep learning):
- `src/user_based_collaborative_filtering.py`
- `src/item_based_collaborative_filtering.py`
- `src/matrix_factorization_svd.py`
- `src/hybrid_recommender_system.py`
- `main.py`

**💡 Enfócate en el notebook tutorial para aprender redes neuronales.**

---

## 📊 Dataset

**Amazon Fashion Reviews**:
- 5,000 interacciones (user-product-rating)
- 200 usuarios
- 500 productos
- Ratings: 1.0 - 5.0 ⭐

---

## 🧠 ¿Qué Aprenderás?

### 1. **Embeddings**
Representar usuarios y productos como vectores de números:
```
user_00038 → [0.23, -0.15, 0.89, ..., 0.45]  (64 dims)
```

### 2. **Arquitectura NCF**
```
Usuario → Embedding (64) ──┐
                           ├─→ Concat (128) → MLP [128→64→32] → Rating
Producto → Embedding (64) ─┘
```

### 3. **Training Loop**
```python
prediction = model(user, product)
loss = MSE(prediction, real_rating)
loss.backward()  # Calcular gradientes
optimizer.step()  # Actualizar pesos
```

### 4. **Métricas**
- **RMSE** (Root Mean Squared Error): Qué tan lejos están las predicciones
  - < 1.0 = Bueno
  - > 1.5 = Mejorable

---

## 🛠️ Comandos Útiles

### Entrenar desde script
```bash
python train_ncf_only.py
```

### Verificar instalación
```bash
python -c "import torch, pandas, numpy; print('✅ Todo OK')"
```

---

## 🎛️ Hiperparámetros
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

**Opción A - Automático (Recomendado):**
```bash
pip install datasets
python download_fashion.py
```
Intenta descargar desde Hugging Face. Si falla, genera datos de prueba.

**Opción B - Manual desde Hugging Face:**
```bash
pip install datasets
python process_huggingface_dataset.py
```
Descarga directamente el dataset completo (~2.7M reviews, ~5 GB).

**Opción C - Datos de Prueba:**
```bash
python generate_test_dataset.py
```
Genera 10,000 reviews simulados para testing rápido.

**Nota**: El dataset completo ocupa ~5 GB. Asegúrate de tener suficiente espacio.

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

### 6. 🧠 Deep Hybrid Recommender (Sistema Avanzado con Deep Learning)

```bash
# Verificar instalación
python verify_deep_hybrid.py

# Entrenar Deep Hybrid System
python train_deep_hybrid.py

# O ejecutar pipeline completo (incluye Deep Hybrid)
python main.py
```

**🎯 Características Principales:**

- **Neural Collaborative Filtering (NCF)**: Red neuronal profunda con embeddings de 64 dimensiones
- **Attention Mechanism**: Aprende pesos dinámicos automáticamente (sin configuración manual)
- **4 Componentes Integrados**: User-CF + Item-CF + SVD + NCF trabajando juntos
- **PyTorch Backend**: Entrenamiento optimizado con GPU (CUDA)
- **Early Stopping**: Evita overfitting automáticamente
- **Visualizaciones Avanzadas**: Curvas de entrenamiento, pesos de atención, predicciones

**🚀 Ventajas vs Híbrido Tradicional:**

| Aspecto | Híbrido Tradicional | Deep Hybrid |
|---------|---------------------|-------------|
| RMSE | ~0.85 | **~0.75-0.79** (12% mejor) |
| Pesos | Manuales (0.3, 0.3, 0.4) | **Aprendidos automáticamente** |
| Patrones | Solo lineales | **Lineales + No lineales** |
| Embeddings | SVD (50 dims) | **NCF (64 dims) + SVD** |
| Cold Start | Regular | **Mejor** |

**📖 Documentación Completa:**

Ver [`DEEP_HYBRID_GUIDE.md`](DEEP_HYBRID_GUIDE.md) para:
- Arquitectura detallada del modelo
- Guía de instalación de PyTorch
- Configuración de hiperparámetros
- Troubleshooting y optimización
- Comparación de resultados

**⚙️ Requisitos Adicionales:**

```bash
# PyTorch (CPU)
pip install torch torchvision

# PyTorch (GPU - CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Surprise (para modelos base)
pip install scikit-surprise
```

**💡 Cuándo Usar:**

- ✅ Dataset grande (>100K ratings)
- ✅ Necesitas máxima precisión (estado del arte)
- ✅ Tienes GPU disponible
- ✅ Quieres experimentar con deep learning

**⏱️ Tiempo de Entrenamiento:**

- Con GPU (CUDA): ~10-15 minutos
- Solo CPU: ~2-3 horas (reducir épocas recomendado)

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
| **🧠 Deep Hybrid** | **~0.75** | **~0.56** | **~0.38** | **~99%** |

**Nota**: Deep Hybrid ofrece la mejor precisión pero requiere PyTorch y mayor tiempo de entrenamiento.

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
