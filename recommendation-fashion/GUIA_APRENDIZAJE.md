# 🎓 Guía de Aprendizaje - Redes Neuronales para Recomendaciones

## 🎯 Objetivo del Proyecto

Aprender **cómo funciona una red neuronal para sistemas de recomendación** usando datos reales de Amazon Fashion.

---

## 📂 Estructura del Proyecto (SIMPLIFICADA)

```
recommendation-fashion/
├── 📓 notebooks/
│   └── 01_neural_network_tutorial.ipynb  ⭐ EMPIEZA AQUÍ
│
├── 📊 data/
│   └── raw/
│       └── fashion_reviews.json          Dataset (5000 reviews)
│
├── 💾 models/
│   └── ncf_model.pth                     Modelo entrenado
│
├── 📈 reports/
│   └── figures/                          Gráficos generados
│
├── ⚙️ config.py                          Configuración (rutas, parámetros)
├── 🚀 train_ncf_only.py                  Script de entrenamiento
└── 📦 requirements.txt                   Dependencias
```

### 🗑️ Archivos que PUEDES IGNORAR (son de modelos tradicionales):
- `src/user_based_collaborative_filtering.py`
- `src/item_based_collaborative_filtering.py`
- `src/matrix_factorization_svd.py`
- `src/hybrid_recommender_system.py`
- `main.py`
- `train_deep_hybrid.py`
- `DEEP_HYBRID_GUIDE.md`

**💡 Estos archivos son métodos antiguos (no deep learning). Enfócate solo en el notebook.**

---

## 🚀 Cómo Empezar (3 Pasos)

### 1️⃣ Abrir el Notebook

```bash
cd d:\work\repos-deep-learning\recommendation-fashion
jupyter notebook notebooks/01_neural_network_tutorial.ipynb
```

O en VS Code: Abrir el archivo `.ipynb` directamente.

### 2️⃣ Ejecutar Celda por Celda

El notebook está diseñado para aprender **paso a paso**:

1. **Paso 1-2**: Entender los datos
2. **Paso 3**: Preparación (IDs → índices)
3. **Paso 4**: Ver la arquitectura de la red
4. **Paso 5**: Entrenar el modelo
5. **Paso 6**: Hacer predicciones
6. **Paso 7**: Explorar embeddings
7. **Paso 8**: Guardar modelo

⏱️ **Tiempo estimado**: 15-20 minutos

### 3️⃣ Experimentar

Cambia hiperparámetros y observa qué pasa:
- `EMBEDDING_DIM`: 32, 64, 128
- `HIDDEN_LAYERS`: [256, 128], [64, 32], etc.
- `EPOCHS`: 5, 10, 20

---

## 🧠 Conceptos que Aprenderás

### 1. **Embeddings** 🎯
- **¿Qué son?** Vectores que representan usuarios/productos
- **¿Por qué?** La red no entiende IDs como `"user_00038"`, solo números
- **Ejemplo**: `user_00038` → `[0.23, -0.15, 0.89, ..., 0.45]` (64 números)

```python
# Cada usuario = 64 números
user_embedding = [0.23, -0.15, 0.89, 0.12, ...]
```

### 2. **Neural Collaborative Filtering (NCF)** 🔗

**Arquitectura**:
```
Usuario (ID) → Embedding (64) ──┐
                                ├─→ Concatenar (128) → MLP [128→64→32] → Rating
Producto (ID) → Embedding (64) ─┘
```

**Flujo**:
1. Usuario `"user_00038"` + Producto `"B652070932"`
2. Buscar embeddings en las tablas
3. Concatenar: `[user_emb | product_emb]` = 128 números
4. Pasar por capas densas (MLP)
5. Output: Rating predicho (ej: 4.2 ⭐)

### 3. **Training Loop** 🏋️

```python
# 1. Forward Pass
prediction = model(user, product)

# 2. Calcular Error
loss = MSE(prediction, real_rating)

# 3. Backward Pass (calcular gradientes)
loss.backward()

# 4. Update (ajustar pesos)
optimizer.step()
```

**Repetir 1000s de veces** → El modelo aprende patrones

### 4. **Métricas** 📊

- **RMSE** (Root Mean Squared Error):
  - RMSE = 0 → Perfecto (imposible)
  - RMSE < 0.5 → Excelente
  - RMSE < 1.0 → Bueno
  - RMSE > 1.5 → Malo

- **MAE** (Mean Absolute Error):
  - Promedio del error absoluto
  - Más fácil de interpretar que RMSE

### 5. **Overfitting** ⚠️

**Problema**: El modelo memoriza el train set pero falla en test.

**Señales**:
- Train RMSE baja (0.5) pero Test RMSE alta (1.5)
- Gap grande entre train y test

**Soluciones**:
- Más dropout (0.2 → 0.4)
- Menos epochs
- Regularización (weight decay)
- Más datos

---

## 📊 Dataset Explicado

```json
{
  "reviewerID": "user_00038",      ← ID del usuario
  "asin": "B652070932",            ← ID del producto
  "overall": 4.5,                  ← Rating (1-5 estrellas)
  "reviewText": "Great product!",  ← Texto (no lo usamos)
  "unixReviewTime": 1590222032     ← Timestamp
}
```

**Estadísticas**:
- 5,000 interacciones
- 200 usuarios
- 500 productos
- Sparsity: ~95% (muchas combinaciones sin rating)

---

## 🎛️ Hiperparámetros Explicados

```python
EMBEDDING_DIM = 64        # Dimensión de embeddings
                          # ↑ Más capacidad pero más overfitting
                          # ↓ Menos flexible pero generaliza mejor

HIDDEN_LAYERS = [128, 64, 32]  # Capas del MLP
                                # Más capas = más capacidad
                                # Menos capas = más simple

BATCH_SIZE = 256          # Ejemplos por actualización
                          # ↑ Más rápido pero menos preciso
                          # ↓ Más lento pero más preciso

LEARNING_RATE = 0.001     # Velocidad de aprendizaje
                          # ↑ Aprende rápido pero inestable
                          # ↓ Aprende lento pero estable

EPOCHS = 10               # Vueltas completas al dataset
                          # ↑ Más entrenamiento (cuidado overfitting)
```

---

## 🔍 Preguntas Frecuentes

### ❓ ¿Por qué usar redes neuronales vs métodos tradicionales?

**Métodos tradicionales** (User/Item CF, SVD):
- ✅ Simples, rápidos
- ❌ No capturan patrones complejos
- ❌ No usan features (edad, categoría, etc.)

**Redes Neuronales** (NCF):
- ✅ Capturan patrones no-lineales
- ✅ Pueden integrar features adicionales
- ✅ Más flexibles
- ❌ Más complejas, necesitan más datos

### ❓ ¿Qué hace `torch.sigmoid(output) * 4 + 1`?

Escala la salida al rango [1, 5]:
- `sigmoid(x)` → rango [0, 1]
- `sigmoid(x) * 4` → rango [0, 4]
- `sigmoid(x) * 4 + 1` → rango [1, 5] ✅

### ❓ ¿Por qué concatenar embeddings?

Queremos que la red aprenda **interacciones** entre user y product:
```
user_emb = [0.1, 0.5, ...]  (64 dims)
prod_emb = [0.3, -0.2, ...] (64 dims)
concat   = [0.1, 0.5, ..., 0.3, -0.2, ...] (128 dims)
```

La red aprende: "cuando user[0]=0.1 Y product[5]=-0.2 → rating alto"

### ❓ ¿Cómo mejorar el modelo?

1. **Más epochs** (si no hay overfitting)
2. **Early stopping** (detener cuando test RMSE sube)
3. **Embeddings más grandes** (128 dims)
4. **Más datos** (descargar dataset completo)
5. **Features adicionales** (edad, categoría, precio)

---

## 🛠️ Comandos Útiles

### Entrenar desde script (sin notebook)
```bash
cd d:\work\repos-deep-learning\recommendation-fashion
python train_ncf_only.py
```

### Verificar dependencias
```bash
python -c "import torch, pandas, numpy; print('✅ Todo OK')"
```

### Ver archivos generados
```bash
ls models/          # Modelo guardado
ls reports/figures/ # Gráficos
```

---

## 📚 Recursos para Aprender Más

### Papers:
- [Neural Collaborative Filtering (NCF)](https://arxiv.org/abs/1708.05031) - Paper original
- [Deep Learning for Recommender Systems](https://arxiv.org/abs/1707.07435) - Survey completo

### Tutoriales:
- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Embeddings Explained](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)
- [Collaborative Filtering with PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

### Libros:
- **"Deep Learning for Recommender Systems"** - Falk, O'Reilly
- **"Hands-On Machine Learning"** - Géron (Chapter on Embeddings)

---

## 🎯 Siguiente Nivel

Una vez domines el notebook, puedes:

1. **Implementar Attention**:
   ```python
   # Dar más peso a ciertas dimensiones del embedding
   attention_weights = softmax(query @ key.T)
   ```

2. **Usar Features**:
   ```python
   # Agregar edad del producto, categoría, etc.
   input = [user_emb, product_emb, age, category]
   ```

3. **Modelos más avanzados**:
   - **DeepFM**: Factorization Machine + Deep Learning
   - **Wide & Deep**: Linear + Neural
   - **Neural Autoregressive Collaborative Filtering**

---

## 💡 Tips de Aprendizaje

1. **🕐 Ve despacio**: Lee cada celda, entiende antes de ejecutar
2. **🧪 Experimenta**: Cambia valores y observa qué pasa
3. **📊 Visualiza**: Los gráficos te ayudan a entender
4. **🐛 Debugging**: Si algo falla, lee el error con calma
5. **📝 Documenta**: Agrega notas en el notebook con tus observaciones

---

## ✅ Checklist de Aprendizaje

- [ ] Entiendo qué es un embedding
- [ ] Entiendo la arquitectura NCF
- [ ] Puedo explicar el training loop
- [ ] Sé interpretar RMSE y MAE
- [ ] Puedo detectar overfitting
- [ ] Puedo hacer predicciones con el modelo
- [ ] Puedo cambiar hiperparámetros y re-entrenar
- [ ] Entiendo cómo encontrar items similares

**Cuando completes todo esto, habrás dominado lo básico de redes neuronales para recomendaciones! 🎉**

---

## 🆘 ¿Problemas?

Si algo no funciona:

1. **Verificar dependencias**:
   ```bash
   pip list | findstr "torch pandas numpy"
   ```

2. **Verificar dataset**:
   ```bash
   python -c "import json; print(len(open('data/raw/fashion_reviews.json').readlines()))"
   ```

3. **Limpiar y re-ejecutar**:
   - En Jupyter: `Kernel → Restart & Clear Output`
   - Ejecutar todo de nuevo

4. **Revisar error**:
   - Leer el traceback completo
   - Buscar la línea que falla
   - Google el mensaje de error

---

¡Mucha suerte con tu aprendizaje! 🚀
