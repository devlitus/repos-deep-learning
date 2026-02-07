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

## 🏗️ Estructura del Proyecto

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

### 🗑️ Archivos que Puedes Ignorar

Los siguientes archivos son modelos tradicionales (no deep learning) - **no los necesitas para aprender redes neuronales**:

```
src/
├── user_based_collaborative_filtering.py   ❌ Método tradicional
├── item_based_collaborative_filtering.py   ❌ Método tradicional
├── matrix_factorization_svd.py             ❌ SVD (no es deep learning)
├── hybrid_recommender_system.py            ❌ Híbrido tradicional
└── deep_hybrid_recommender.py              ❌ Muy complejo para empezar

main.py                                     ❌ Usa métodos tradicionales
train_deep_hybrid.py                        ❌ Muy avanzado
DEEP_HYBRID_GUIDE.md                        ❌ Sistema complejo
```

**💡 Enfócate solo en:**
- `notebooks/01_neural_network_tutorial.ipynb` ⭐
- `train_ncf_only.py` (versión script del notebook)
- `GUIA_APRENDIZAJE.md` (conceptos explicados)

---

## 📊 Dataset

**Amazon Fashion Reviews**:
- 5,000 interacciones (user-product-rating)
- 200 usuarios
- 500 productos
- Ratings: 1.0 - 5.0 ⭐

Formato JSON Lines:
```json
{"reviewerID": "user_00038", "asin": "B652070932", "overall": 4.5, ...}
```

---

## 🧠 ¿Qué Aprenderás?

### 1. **Embeddings**
Representar usuarios y productos como vectores de números:
```python
user_00038 → [0.23, -0.15, 0.89, ..., 0.45]  # 64 números
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
- **RMSE** (Root Mean Squared Error):
  - < 1.0 = Bueno ✅
  - > 1.5 = Mejorable ⚠️

---

## 🛠️ Comandos Útiles

### Entrenar desde script (alternativa al notebook)
```bash
python train_ncf_only.py
```

### Verificar instalación
```bash
python -c "import torch, pandas, numpy; print('✅ Todo OK')"
```

### Ver archivos generados
```powershell
ls models/          # Modelo guardado
ls reports/figures/ # Gráficos
```

---

## 🎛️ Hiperparámetros (Experimentar)

En el notebook o `train_ncf_only.py`:

```python
EMBEDDING_DIM = 64        # Dimensión embeddings (32, 64, 128)
HIDDEN_LAYERS = [128, 64, 32]  # Capas MLP
BATCH_SIZE = 256          # Ejemplos por batch
LEARNING_RATE = 0.001     # Velocidad de aprendizaje
EPOCHS = 10               # Número de vueltas
```

**Experimenta cambiando estos valores y observa cómo afecta el RMSE!**

---

## 📈 Resultados Esperados

Con la configuración por defecto:

| Métrica | Train | Test |
|---------|-------|------|
| **RMSE** | ~0.50 | ~1.14 |

Si test RMSE > 1.5 → Aumenta dropout o reduce epochs (hay overfitting)

---

## 🔍 Preguntas Frecuentes

### ❓ ¿Necesito GPU?
No, funciona en CPU. GPU es ~10x más rápido pero no necesario para este dataset pequeño.

### ❓ ¿Qué son embeddings?
Vectores de números que representan usuarios/productos. La red aprende estos vectores durante el entrenamiento.

### ❓ ¿Por qué NCF y no otros modelos?
NCF es un buen balance entre simplicidad y efectividad - perfecto para aprender.

### ❓ ¿Puedo usar mis propios datos?
Sí! Modifica `data/raw/fashion_reviews.json` con el mismo formato JSON Lines.

---

## 📚 Recursos para Aprender Más

### Papers:
- [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031) - Paper original de NCF
- [Deep Learning for Recommender Systems](https://arxiv.org/abs/1707.07435) - Survey completo

### Tutoriales:
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Embeddings Explained](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

### Libros:
- **"Hands-On Machine Learning"** - Aurélien Géron (Chapter 13: Embeddings)
- **"Deep Learning for Recommender Systems"** - Robin Burke

---

## 🐛 Solución de Problemas

### Error: "module 'config' has no attribute..."
```bash
# Verificar que estás en el directorio correcto
cd d:\work\repos-deep-learning\recommendation-fashion
```

### Error: "torch not found"
```bash
pip install torch
```

### Error: "fashion_reviews.json not found"
```bash
# Verificar que el archivo existe
ls data/raw/fashion_reviews.json
```

### El notebook no se conecta al kernel
```bash
# Instalar ipykernel
pip install ipykernel
python -m ipykernel install --user
```

---

## ✅ Checklist de Aprendizaje

Marca cuando completes cada concepto:

- [ ] Entiendo qué es un embedding
- [ ] Entiendo la arquitectura NCF
- [ ] Puedo explicar el training loop (forward, loss, backward, update)
- [ ] Sé interpretar RMSE
- [ ] Puedo detectar overfitting en los gráficos
- [ ] Puedo hacer predicciones con el modelo
- [ ] Puedo cambiar hiperparámetros y observar resultados
- [ ] Entiendo cómo encontrar items similares con embeddings

**¡Cuando completes todo esto, habrás dominado lo básico de redes neuronales para recomendaciones! 🎉**

---

## 🤝 Contribuir

Este es un proyecto educativo. Si encuentras errores o mejoras:

1. Abre un issue en GitHub
2. Propón cambios vía pull request
3. Comparte tus experimentos y resultados

---

## 📝 Licencia

MIT License - Úsalo libremente para aprender y enseñar.

---

## 🙏 Agradecimientos

- Dataset: [Amazon Review Data (2018)](http://jmcauley.ucsd.edu/data/amazon/index.html) por Jianmo Ni
- Paper NCF: He et al. (WWW 2017)
- Framework: PyTorch Team

---

**¡Empieza tu aprendizaje abriendo el notebook tutorial ahora! 🚀**

```bash
jupyter notebook notebooks/01_neural_network_tutorial.ipynb
```
