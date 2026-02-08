# 🎯 Guía de Ajuste de Hiperparámetros - NCF

## 📊 Resultados Actuales (Baseline con mejoras)

```
Train RMSE: 0.4046
Val RMSE:   0.7548
Test RMSE:  0.7631
Gap:        0.3502 ⚠️ (alto - indica overfitting)
```

---

## 🔬 Configuraciones Propuestas para Experimentar

### ⚡ Opción 1: **Regularización Agresiva** (Recomendado)

**Objetivo:** Reducir overfitting aumentando regularización

```python
# En config.py o directamente en train_ncf_only.py (líneas 202-208)

DL_EMBEDDING_SIZE = 64           # Mantener
DL_HIDDEN_LAYERS = [128, 64, 32] # Mantener
DL_DROPOUT_RATE = 0.5            # ⬆️ 0.3 → 0.5 (más dropout)
DL_LEARNING_RATE = 0.0005        # ⬇️ 0.001 → 0.0005 (empezar más lento)
DL_BATCH_SIZE = 512              # ⬆️ 256 → 512 (más generalización)
DL_WEIGHT_DECAY = 5e-4           # ⬆️ 1e-4 → 5e-4 (más L2)
DL_EPOCHS = 50                   # Mantener
DL_EARLY_STOPPING_PATIENCE = 3   # ⬇️ 5 → 3 (detener antes)
```

**Resultado esperado:**
- Gap Train-Val: ~0.20-0.25
- Test RMSE: ~0.70-0.75

---

### 🏗️ Opción 2: **Modelo Más Simple**

**Objetivo:** Reducir capacidad del modelo para evitar memorización

```python
DL_EMBEDDING_SIZE = 32           # ⬇️ 64 → 32 (menos capacidad)
DL_HIDDEN_LAYERS = [64, 32]      # ⬇️ [128,64,32] → [64,32] (menos capas)
DL_DROPOUT_RATE = 0.4            # ⬆️ 0.3 → 0.4
DL_LEARNING_RATE = 0.001         # Mantener
DL_BATCH_SIZE = 256              # Mantener
DL_WEIGHT_DECAY = 2e-4           # ⬆️ 1e-4 → 2e-4
DL_EPOCHS = 50                   # Mantener
DL_EARLY_STOPPING_PATIENCE = 4   # ⬇️ 5 → 4
```

**Resultado esperado:**
- Gap Train-Val: ~0.15-0.20
- Test RMSE: ~0.72-0.76 (puede subir ligeramente)
- Entrenamiento más rápido

---

### ⚙️ Opción 3: **Balance Fino**

**Objetivo:** Ajuste más conservador

```python
DL_EMBEDDING_SIZE = 48           # ⬇️ 64 → 48 (reducción moderada)
DL_HIDDEN_LAYERS = [96, 48, 24]  # Escalar proporcionalmente
DL_DROPOUT_RATE = 0.4            # ⬆️ 0.3 → 0.4
DL_LEARNING_RATE = 0.0008        # ⬇️ 0.001 → 0.0008
DL_BATCH_SIZE = 384              # ⬆️ 256 → 384
DL_WEIGHT_DECAY = 2e-4           # ⬆️ 1e-4 → 2e-4
DL_EPOCHS = 50                   # Mantener
DL_EARLY_STOPPING_PATIENCE = 4   # ⬇️ 5 → 4
```

**Resultado esperado:**
- Gap Train-Val: ~0.18-0.23
- Test RMSE: ~0.71-0.75

---

## 🎯 Recomendación

**Prueba en este orden:**

1. **Opción 1** (Regularización Agresiva) - Es la más prometedora
2. Si Gap sigue > 0.25, prueba **Opción 2** (Modelo más simple)
3. Si Opción 1 funcionó bien pero quieres optimizar más, prueba **Opción 3**

---

## 📈 Técnicas Adicionales (Avanzadas)

### 1. **Dropout en Embeddings**

Añadir dropout directamente después de los embeddings:

```python
class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_products, embedding_dim=64, hidden_layers=None,
                 dropout=0.2, use_batch_norm=False, embedding_dropout=0.1):
        super().__init__()
        # ... embeddings ...

        # Nuevo: Dropout en embeddings
        self.embedding_dropout = nn.Dropout(embedding_dropout)

    def forward(self, user_indices, product_indices):
        user_emb = self.embedding_dropout(self.user_embedding(user_indices))
        product_emb = self.embedding_dropout(self.product_embedding(product_indices))
        # ...
```

### 2. **Label Smoothing**

Suavizar los ratings para reducir confianza:

```python
# En el loop de entrenamiento
def label_smoothing(ratings, epsilon=0.1):
    return ratings * (1 - epsilon) + epsilon * 2.5  # 2.5 es el rating medio

# En train loop:
ratings_smooth = label_smoothing(ratings, epsilon=0.1)
loss = criterion(predictions, ratings_smooth)
```

### 3. **Gradient Noise**

Añadir ruido gaussiano a los gradientes:

```python
# Después del backward(), antes del step()
for param in model.parameters():
    if param.grad is not None:
        param.grad += torch.randn_like(param.grad) * 0.001
```

### 4. **Mixup de Embeddings**

Mezclar pares de usuarios/productos durante entrenamiento:

```python
def mixup_embeddings(user_emb, product_emb, ratings, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = user_emb.size(0)
    index = torch.randperm(batch_size).to(user_emb.device)

    mixed_user = lam * user_emb + (1 - lam) * user_emb[index]
    mixed_product = lam * product_emb + (1 - lam) * product_emb[index]
    mixed_ratings = lam * ratings + (1 - lam) * ratings[index]

    return mixed_user, mixed_product, mixed_ratings
```

---

## 📝 Protocolo de Experimentación

Para cada configuración:

1. **Modificar hiperparámetros** en `config.py` o `train_ncf_only.py`
2. **Ejecutar:** `python train_ncf_only.py`
3. **Registrar resultados:**
   - Train/Val/Test RMSE
   - Gap Train-Val
   - Época de mejor modelo
   - Tiempo de entrenamiento
4. **Comparar** con baseline (Test RMSE: 0.763, Gap: 0.350)

---

## 🎓 Criterios de Éxito

| Métrica | 🟢 Excelente | 🟡 Bueno | 🔴 Malo |
|---------|-------------|----------|---------|
| **Test RMSE** | < 0.70 | 0.70-0.75 | > 0.75 |
| **Gap Train-Val** | < 0.15 | 0.15-0.25 | > 0.25 |
| **Val RMSE (mejor)** | < 0.68 | 0.68-0.73 | > 0.73 |

**Tu resultado actual:**
- Test RMSE: 0.763 (🟡 Bueno)
- Gap: 0.350 (🔴 Necesita mejora)

---

## 🚀 Siguiente Paso

**Ejecuta Opción 1 ahora:**

1. Edita `train_ncf_only.py` líneas 207-208:
   ```python
   dropout_rate = 0.5      # Cambiar de 0.3 a 0.5
   weight_decay = 5e-4     # Cambiar de 1e-4 a 5e-4
   ```

2. Edita línea 204 (batch_size):
   ```python
   batch_size = 512        # Cambiar de 256 a 512
   ```

3. Edita línea 205 (learning_rate):
   ```python
   learning_rate = 0.0005  # Cambiar de 0.001 a 0.0005
   ```

4. Edita línea 330 (early stopping patience):
   ```python
   early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=True)
   ```

5. Ejecuta: `python train_ncf_only.py`

Espero ver:
- Gap < 0.25
- Test RMSE < 0.75
