# 📋 Plan: Mejoras Recomendadas por Experimento — Laboratorio NCF

## Contexto

En la sección **Comparación de Experimentos** del Laboratorio NCF (`web/pages/ncf_lab/results.py`), se añade una nueva pestaña **"💡 Mejoras Recomendadas"** que analiza automáticamente el historial de cada entrenamiento y genera recomendaciones personalizadas para mejorar el modelo.

---

## 🎯 Objetivo

Para **cada experimento** en `st.session_state.experiments`, analizar sus métricas, curvas de entrenamiento e hiperparámetros usados, y generar un **plan de mejora específico** con acciones concretas que el usuario puede aplicar en su siguiente entrenamiento.

---

## 🧠 Motor de Diagnóstico

### Señales que se analizan por experimento

| # | Señal | Fuente | Cómo se calcula |
|---|-------|--------|-----------------|
| 1 | **Overfitting** | `history['train_rmse']`, `history['val_rmse']` | `gap = val_rmse[-1] - train_rmse[-1]` |
| 2 | **Underfitting** | `final_test_rmse` | RMSE > 0.85 indica underfitting |
| 3 | **Convergencia prematura** | `history['val_rmse']` | val_rmse se estabiliza antes del 50% de las epochs |
| 4 | **Inestabilidad** | `history['val_rmse']` | Varianza alta entre epochs consecutivas |
| 5 | **Early stopping** | `stopped_early`, `total_epochs`, `max_epochs` | Si paró mucho antes del máximo disponible |
| 6 | **Learning rate** | `learning_rate`, `history['learning_rates']` | Si LR no bajó o bajó demasiado rápido |
| 7 | **Capacidad del modelo** | `total_params`, `embedding_dim`, `hidden_layers` | Relación parámetros vs datos de entrenamiento |
| 8 | **Regularización** | `dropout`, `weight_decay`, `batch_norm` | Nivel de regularización combinada |

### Umbrales de diagnóstico

```python
THRESHOLDS = {
    'overfitting_mild':     0.10,   # gap >= 0.10
    'overfitting_severe':   0.20,   # gap >= 0.20
    'underfitting':         0.85,   # test_rmse >= 0.85
    'good_rmse':            0.70,   # test_rmse < 0.70
    'convergence_ratio':    0.50,   # Si best_epoch < 50% de total_epochs
    'instability_std':      0.02,   # std de val_rmse entre epochs
    'early_stop_ratio':     0.40,   # Si total_epochs < 40% de max_epochs
    'high_lr':              0.005,  # LR >= 0.005
    'low_lr':               0.0003, # LR <= 0.0003
    'params_per_sample':    10,     # ratio parámetros/muestras de entrenamiento
}
```

---

## 🔍 Reglas de Recomendación

### R1: Overfitting severo (gap ≥ 0.20)
**Diagnóstico:** El modelo memoriza los datos de entrenamiento.
**Recomendaciones:**
- ⬆️ Subir **Dropout** al menos +0.1 (máx 0.6)
- ⬆️ Subir **Weight Decay** al menos ×5
- ✅ Activar **Batch Normalization** si está desactivada
- ⬇️ Reducir **Embedding Dimension** (ej: 64 → 32)
- ⬇️ Reducir **capas ocultas** (quitar la capa más grande)
- ⬇️ Reducir **Paciencia** a 2-3

### R2: Overfitting moderado (0.10 ≤ gap < 0.20)
**Diagnóstico:** Hay signos de memorización pero controlables.
**Recomendaciones:**
- ⬆️ Subir **Dropout** +0.05
- ⬆️ Subir **Weight Decay** ×2
- ⬆️ Subir **Batch Size** (ej: 256 → 512)
- Verificar que **Batch Norm** esté activado

### R3: Underfitting (test_rmse ≥ 0.85)
**Diagnóstico:** El modelo no tiene suficiente capacidad para aprender los patrones.
**Recomendaciones:**
- ⬆️ Subir **Embedding Dimension** (ej: 32 → 64)
- ⬆️ Añadir capas ocultas más grandes
- ⬇️ Reducir **Dropout** (ej: 0.4 → 0.2)
- ⬇️ Reducir **Weight Decay**
- ⬆️ Subir **Epochs** máximas
- ⬆️ Subir **Paciencia**

### R4: Convergencia prematura
**Diagnóstico:** El modelo encontró su mejor punto muy pronto y luego no mejoró.
**Recomendaciones:**
- ⬇️ Reducir **Learning Rate** (ej: 0.001 → 0.0005)
- ✅ Activar **LR Scheduler** si está desactivado
- ⬆️ Subir **Paciencia** para dar más tiempo
- Probar **AdamW** si usa SGD

### R5: Inestabilidad en el entrenamiento
**Diagnóstico:** Las curvas de validación oscilan mucho entre epochs.
**Recomendaciones:**
- ⬇️ Reducir **Learning Rate**
- ⬆️ Subir **Batch Size** para gradientes más estables
- ⬇️ Reducir **Gradient Clip** (ej: 5.0 → 2.0)
- ✅ Activar **Batch Normalization**
- Probar **AdamW** en vez de SGD

### R6: Early stopping muy temprano (< 40% de epochs)
**Diagnóstico:** El modelo paró antes de explorar todo su potencial.
**Recomendaciones:**
- ⬆️ Subir **Paciencia** (ej: 3 → 5)
- ⬇️ Reducir **Learning Rate** para convergencia más gradual
- ✅ Activar **LR Scheduler** para auto-ajuste

### R7: Sin early stopping + agotó todas las epochs
**Diagnóstico:** El modelo podría seguir mejorando.
**Recomendaciones:**
- ⬆️ Subir **Epochs** máximas
- Verificar que la **Paciencia** no sea demasiado alta

### R8: Modelo excelente (test_rmse < 0.70, gap < 0.10)
**Diagnóstico:** ¡Buen resultado! Pocas mejoras necesarias.
**Recomendaciones:**
- 🎉 Guardar este modelo como baseline
- Probar variaciones mínimas para confirmar estabilidad
- Considerar ensemble con otros modelos buenos

---

## 🏗️ Arquitectura de Implementación

### Archivo: `web/pages/ncf_lab/results.py`

Se añade una **4ª pestaña** en el sistema de tabs existente:

```python
tab_curves, tab_preds, tab_compare, tab_recommendations = st.tabs([
    "📈 Curvas de Entrenamiento",
    "🎯 Predicciones",
    "🔬 Comparación de Experimentos",
    "💡 Mejoras Recomendadas"
])
```

### Funciones nuevas en `results.py`

```python
def _diagnose_experiment(exp: dict) -> list[dict]:
    """
    Analiza un experimento y retorna lista de diagnósticos.
    Cada diagnóstico: {'id': str, 'severity': str, 'title': str, 'description': str}
    severity: 'critical' | 'warning' | 'info' | 'success'
    """

def _generate_recommendations(exp: dict, diagnostics: list) -> list[dict]:
    """
    Genera recomendaciones concretas basadas en los diagnósticos.
    Cada recomendación: {
        'param': str,           # Nombre del hiperparámetro
        'current': str,         # Valor actual
        'suggested': str,       # Valor sugerido
        'direction': str,       # '⬆️' | '⬇️' | '✅' | '🔄'
        'reason': str,          # Explicación breve
        'priority': int         # 1 (alta) - 3 (baja)
    }
    """

def _render_experiment_recommendations(exp: dict):
    """
    Renderiza tarjeta de recomendaciones para un experimento individual.
    Usa st.expander con diagnósticos y tabla de recomendaciones.
    """

def render_recommendations_tab():
    """
    Renderiza la pestaña completa de mejoras recomendadas.
    Itera por todos los experimentos en session_state.
    """
```

### Flujo de datos

```
experiment (dict)
    ├── history['train_rmse'], history['val_rmse']
    ├── final_test_rmse, best_val_rmse, best_epoch
    ├── stopped_early, total_epochs, max_epochs
    ├── embedding_dim, hidden_layers, dropout
    ├── learning_rate, weight_decay, batch_norm
    ├── optimizer, batch_size, patience
    └── total_params
         │
         ▼
   _diagnose_experiment()
         │
         ▼
   diagnostics[] ──→ _generate_recommendations()
         │                      │
         ▼                      ▼
   Tarjeta diagnóstico    Tabla de mejoras
   (severity icons)       (param → sugerido)
```

---

## 🎨 Diseño de UI

### Para cada experimento se muestra:

1. **Banner de diagnóstico general** — emoji + texto resumen del estado
2. **Diagnósticos individuales** — lista con iconos de severidad (🔴🟡🟢ℹ️)
3. **Tabla de recomendaciones** — DataFrame con columnas:
   - Prioridad (🔴 Alta / 🟡 Media / 🟢 Baja)
   - Parámetro
   - Valor Actual → Sugerido
   - Razón
4. **Resumen rápido** — texto en lenguaje natural tipo: *"Sube el Dropout a 0.4 y baja el Learning Rate a 0.0005 para reducir el overfitting"*

### Si hay múltiples experimentos:
- Se muestra un **resumen comparativo** arriba con el mejor experimento y las mejoras globales
- Cada experimento se despliega con `st.expander()`

---

## 📁 Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `web/pages/ncf_lab/results.py` | Nueva pestaña + funciones de diagnóstico y recomendación |

**No se crean archivos nuevos** en la web. Todo el código queda en `results.py` ya que es lógica de visualización de resultados.

---

## ✅ Criterios de aceptación

1. [ ] La pestaña "💡 Mejoras Recomendadas" aparece en el tab de resultados
2. [ ] Cada experimento del historial muestra diagnósticos y recomendaciones
3. [ ] Las recomendaciones son específicas (valores concretos, no genéricas)
4. [ ] Los valores sugeridos se calculan a partir de los valores actuales del experimento
5. [ ] Sin experimentos se muestra mensaje informativo
6. [ ] UI consistente con el estilo existente (emojis, español, explicaciones educativas)
7. [ ] No se rompe ninguna funcionalidad existente
