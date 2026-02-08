# Plan: Refactorizar `web/app.py`

## Contexto

El archivo `app.py` tiene **1806 líneas** con toda la lógica en un solo archivo: clases PyTorch, funciones de predicción, carga de datos, 4 modos de UI y visualizaciones. Esto lo hace difícil de mantener y navegar. El objetivo es separarlo en módulos pequeños y cohesivos sin perder ninguna funcionalidad.

## Estructura propuesta

```
web/
├── app.py                          # ~60 líneas  — Entry point: config, carga, routing
├── __init__.py                     # (ya existe)
├── components/
│   ├── __init__.py
│   ├── header.py                   # ~40 líneas  — Título, info técnica, session state init
│   └── footer.py                   # ~60 líneas  — Footer con detalles de persistencia
├── core/
│   ├── __init__.py
│   ├── data_loader.py              # ~130 líneas — load_and_prepare_data, compute_user/item_similarity, compute_svd, prepare_ncf_data
│   ├── predictions.py              # ~100 líneas — predict_user_based, predict_item_based, predict_svd, predict_hybrid, get_recommendations
│   └── ncf_models.py              # ~80 líneas  — NCFDataset, NeuralCollaborativeFiltering, EarlyStopping (PyTorch)
├── pages/
│   ├── __init__.py
│   ├── existing_user.py            # ~130 líneas — Modo 1: Usuario Existente (tabs: recomendaciones, historial, análisis)
│   ├── new_user.py                 # ~70 líneas  — Modo 2: Usuario Nuevo (cold start)
│   ├── statistics.py               # ~190 líneas — Modo 3: Estadísticas del Sistema (4 tabs)
│   └── ncf_lab/
│       ├── __init__.py             # ~30 líneas  — render_ncf_lab() que orquesta los sub-módulos
│       ├── hyperparameters.py      # ~170 líneas — UI de configuración de hiperparámetros
│       ├── training.py             # ~200 líneas — Bucle de entrenamiento + evaluación test
│       └── results.py             # ~370 líneas — Tabs: curvas, predicciones, comparación de experimentos
```

**Total: 13 archivos** (vs 1 archivo monolítico actual). Ningún archivo supera las 370 líneas.

## Detalle de cada archivo

### 1. `app.py` (entry point — ~60 líneas)
**Rol:** Configuración de página, imports, carga de datos, routing por modo.

Contenido:
- `sys.path` setup (líneas 6-13 actuales)
- `st.set_page_config()`
- Import y llamada a `render_header()`
- Import y llamada a `load_and_prepare_data()`, `compute_*()` (con spinner)
- Sidebar: selección de modo
- `if/elif` que llama a `render_existing_user()`, `render_new_user()`, `render_statistics()`, `render_ncf_lab()`
- Import y llamada a `render_footer()`

### 2. `components/header.py` (~40 líneas)
**Función exportada:** `render_header()`
- Título principal (línea 50-51 del app.py actual)
- Expander de info técnica (líneas 54-84)
- Divider
- Inicialización de session state (líneas 92-97)

### 3. `components/footer.py` (~60 líneas)
**Función exportada:** `render_footer()`
- Expander "Detalles Técnicos - Persistencia de Modelos" (líneas 1749-1792)
- Footer HTML con créditos (líneas 1794-1805)

### 4. `core/data_loader.py` (~130 líneas)
Funciones con `@st.cache_data`:
- `load_and_prepare_data()` (líneas 185-242)
- `compute_user_similarity()` (líneas 245-250)
- `compute_item_similarity()` (líneas 253-258)
- `compute_svd()` (líneas 261-268)
- `prepare_ncf_data()` (líneas 271-307)

### 5. `core/predictions.py` (~100 líneas)
Funciones de predicción CF:
- `predict_user_based()` (líneas 311-325)
- `predict_item_based()` (líneas 328-342)
- `predict_svd()` (líneas 345-352)
- `predict_hybrid()` (líneas 355-368)
- `get_recommendations()` (líneas 371-399)

### 6. `core/ncf_models.py` (~80 líneas)
Clases PyTorch (solo si PyTorch disponible):
- `PYTORCH_AVAILABLE` flag + imports condicionales de torch
- `NCFDataset(Dataset)` (líneas 104-115)
- `NeuralCollaborativeFiltering(nn.Module)` (líneas 117-156)
- `EarlyStopping` (líneas 158-179)

### 7. `pages/existing_user.py` (~130 líneas)
**Función exportada:** `render_existing_user(df, rating_matrix, user_sim_df, item_sim_df, U, sigma, Vt)`
- Todo el bloque del Modo 1 (líneas 430-555)
- 3 tabs: Recomendaciones, Historial, Análisis

### 8. `pages/new_user.py` (~70 líneas)
**Función exportada:** `render_new_user(df)`
- Todo el bloque del Modo 2 (líneas 560-622)
- Recomendaciones cold start basadas en popularidad

### 9. `pages/statistics.py` (~190 líneas)
**Función exportada:** `render_statistics(df, rating_matrix)`
- Todo el bloque del Modo 3 (líneas 628-816)
- 4 tabs: Distribuciones, Top Productos, Usuarios Activos, Modelos Técnicos

### 10. `pages/ncf_lab/__init__.py` (~30 líneas)
**Función exportada:** `render_ncf_lab(df)`
- Orquesta: header del lab, verificación PyTorch, llama a `prepare_ncf_data`
- Delega a `render_hyperparameters()` → `run_training()` → `render_results()`

### 11. `pages/ncf_lab/hyperparameters.py` (~170 líneas)
**Función exportada:** `render_hyperparameters(ncf_data) -> dict`
- UI de sliders/selectores (líneas 842-1003)
- Parseo de arquitectura personalizada
- Cálculo de total_params
- Expander de resumen de configuración (líneas 1019-1069)
- Retorna un dict con todos los hiperparámetros seleccionados

### 12. `pages/ncf_lab/training.py` (~200 líneas)
**Función exportada:** `run_training(ncf_data, hyperparams) -> tuple[dict, np.array, np.array]`
- Creación de NCFDataset/DataLoader para train, val y test (líneas 1081-1098)
- Creación del modelo NCF y optimizador (líneas 1100-1124)
- Bucle de entrenamiento con UI en tiempo real (líneas 1128-1245)
- Evaluación final en test set (líneas 1247-1265)
- Guardar experiment en session state (líneas 1284-1310)
- Opción de guardar modelo .pth (líneas 1312-1356)
- Retorna (experiment, all_preds, all_actuals)

### 13. `pages/ncf_lab/results.py` (~370 líneas)
**Función exportada:** `render_results(experiment, all_preds, all_actuals)`
- Tab "Curvas de Entrenamiento" con gráficos de RMSE, salud del modelo, LR (líneas 1385-1533)
- Tab "Predicciones" con scatter, histograma de errores, métricas (líneas 1535-1653)
- Tab "Comparación de Experimentos" con tabla, curvas superpuestas, ranking (líneas 1655-1742)

## Optimizaciones de código

1. **Eliminar `import os`** en statistics (línea 791 actual) — usar `pathlib` que ya está en config
2. **Usar constantes de colores de `config.py`** (`COLOR_PRIMARY=#2E86AB`, `COLOR_SECONDARY=#A23B72`, `COLOR_ACCENT=#F18F01`, `COLOR_SUCCESS=#06A77D`, `COLOR_DANGER=#D62828`) en lugar de hardcodear hex en gráficos
3. **Extraer imports comunes** — cada módulo solo importa lo que necesita (reducir imports globales)

## Orden de implementación

1. Crear directorios: `web/components/`, `web/core/`, `web/pages/`, `web/pages/ncf_lab/`
2. Crear `core/ncf_models.py` (clases PyTorch, sin dependencias de otros módulos web)
3. Crear `core/data_loader.py` (funciones de carga con cache)
4. Crear `core/predictions.py` (funciones de predicción CF)
5. Crear `components/header.py` y `components/footer.py`
6. Crear `pages/existing_user.py`
7. Crear `pages/new_user.py`
8. Crear `pages/statistics.py`
9. Crear `pages/ncf_lab/hyperparameters.py`
10. Crear `pages/ncf_lab/training.py`
11. Crear `pages/ncf_lab/results.py`
12. Crear `pages/ncf_lab/__init__.py`
13. Reescribir `app.py` como entry point mínimo
14. Crear todos los `__init__.py` necesarios

## Verificación

- Ejecutar `streamlit run web/app.py` desde `recommendation-fashion/`
- Probar los 4 modos verificando que la funcionalidad es idéntica:
  - Modo 1 (Usuario Existente): seleccionar usuario, generar recomendaciones con cada algoritmo, ver historial y análisis
  - Modo 2 (Usuario Nuevo): generar recomendaciones cold start con distintos filtros
  - Modo 3 (Estadísticas): verificar las 4 tabs con métricas y gráficos
  - Modo 4 (Laboratorio NCF): configurar hiperparámetros, entrenar modelo, ver curvas/predicciones, comparar experimentos, guardar .pth
