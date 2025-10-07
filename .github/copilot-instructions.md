# Instrucciones para Agentes de IA - repos-deep-learning

## 🎯 Arquitectura del Proyecto

Este repositorio contiene **dos proyectos independientes de ML** con estructura modular idéntica:

- **`predictor-house/`**: Regresión lineal para precios de casas (Kaggle dataset)
- **`predictor-titanic/`**: Clasificación binaria de supervivencia del Titanic

Ambos siguen el patrón: `data/` → `src/` → `models/` → `reports/`

## 📐 Estructura de Módulos

### Patrón Obligatorio por Proyecto

```
config.py           # Rutas absolutas con os.path.join(), features, hiperparámetros
src/
  data_loader.py    # load_data() → explore_data() → prepare_data()
  model.py          # split_data() → train_model() → evaluate_model() → save_model()
  predictor.py      # Predicciones con modelo guardado
  visualizations.py # plot_*() usando matplotlib
main.py             # Pipeline completo siguiendo el orden exacto
```

### Config Pattern - SIEMPRE usar

```python
# Todas las rutas son ABSOLUTAS con BASE_DIR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
# ... NUNCA rutas relativas hardcodeadas
```

### Patrón de Rutas en Mensajes/UI

**Principio:** Rutas **ABSOLUTAS** internamente, **RELATIVAS** en mensajes al usuario

```python
# ✅ CORRECTO: Funcionalidad usa rutas absolutas
model_path = config.MODEL_FILE  # Ruta absoluta completa
joblib.dump(model, model_path)

# ✅ CORRECTO: UI muestra solo nombre de archivo
display_name = os.path.basename(model_path)
print(f"💾 Modelo guardado: {display_name}")
# Output: "💾 Modelo guardado: titanic_random_forest.pkl"

# ❌ INCORRECTO: Mostrar ruta completa en UI
print(f"💾 Modelo guardado: {model_path}")
# Output: "💾 Modelo guardado: C:\dev\repos\predictor-titanic\models\titanic_random_forest.pkl"
```

**Razón:** Mejor UX sin comprometer robustez del código.

## 🔄 Flujo de Trabajo Estándar

### Orden de ejecución en `main.py`:

1. `load_data()` → 2. `explore_data()` → 3. `prepare_data()` → 4. `plot_feature_vs_target()` →
2. `split_data()` → 6. `train_model()` → 7. `evaluate_model()` → 8. `plot_predictions_vs_actual()` →
3. `save_model()` → 10. `predict_new_houses()`

**Ejemplo real** (ver `predictor-house/main.py`):

```python
df = load_data()           # Carga desde config.RAW_DATA_FILE
df = explore_data(df)      # Imprime head(), info(), describe()
X, y = prepare_data(df)    # Separa features del target
X_train, X_test, y_train, y_test = split_data(X, y)
modelo = train_model(X_train, y_train)
```

## 🐼 Convenciones Pandas

### ❌ NUNCA usar chained assignment con inplace

```python
# ❌ MAL - Causa FutureWarning en pandas 3.0
df['age'].fillna(value, inplace=True)
```

### ✅ SIEMPRE asignación directa

```python
# ✅ BIEN - Compatible con pandas 3.0
df['age'] = df['age'].fillna(value)
```

**Razón**: `df['columna']` puede devolver una vista o copia. El `inplace=True` en chained assignment dejará de funcionar en pandas 3.0.

## 🎨 Patrones de Impresión

Use **output formateado** para claridad (ver `data_preprocessing.py`):

```python
print("=" * 60)
print("🧹 INICIANDO PREPROCESAMIENTO DE DATOS")
print("=" * 60)
print("\n📋 PASO 1: Manejo de valores faltantes")
print("-" * 60)
print(f"✅ Edad: Rellenados {count} valores con mediana ({median:.1f} años)")
```

## 📊 Gestión de Datos

### Dataset Simple (predictor-house)

- **Features**: `tamano_m2`, `habitaciones`, `banos`, `edad_anos`, `distancia_centro_km`
- **Target**: `precio`
- **Fuente**: CSV local (`data/raw/casas.csv`)

### Dataset Titanic

- **Features**: `pclass`, `sex`, `age`, `fare`, `embarked`, `family_size`, `is_alone`
- **Target**: `survived`
- **Fuente**: `sns.load_dataset('titanic')` en `data_loader.py`

### Feature Engineering Pattern

```python
# Crear variables derivadas ANTES de seleccionar columnas
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)
```

## 💾 Persistencia de Modelos

```python
# Guardar con pickle (no joblib en predictor-house, joblib en predictor-titanic)
# predictor-house usa pickle
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(modelo, f)

# predictor-titanic usa joblib
joblib.dump(modelo, MODEL_FILE)

# Cargar (según el proyecto)
with open(MODEL_FILE, 'rb') as f:
    modelo = pickle.load(f)
# O
modelo = joblib.load(MODEL_FILE)
```

### Patrón de Mensajes al Guardar/Cargar

```python
# ✅ SIEMPRE: Mostrar solo nombre de archivo en mensajes
import os

def save_model(model, filepath):
    joblib.dump(model, filepath)  # ← Usa ruta absoluta
    display_name = os.path.basename(filepath)  # ← Extrae nombre
    print(f"💾 Modelo guardado: {display_name}")  # ← Muestra nombre limpio

def load_model(filepath):
    model = joblib.load(filepath)  # ← Usa ruta absoluta
    display_name = os.path.basename(filepath)
    print(f"✅ Modelo cargado: {display_name}")
    return model
```

## 🧪 Testing y Debugging

### Ejecutar scripts individuales:

```powershell
# Desde la raíz del proyecto específico
cd predictor-house
python main.py

# Para módulos individuales (requiere __main__ o script wrapper)
cd predictor-titanic/src
python data_preprocessing.py
```

### No hay tests unitarios implementados

- El directorio `tests/` existe pero está vacío en ambos proyectos
- La validación se hace mediante prints en el pipeline

## 🔧 Dependencias

### predictor-house (especificado):

```
pandas==2.3.2, numpy==2.3.3, scikit-learn==1.7.2
matplotlib==3.10.6, jupyter==1.1.1, streamlit==1.40.1
```

### predictor-titanic (versiones libres):

```
pandas, numpy, matplotlib, seaborn
```

**Instalar**: `pip install -r requirements.txt` (en cada directorio de proyecto)

## 📁 Convenciones de Archivos

- **Datos Raw**: NUNCA modificar archivos en `data/raw/`
- **Modelos**: `.pkl` guardados en `models/`, nombres descriptivos (`modelo_casas.pkl`, `modelo_kaggle_rf.pkl`)
- **Visualizaciones**: PNGs en `reports/figures/` con nombres descriptivos (`feature_analysis.png`)
- **Procesados**: Datos limpios opcionales en `data/processed/`

## 🚨 Errores Comunes a Evitar

1. **Rutas relativas hardcodeadas** → Usar siempre `config.py`
2. **Chained assignment con inplace** → Asignación directa
3. **Ejecutar desde directorio incorrecto** → `cd` al proyecto antes de ejecutar
4. **Modificar datos raw** → Trabajar siempre en copias (`df.copy()`)
5. **Olvidar strip() en columnas CSV** → `df.columns = df.columns.str.strip()`

## 🔍 Para Entender el Código

- **Punto de entrada**: Leer `main.py` de cada proyecto primero
- **Configuración**: Revisar `config.py` para features y rutas
- **Preprocesamiento**: El pipeline de limpieza está en `data_preprocessing.py` (Titanic) o es simple en `data_loader.py` (House)
- **Visualizaciones educativas**: Código en `visualization.py` muestra análisis exploratorio detallado
