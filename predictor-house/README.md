# 🏠 Predictor de Precios de Casas

Proyecto de **Machine Learning de regresión** para predecir precios de viviendas utilizando múltiples algoritmos y datasets (sintético local + Kaggle House Prices Competition).

## 📋 Descripción del Proyecto

Este proyecto implementa un sistema completo de predicción de precios inmobiliarios con tres aproximaciones diferentes:

1. **Modelo Simple**: Regresión lineal con dataset sintético local
2. **Modelo Kaggle**: Regresión lineal con dataset de Kaggle
3. **Modelo Kaggle RF**: Random Forest con dataset de Kaggle (mejor rendimiento)

**Tipo de problema**: Regresión (predicción de valores continuos)  
**Algoritmos**: Regresión Lineal Múltiple, Random Forest  
**Datasets**: Dataset sintético local + Kaggle House Prices Competition  
**Objetivo**: Predecir el precio de una vivienda basándose en sus características

---

## 🎯 Características del Proyecto

### ✨ Funcionalidades Principales

- 📊 **Tres implementaciones comparables** de modelos de regresión
- 🔍 **Análisis exploratorio** completo de datos
- 🧹 **Pipeline de preprocesamiento** automatizado
- 📈 **Visualizaciones** detalladas de features y predicciones
- 💾 **Persistencia de modelos** con pickle
- 🎨 **Aplicación web interactiva** con Streamlit
- 🎯 **Sistema de predicción** para nuevas propiedades

### 🔢 Variables del Dataset Simple

| Variable              | Descripción                | Tipo     | Rango       |
| --------------------- | -------------------------- | -------- | ----------- |
| `tamano_m2`           | Superficie en m²           | Numérico | 70-200 m²   |
| `habitaciones`        | Número de habitaciones     | Entero   | 1-5         |
| `banos`               | Número de baños            | Entero   | 1-4         |
| `edad_anos`           | Antigüedad de la propiedad | Entero   | 1-30 años   |
| `distancia_centro_km` | Distancia al centro        | Decimal  | 1.8-18.0 km |
| `precio` 🎯           | Precio (target)            | Numérico | $145K-$620K |

---

## 📁 Estructura del Proyecto

```
predictor-house/
│
├── data/
│   ├── raw/                         # Datos originales (NUNCA modificar)
│   │   ├── casas.csv                # Dataset sintético local
│   │   ├── train.csv                # Kaggle training data
│   │   ├── test.csv                 # Kaggle test data
│   │   └── data_description.txt     # Descripción features Kaggle
│   └── processed/                   # Datos procesados (opcional)
│
├── src/                             # Código fuente modular
│   ├── __init__.py                  # Package initialization
│   ├── data_loader.py               # Carga y exploración de datos
│   ├── model.py                     # Entrenamiento y evaluación
│   ├── predictor.py                 # Sistema de predicciones
│   ├── visualizations.py            # Gráficos y análisis visual
│   └── explore_kaggle.py            # Exploración dataset Kaggle
│
├── models/                          # Modelos entrenados (.pkl)
│   ├── modelo_casas.pkl             # Modelo con dataset simple
│   ├── modelo_kaggle.pkl            # Modelo Kaggle (regresión lineal)
│   └── modelo_kaggle_rf.pkl         # Modelo Kaggle (Random Forest)
│
├── reports/                         # Reportes y visualizaciones
│   └── figures/
│       ├── feature_analysis.png
│       ├── predictions_vs_actual.png
│       ├── kaggle_predictions.png
│       └── kaggle_rf_predictions.png
│
├── notebooks/                       # Jupyter notebooks (exploración)
├── tests/                           # Tests unitarios (vacío por ahora)
│
├── main.py                          # Pipeline completo (dataset simple)
├── main_kaggle.py                   # Pipeline Kaggle (regresión lineal)
├── main_kaggle_rf.py                # Pipeline Kaggle (Random Forest)
├── app.py                           # Aplicación Streamlit
├── config.py                        # Configuraciones y rutas absolutas
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Este archivo
```

---

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Navegar al directorio del proyecto**

   ```powershell
   cd predictor-house
   ```

2. **Crear entorno virtual (recomendado)**

   ```powershell
   python -m venv venv
   # Activar en Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

3. **Instalar dependencias**

   ```powershell
   pip install -r requirements.txt
   ```

---

## 🎮 Uso del Sistema

### Opción 1: Pipeline Completo (Dataset Simple)

```powershell
python main.py
```

**Este comando ejecuta**:

1. ✅ Carga y exploración de datos desde `casas.csv`
2. ✅ Preparación y limpieza
3. ✅ Análisis visual de características
4. ✅ División train/test (80/20)
5. ✅ Entrenamiento de regresión lineal
6. ✅ Evaluación con métricas (R², MAE, RMSE)
7. ✅ Generación de visualizaciones
8. ✅ Guardado del modelo en `models/modelo_casas.pkl`
9. ✅ Predicciones de ejemplo

### Opción 2: Pipeline Kaggle (Regresión Lineal)

```powershell
python main_kaggle.py
```

**Características**:

- Usa dataset de Kaggle House Prices Competition
- Regresión lineal múltiple
- Genera `modelo_kaggle.pkl`
- Visualización en `kaggle_predictions.png`

### Opción 3: Pipeline Kaggle (Random Forest)

```powershell
python main_kaggle_rf.py
```

**Características**:

- Usa dataset de Kaggle House Prices Competition
- Random Forest Regressor (mejor rendimiento)
- Genera `modelo_kaggle_rf.pkl`
- Visualización en `kaggle_rf_predictions.png`

### Opción 4: Aplicación Web Interactiva

```powershell
streamlit run app.py
```

**Características**:

- Interfaz web amigable
- Predicciones en tiempo real
- Ajuste de parámetros interactivo
- Visualizaciones dinámicas

---

## 📊 Ejemplos de Uso

### Predicciones con el Modelo Simple

```python
import pickle
from src.predictor import predict_new_houses

# Cargar modelo entrenado
with open('models/modelo_casas.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Formato: [tamaño_m2, habitaciones, baños, edad_años, distancia_centro_km]
casas_nuevas = [
    [130, 3, 2, 10, 5.0],  # Casa mediana, 10 años, 5km centro
    [200, 5, 4, 1, 2.0],   # Casa grande, nueva, cercana al centro
    [85, 2, 1, 20, 12.0],  # Casa pequeña, antigua, alejada
]

# Obtener predicciones
predict_new_houses(modelo, casas_nuevas)
```

**Output esperado**:

```
🏠 PREDICCIONES PARA NUEVAS CASAS
═══════════════════════════════════════
Casa 1: [130m², 3 hab, 2 baños, 10 años, 5.0 km]
💰 Precio estimado: $285,430

Casa 2: [200m², 5 hab, 4 baños, 1 año, 2.0 km]
💰 Precio estimado: $547,890
...
```

### Uso Programático del Pipeline

```python
from src.data_loader import load_data, explore_data, prepare_data
from src.model import split_data, train_model, evaluate_model

# Cargar y preparar datos
df = load_data()
df = explore_data(df)
X, y = prepare_data(df)

# Entrenar modelo
X_train, X_test, y_train, y_test = split_data(X, y)
modelo = train_model(X_train, y_train)

# Evaluar
evaluate_model(modelo, X_test, y_test)
```

---

## 🔬 Métricas de Evaluación

### Modelo Simple (Dataset Local)

- **R² Score**: Coeficiente de determinación (bondad de ajuste)
- **MAE** (Mean Absolute Error): Error promedio en dólares
- **RMSE** (Root Mean Squared Error): Penaliza errores grandes

### Modelo Kaggle Random Forest

**Ventajas sobre regresión lineal**:

- ✅ Captura relaciones no lineales
- ✅ Maneja mejor outliers
- ✅ Menos overfitting con parámetros adecuados
- ✅ Feature importance automático

---

## 📈 Visualizaciones Generadas

### 1. Análisis de Características (`feature_analysis.png`)

- Histogramas de distribución
- Gráficos de dispersión vs precio
- Matriz de correlación

### 2. Predicciones vs Valores Reales

- `predictions_vs_actual.png`: Modelo simple
- `kaggle_predictions.png`: Modelo Kaggle lineal
- `kaggle_rf_predictions.png`: Modelo Kaggle RF
- Línea de regresión ideal (y = x)
- Distribución de errores

---

## 🛠️ Tecnologías y Dependencias

```
pandas==2.3.2          # Manipulación de datos
numpy==2.3.3           # Computación numérica
scikit-learn==1.7.2    # Algoritmos de ML
matplotlib==3.10.6     # Visualizaciones
jupyter==1.1.1         # Notebooks interactivos
streamlit==1.40.1      # Aplicación web
```

**Instalar todo**: `pip install -r requirements.txt`

---

## ⚙️ Configuración (`config.py`)

El archivo `config.py` centraliza todas las configuraciones:

```python
# Rutas absolutas (compatibilidad multiplataforma)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Features del modelo
FEATURES = ['tamano_m2', 'habitaciones', 'banos', 'edad_anos', 'distancia_centro_km']
TARGET = 'precio'

# Parámetros de entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

**Nunca usar rutas relativas hardcodeadas** - Siempre referencia `config.py`.

---

## 🎓 Conceptos de ML Implementados

### Técnicas Aplicadas

- ✅ **Regresión Lineal Múltiple**: Relaciones lineales entre features y target
- ✅ **Random Forest Regressor**: Ensemble de árboles de decisión
- ✅ **Train/Test Split**: División estratificada 80/20
- ✅ **Feature Selection**: Selección de variables relevantes
- ✅ **Model Evaluation**: Múltiples métricas de rendimiento
- ✅ **Model Persistence**: Serialización con pickle
- ✅ **Data Visualization**: Análisis exploratorio gráfico

### Pipeline de Trabajo

```
Datos Raw → Exploración → Preparación → Split → Entrenamiento →
Evaluación → Visualización → Persistencia → Predicciones
```

---

## 🧪 Testing

Actualmente el directorio `tests/` está vacío. La validación se realiza mediante:

- **Prints descriptivos** en el pipeline
- **Visualizaciones** de predicciones vs valores reales
- **Métricas cuantitativas** (R², MAE, RMSE)

**Próxima implementación**: Tests unitarios con pytest

---

## 📚 Recursos y Referencias

- **Kaggle Competition**: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Dataset Local**: `data/raw/casas.csv` (sintético)
- **Documentación scikit-learn**: [Regresión Lineal](https://scikit-learn.org/stable/modules/linear_model.html)
- **Streamlit Docs**: [streamlit.io](https://docs.streamlit.io/)

---

## 🐛 Troubleshooting

### Error: "No module named 'src'"

**Solución**: Ejecuta desde el directorio `predictor-house/`

```powershell
cd predictor-house
python main.py
```

### Error: "FileNotFoundError: casas.csv"

**Solución**: Verifica que `data/raw/casas.csv` existe

```powershell
ls data\raw\casas.csv
```

### Advertencia: FutureWarning con pandas

**Solución**: El código ya está actualizado para pandas 3.0 (evita chained assignment)

---

## 🚀 Próximas Mejoras

- [ ] Validación cruzada (k-fold CV)
- [ ] Grid Search para hyperparameter tuning
- [ ] Más algoritmos (XGBoost, Gradient Boosting)
- [ ] API REST con FastAPI
- [ ] Tests unitarios con pytest
- [ ] Feature engineering avanzado
- [ ] Detección automática de outliers
- [ ] Integration con datos inmobiliarios reales

---

## 📄 Licencia

Este proyecto es parte del repositorio [repos-deep-learning](https://github.com/devlitus/repos-deep-learning) bajo licencia MIT.

---

## 👨‍💻 Autor

Proyecto educativo de Machine Learning - Predicción de Precios Inmobiliarios

**Repositorio padre**: [devlitus/repos-deep-learning](https://github.com/devlitus/repos-deep-learning)

---

<div align="center">

💡 **¿Tienes preguntas?** Abre un issue en el repositorio principal

⭐ **¿Te fue útil?** Considera dar una estrella al proyecto

</div>
