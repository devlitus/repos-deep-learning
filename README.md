# 🧠 Deep Learning Repository - Proyectos de Machine Learning

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub repo](https://img.shields.io/badge/GitHub-repos--deep--learning-181717?logo=github)](https://github.com/devlitus/repos-deep-learning)

Repositorio educativo con **tres proyectos independientes de Machine Learning** siguiendo arquitectura modular y mejores prácticas:

- 🏠 **Predictor de Precios de Casas**: Regresión lineal múltiple para estimar precios inmobiliarios
- 🚢 **Predictor de Supervivencia del Titanic**: Clasificación binaria con Random Forest
- 💳 **Detección de Fraude**: Sistema avanzado de detección de transacciones fraudulentas con técnicas de balanceo

Cada proyecto incluye pipeline completo: carga de datos → preprocesamiento → entrenamiento → evaluación → predicciones.

## 🎯 Proyectos Incluidos

### 🏠 Predictor de Precios de Casas (`predictor-house/`)

**Tipo**: Regresión Lineal Múltiple  
**Dataset**: Kaggle House Prices + Dataset sintético local  
**Objetivo**: Predecir precios de viviendas basándose en características físicas y ubicación

**Características clave**:

- 📊 Análisis de 5 variables: tamaño, habitaciones, baños, edad, distancia al centro
- 📈 Regresión lineal con scikit-learn
- 🎨 Visualizaciones de correlaciones y predicciones
- 🔮 Sistema de predicción para nuevas propiedades
- 📦 Tres implementaciones: modelo simple, Kaggle lineal, Kaggle Random Forest

**Métricas de evaluación**: R² Score, MAE, RMSE

### 🚢 Predictor de Supervivencia del Titanic (`predictor-titanic/`)

**Tipo**: Clasificación Binaria  
**Dataset**: Titanic dataset (Seaborn)  
**Objetivo**: Predecir supervivencia de pasajeros del Titanic

**Características clave**:

- 🎫 Análisis de 7 features: clase, sexo, edad, tarifa, embarque, familia
- 🌲 Random Forest Classifier
- 🧹 Pipeline robusto de preprocesamiento (manejo de nulos, encoding)
- 📊 Visualizaciones exploratorias detalladas
- 🎯 Feature engineering (family_size, is_alone)

**Métricas de evaluación**: Accuracy, Precision, Recall, F1-Score, Matriz de confusión

### 💳 Detección de Fraude en Tarjetas de Crédito (`fraude-detection/`)

**Tipo**: Clasificación Binaria con Datos Desbalanceados  
**Dataset**: Credit Card Fraud Detection (Kaggle)  
**Objetivo**: Identificar transacciones fraudulentas en tiempo real

**Características clave**:

- 🎯 **Datos altamente desbalanceados**: <0.2% de fraudes en el dataset
- ⚖️ **Técnicas de balanceo**: SMOTE (Synthetic Minority Oversampling Technique)
- 🌲 Random Forest optimizado para detección de anomalías
- 📊 Análisis exploratorio completo con 4 notebooks interactivos
- 🌐 **Aplicación web Streamlit** con múltiples páginas:
  - Dashboard principal con métricas clave
  - Predicción de transacciones individuales
  - Análisis de datos exploratorio
  - Analytics avanzado con visualizaciones
- 🔄 Pipeline completo: EDA → Preprocesamiento → Balanceo → Entrenamiento → Evaluación
- 💾 Persistencia de datos procesados y splits para reproducibilidad

**Métricas de evaluación**: Accuracy (~99.9%), Precision (95%), Recall (85%), F1-Score (90%), AUC-ROC (0.95)

## 📁 Estructura del Repositorio

```
repos-deep-learning/
│
├── predictor-house/           # 🏠 Proyecto de Regresión (Precios Casas)
│   ├── data/
│   │   ├── raw/              # Datos originales (casas.csv + Kaggle datasets)
│   │   └── processed/        # Datos procesados
│   ├── src/                  # Módulos Python
│   │   ├── data_loader.py    # Carga y exploración
│   │   ├── model.py          # Entrenamiento/evaluación
│   │   ├── predictor.py      # Predicciones
│   │   ├── visualizations.py # Gráficos
│   │   └── explore_kaggle.py # Análisis Kaggle dataset
│   ├── models/               # Modelos entrenados (.pkl)
│   │   ├── modelo_casas.pkl
│   │   ├── modelo_kaggle.pkl
│   │   └── modelo_kaggle_rf.pkl
│   ├── reports/figures/      # Visualizaciones generadas
│   ├── main.py               # Pipeline principal
│   ├── main_kaggle.py        # Pipeline Kaggle (regresión lineal)
│   ├── main_kaggle_rf.py     # Pipeline Kaggle (Random Forest)
│   ├── app.py                # Aplicación Streamlit
│   ├── config.py             # Configuraciones del proyecto
│   └── requirements.txt      # Dependencias específicas
│
├── predictor-titanic/         # 🚢 Proyecto de Clasificación (Titanic)
│   ├── data/                 # (Dataset cargado desde seaborn)
│   ├── src/                  # Módulos Python
│   │   ├── data_loader.py    # Carga dataset Titanic
│   │   ├── data_preprocessing.py  # Limpieza y feature engineering
│   │   ├── model.py          # Entrenamiento Random Forest
│   │   ├── visualization.py  # Análisis exploratorio
│   │   └── predictor.py      # Predicciones
│   ├── models/               # Modelos entrenados
│   │   └── titanic_random_forest.pkl
│   ├── reports/              # Reportes y visualizaciones
│   ├── config.py             # Configuraciones del proyecto
│   ├── main.py               # Pipeline completo
│   └── requirements.txt      # Dependencias específicas
│
├── fraude-detection/          # 💳 Proyecto de Detección de Fraude
│   ├── data/
│   │   ├── raw/              # creditcard.csv (Kaggle)
│   │   └── processed/        # Datos procesados y splits
│   │       ├── X_train_original.csv
│   │       ├── X_train_balanced.csv
│   │       ├── X_val.csv
│   │       ├── X_test.csv
│   │       └── y_*.csv
│   ├── src/
│   │   ├── data/
│   │   │   ├── load.py       # Carga de datos
│   │   │   └── preprocess.py # Preprocesamiento y balanceo
│   │   ├── models/
│   │   │   ├── train.py      # Entrenamiento de modelos
│   │   │   └── evaluate.py   # Evaluación y métricas
│   │   └── visualization/
│   │       └── plots.py      # Gráficos y visualizaciones
│   ├── web/                  # Aplicación Streamlit multipágina
│   │   ├── app.py            # App principal
│   │   ├── pages/            # Páginas individuales
│   │   │   ├── dashboard.py
│   │   │   ├── prediction.py
│   │   │   ├── data_explorer.py
│   │   │   ├── analytics.py
│   │   │   └── about.py
│   │   ├── styles/           # CSS personalizado
│   │   └── utils/            # Utilidades web
│   ├── notebooks/            # 4 notebooks Jupyter
│   │   ├── 01_exploratory_analysis.ipynb
│   │   ├── 02_preprocessing.ipynb
│   │   ├── 03_model_training.ipynb
│   │   └── 04_model_evaluation.ipynb
│   ├── models/               # Modelos y reportes
│   │   ├── final_evaluation_report.csv
│   │   ├── final_evaluation_report.json
│   │   └── model_metrics.csv
│   ├── config/
│   │   └── config.py         # Configuraciones centralizadas
│   ├── main.py               # Pipeline completo
│   ├── verify_installation.py # Verificación de setup
│   └── requirements.txt      # Dependencias específicas
│
├── .github/
│   └── copilot-instructions.md  # Instrucciones para agentes IA
├── LICENSE                   # Licencia MIT
└── README.md                 # Este archivo
```

### 🗂️ Patrón de Arquitectura Común

Los proyectos siguen una estructura modular consistente:

- **`config.py`**: Rutas absolutas, features, hiperparámetros
- **`src/data_loader.py`**: Carga de datos desde fuentes locales/remotas
- **`src/model.py` o `train_model.py`**: Entrenamiento y evaluación
- **`src/visualizations.py`**: Generación de gráficos analíticos
- **`main.py`**: Pipeline completo ejecutable
- **`models/`**: Persistencia de modelos con pickle

## 🚀 Instalación y Uso

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git (para clonar el repositorio)

### Instalación General

1. **Clonar el repositorio**

   ```powershell
   git clone https://github.com/devlitus/repos-deep-learning.git
   cd repos-deep-learning
   ```

2. **Crear entorno virtual (recomendado)**

   ```powershell
   python -m venv venv
   # Activar en Windows PowerShell
   .\venv\Scripts\Activate.ps1
   ```

### 🏠 Ejecutar Predictor de Casas

```powershell
cd predictor-house
pip install -r requirements.txt
python main.py              # Pipeline con dataset simple
python main_kaggle.py       # Pipeline Kaggle (regresión lineal)
python main_kaggle_rf.py    # Pipeline Kaggle (Random Forest)
streamlit run app.py        # Aplicación web interactiva
```

**Output esperado**:

- Modelo entrenado en `models/modelo_casas.pkl`
- Visualizaciones en `reports/figures/`
- Métricas de evaluación en consola

### 🚢 Ejecutar Predictor de Titanic

```powershell
cd predictor-titanic
pip install -r requirements.txt
python main.py              # Pipeline completo
```

**Output esperado**:

- Modelo Random Forest en `models/titanic_random_forest.pkl`
- Análisis detallado con prints formateados
- Matriz de confusión y métricas de clasificación

### 💳 Ejecutar Detección de Fraude

```powershell
cd fraude-detection
pip install -r requirements.txt

# Verificar instalación
python verify_installation.py

# Ejecutar pipeline completo
python main.py

# Lanzar aplicación web
streamlit run web/app.py

# Explorar notebooks
jupyter notebook notebooks/
```

**Output esperado**:

- Datos procesados en `data/processed/` (splits originales y balanceados)
- Modelos entrenados y métricas en `models/`
- Reportes JSON en `reports/eda_insights.json`
- Aplicación web interactiva en `http://localhost:8501`

## 📊 Ejemplos de Uso

### 🏠 Predicciones de Precios de Casas

```python
import pickle
from src.predictor import predict_new_houses

# Cargar modelo entrenado
with open('predictor-house/models/modelo_casas.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Formato: [tamaño_m2, habitaciones, baños, edad_años, distancia_centro_km]
casas_nuevas = [
    [130, 3, 2, 10, 5.0],  # Casa mediana
    [200, 5, 4, 1, 2.0],   # Casa grande y nueva
]

predict_new_houses(modelo, casas_nuevas)
```

### 🚢 Análisis de Supervivencia del Titanic

```python
import joblib
import pandas as pd

# Cargar modelo
modelo = joblib.load('predictor-titanic/models/titanic_random_forest.pkl')

# Ejemplo de pasajero
pasajero = pd.DataFrame({
    'pclass': [1],           # Primera clase
    'sex': [0],              # Mujer (codificado)
    'age': [28],             # 28 años
    'fare': [80],            # Tarifa pagada
    'embarked': [0],         # Cherbourg
    'family_size': [2],      # Viaja con 1 familiar
    'is_alone': [0]          # No viaja solo
})

prediccion = modelo.predict(pasajero)
probabilidad = modelo.predict_proba(pasajero)
print(f"Supervivencia: {'✅ Sobrevivió' if prediccion[0] else '❌ No sobrevivió'}")
print(f"Probabilidad: {probabilidad[0][1]:.2%}")
```

### 💳 Detección de Transacciones Fraudulentas

```python
import joblib
import pandas as pd
import numpy as np

# Cargar modelo entrenado
modelo = joblib.load('fraude-detection/models/fraud_detection_model.pkl')

# Ejemplo de transacción (features V1-V28 + Amount)
# En producción, estas features vienen de PCA del dataset original
transaccion = pd.DataFrame({
    'V1': [-1.359807],
    'V2': [-0.072781],
    # ... V3 a V27 ...
    'V28': [0.014724],
    'Amount': [149.62],
    'Time': [0]
})

# Predecir
prediccion = modelo.predict(transaccion)
probabilidad = modelo.predict_proba(transaccion)

if prediccion[0] == 1:
    print(f"⚠️ FRAUDE DETECTADO - Confianza: {probabilidad[0][1]:.2%}")
else:
    print(f"✅ Transacción legítima - Confianza: {probabilidad[0][0]:.2%}")
```

## 📋 Datasets Utilizados

### Predictor de Casas

| Variable              | Descripción                | Tipo     | Rango       |
| --------------------- | -------------------------- | -------- | ----------- |
| `tamano_m2`           | Superficie en m²           | Numérico | 70-200 m²   |
| `habitaciones`        | Número de habitaciones     | Entero   | 1-5         |
| `banos`               | Número de baños            | Entero   | 1-4         |
| `edad_anos`           | Antigüedad de la propiedad | Entero   | 1-30 años   |
| `distancia_centro_km` | Distancia al centro        | Decimal  | 1.8-18.0 km |
| `precio` 🎯           | Precio (target)            | Numérico | $145K-$620K |

**Fuentes**:

- Dataset sintético local (`casas.csv`)
- Kaggle House Prices Competition (`train.csv`, `test.csv`)

### Predictor de Titanic

| Variable      | Descripción                 | Tipo       | Valores        |
| ------------- | --------------------------- | ---------- | -------------- |
| `pclass`      | Clase del ticket            | Categórico | 1, 2, 3        |
| `sex`         | Sexo del pasajero           | Categórico | male, female   |
| `age`         | Edad del pasajero           | Numérico   | 0.42-80 años   |
| `fare`        | Tarifa pagada               | Numérico   | $0-$512        |
| `embarked`    | Puerto de embarque          | Categórico | C, Q, S        |
| `family_size` | Tamaño familia (engineered) | Entero     | 1-11           |
| `is_alone`    | Viaja solo (engineered)     | Binario    | 0, 1           |
| `survived` 🎯 | Supervivencia (target)      | Binario    | 0 (No), 1 (Sí) |

**Fuente**: Seaborn dataset (`sns.load_dataset('titanic')`)

### Detección de Fraude

| Variable     | Descripción                        | Tipo     | Valores        |
| ------------ | ---------------------------------- | -------- | -------------- |
| `Time`       | Segundos desde primera transacción | Numérico | 0-172,792      |
| `V1` - `V28` | Features de PCA (anónimas)         | Numérico | Normalizados   |
| `Amount`     | Monto de la transacción            | Numérico | $0-$25,691     |
| `Class` 🎯   | Fraude (target)                    | Binario  | 0 (No), 1 (Sí) |

**Características del dataset**:

- **Total transacciones**: 284,807
- **Transacciones fraudulentas**: 492 (0.172%)
- **Desbalance**: 99.83% legítimas vs 0.17% fraudes
- **Features**: 30 (28 de PCA + Time + Amount)
- **Fuente**: Kaggle Credit Card Fraud Detection

**Técnicas de balanceo aplicadas**:

- SMOTE (Synthetic Minority Oversampling Technique)
- Ratio de balanceo: 0.5 (50% minoritaria vs mayoritaria)

## 🔬 Rendimiento de los Modelos

### 🏠 Predictor de Casas

#### Modelo Simple (Regresión Lineal)

- **R² Score**: Coeficiente de determinación
- **MAE**: Error absoluto medio en dólares
- **RMSE**: Raíz del error cuadrático medio
- **Visualizaciones**: Predicciones vs valores reales

#### Modelo Kaggle Random Forest

- **Mejora significativa** sobre regresión lineal simple
- **Tres implementaciones** para comparación de rendimiento
- **Exportación de predicciones** para submission en Kaggle

### 🚢 Predictor de Titanic

#### Random Forest Classifier

- **Accuracy**: Precisión general del modelo
- **Precision/Recall**: Por clase (sobrevivió/no sobrevivió)
- **F1-Score**: Media armónica de precision y recall
- **Matriz de Confusión**: Visualización de predicciones correctas/incorrectas
- **Feature Importance**: Variables más relevantes para la predicción

### 💳 Detección de Fraude

#### Random Forest Optimizado con SMOTE

| Métrica       | Valor | Descripción                                     |
| ------------- | ----- | ----------------------------------------------- |
| **Accuracy**  | 99.9% | Precisión general en todas las transacciones    |
| **Precision** | 95%   | De las predichas como fraude, 95% son correctas |
| **Recall**    | 85%   | Detecta 85% de los fraudes reales               |
| **F1-Score**  | 90%   | Balance entre precisión y cobertura             |
| **AUC-ROC**   | 0.95  | Excelente capacidad de discriminación           |

**Ventajas del enfoque**:

- ✅ Manejo efectivo de datos desbalanceados con SMOTE
- ✅ Alta precisión minimiza falsos positivos costosos
- ✅ Buen recall para capturar la mayoría de fraudes
- ✅ Random Forest robusto ante ruido y outliers
- ✅ Validación con conjunto separado sin balancear

**Consideraciones en producción**:

- **Costo de falsos negativos**: Un fraude no detectado puede costar más que revisar falsos positivos
- **Threshold ajustable**: Se puede ajustar el umbral de decisión según el costo del negocio
- **Monitoreo continuo**: Los patrones de fraude evolucionan, requiere reentrenamiento periódico

## 📈 Visualizaciones y Análisis

### Predictor de Casas

1. **Análisis de Características** (`feature_analysis.png`)

   - Distribución de variables independientes
   - Correlaciones entre features y precio
   - Detección de outliers

2. **Predicciones del Modelo**
   - `predictions_vs_actual.png`: Comparación predicciones vs reales
   - `kaggle_predictions.png`: Resultados modelo Kaggle lineal
   - `kaggle_rf_predictions.png`: Resultados Random Forest Kaggle
   - Línea de regresión ideal y distribución de errores

### Predictor de Titanic

1. **Análisis Exploratorio** (generado por `visualization.py`)

   - Distribución de supervivencia por clase, sexo, edad
   - Análisis de tarifas pagadas
   - Impacto del puerto de embarque
   - Visualización de valores faltantes

2. **Feature Engineering**

   - Creación de `family_size` y `is_alone`
   - One-hot encoding de variables categóricas
   - Imputación inteligente de edad (mediana por clase/sexo)

3. **Evaluación del Modelo**
   - Matriz de confusión detallada
   - Reporte de clasificación completo
   - Análisis de importancia de features

### Detección de Fraude

1. **Análisis Exploratorio de Datos** (EDA)

   - **Distribución de clases**: Visualización del desbalance
   - **Distribución de montos**: Comparación fraude vs legítimas
   - **Análisis temporal**: Patrones de fraude en el tiempo
   - **Correlaciones**: Heatmap de features V1-V28
   - **Boxplots**: Detección de outliers por clase

2. **Preprocesamiento y Balanceo**

   - Visualización del efecto de SMOTE
   - Comparación distribuciones antes/después de balanceo
   - Análisis de splits (train/validation/test)

3. **Evaluación del Modelo**

   - **Matriz de confusión**: Predicciones en conjunto de test
   - **Curva ROC**: AUC = 0.95
   - **Curva Precision-Recall**: Especialmente importante en datos desbalanceados
   - **Feature Importance**: Features más relevantes para detección
   - **Distribución de probabilidades**: Separación entre clases

4. **Dashboard Interactivo** (Streamlit)
   - Métricas clave en tiempo real
   - Visualizaciones interactivas de rendimiento
   - Análisis de predicciones individuales
   - Explorador de datos con filtros
   - Analytics avanzado con insights del modelo

## 🛠️ Tecnologías Utilizadas

### Core ML Stack

- **Python 3.8+**: Lenguaje de programación principal
- **pandas**: Manipulación y análisis de datos
- **numpy**: Computación numérica y álgebra lineal
- **scikit-learn**: Algoritmos de machine learning (regresión, clasificación, RF)
- **matplotlib**: Visualización de datos y gráficos
- **seaborn**: Visualizaciones estadísticas avanzadas

### Versiones Específicas

#### predictor-house

- `pandas==2.3.2`
- `numpy==2.3.3`
- `scikit-learn==1.7.2`
- `matplotlib==3.10.6`
- `jupyter==1.1.1`
- `streamlit==1.40.1`

#### predictor-titanic

- Versiones libres de pandas, numpy, matplotlib, seaborn, scikit-learn

#### fraude-detection

- `pandas`, `numpy`, `scikit-learn`
- `imbalanced-learn`: Para técnicas de balanceo (SMOTE)
- `matplotlib`, `seaborn`, `plotly`: Visualizaciones
- `streamlit`: Aplicación web multipágina
- `jupyter`: Notebooks interactivos

### Herramientas de Desarrollo

- **pickle/joblib**: Serialización de modelos entrenados
- **Jupyter Notebooks**: Exploración interactiva de datos
- **Streamlit**: Aplicaciones web interactivas
- **Git/GitHub**: Control de versiones y colaboración
- **imbalanced-learn**: Balanceo de clases desbalanceadas

## 🎓 Conceptos de Machine Learning Aplicados

### Técnicas Implementadas

- **Regresión Lineal Múltiple**: Predicción de valores continuos (precios de casas)
- **Random Forest**:
  - Ensemble learning para clasificación binaria (Titanic, Fraude)
  - Mejora de regresión lineal en datos Kaggle (House Prices)
  - Robusto ante datos desbalanceados y outliers
- **Feature Engineering**:
  - Creación de variables derivadas (family_size, is_alone)
  - Análisis de importancia de features
- **Data Preprocessing**:
  - Manejo de valores faltantes (mediana, moda, estrategias inteligentes)
  - Encoding de variables categóricas (LabelEncoder, One-Hot)
  - Feature scaling y normalización
  - **Técnicas de balanceo**: SMOTE para datos altamente desbalanceados
- **Train/Test/Validation Split**: División estratificada de datos
- **Model Persistence**: Serialización con pickle/joblib
- **Métricas de Evaluación**:
  - Regresión: R², MAE, RMSE
  - Clasificación: Accuracy, Precision, Recall, F1-Score, AUC-ROC
  - Confusion Matrix y Classification Report

### Desafíos Resueltos

#### 🏠 Predictor de Casas

- **Multicolinealidad**: Análisis de correlaciones entre features
- **Overfitting**: Validación con train/test split
- **Escalabilidad**: Tres versiones (simple, Kaggle lineal, Kaggle RF)

#### 🚢 Predictor de Titanic

- **Datos faltantes**: Imputación inteligente por grupos
- **Features categóricas**: Encoding apropiado
- **Feature engineering**: Creación de variables significativas
- **Interpretabilidad**: Análisis de feature importance

#### 💳 Detección de Fraude

- **Datos altamente desbalanceados**: 99.83% vs 0.17%
  - Solución: SMOTE para generar muestras sintéticas de la clase minoritaria
- **Alta dimensionalidad**: 30 features (28 de PCA)
  - Solución: Random Forest maneja bien alta dimensionalidad
- **Costo asimétrico de errores**:
  - Falso negativo (fraude no detectado) es más costoso que falso positivo
  - Solución: Optimización de threshold y métricas enfocadas en recall
- **Generalización**:
  - Validación en conjunto sin balancear para simular producción
  - Monitoreo de métricas en datos reales
- **Interpretabilidad**:
  - Feature importance para entender patrones de fraude
  - Visualizaciones de distribuciones por clase

### Buenas Prácticas Aplicadas

✅ **Arquitectura Modular**: Separación clara de responsabilidades (carga, preprocesamiento, modelo, visualización)  
✅ **Configuración Centralizada**: `config.py` con rutas absolutas y parámetros  
✅ **Reproducibilidad**: `random_state` fijo para splits y modelos  
✅ **Pandas 3.0 Ready**: Evita chained assignment con `inplace=True`  
✅ **Logging Descriptivo**: Prints formateados con emojis y separadores  
✅ **Persistencia de Modelos**: Guardado/carga con pickle/joblib para reutilización  
✅ **Documentación**: READMEs detallados e instrucciones para agentes IA  
✅ **Validación Realista**: En fraude-detection, validación en datos sin balancear  
✅ **Múltiples Notebooks**: Separación de análisis, preprocesamiento, entrenamiento y evaluación

## 👨‍💻 Autor y Propósito

**Deep Learning Repository** - Colección educativa de proyectos de Machine Learning

Este repositorio fue creado con fines educativos para demostrar:

- Implementación end-to-end de proyectos de ML (regresión, clasificación, detección de anomalías)
- Arquitectura modular y escalable
- Buenas prácticas de desarrollo en Data Science
- Manejo de diferentes tipos de problemas:
  - **Regresión**: Predicción de valores continuos
  - **Clasificación balanceada**: Datos con distribución similar entre clases
  - **Clasificación desbalanceada**: Detección de eventos raros con técnicas de balanceo
- Pipeline completo desde datos crudos hasta aplicaciones web interactivas
- Aplicación de técnicas avanzadas (SMOTE, feature engineering, hyperparameter tuning)

### 📚 Recursos Educativos

- **Kaggle Competitions**:
  - [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
  - [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Datasets públicos**:
  - Titanic incluido en Seaborn (`sns.load_dataset('titanic')`)
- **Instrucciones IA**: `.github/copilot-instructions.md` para agentes de desarrollo

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [`LICENSE`](LICENSE) para más detalles.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

**Guías de contribución**:

- Seguir la estructura modular existente
- Actualizar READMEs si se añaden features
- Mantener compatibilidad con pandas 3.0 (evitar chained assignment)
- Documentar funciones y añadir prints descriptivos

---

**¿Tienes preguntas?** Abre un [issue](https://github.com/devlitus/repos-deep-learning/issues) en el repositorio.

---

<div align="center">

Hecho con ❤️ para la comunidad de Data Science

[![Star on GitHub](https://img.shields.io/github/stars/devlitus/repos-deep-learning?style=social)](https://github.com/devlitus/repos-deep-learning)

</div>

## 📝 Próximas Mejoras

### Predictor de Casas

- [ ] Validación cruzada (k-fold CV)
- [ ] Hyperparameter tuning (GridSearchCV)
- [ ] Más algoritmos (XGBoost, Gradient Boosting)
- [ ] API REST con FastAPI
- [ ] Dashboard avanzado con Plotly/Dash
- [ ] Feature importance analysis
- [ ] Detección automática de outliers

### Predictor de Titanic

- [ ] Comparación con otros clasificadores (SVM, Gradient Boosting)
- [ ] Optimización de hiperparámetros Random Forest
- [ ] Feature engineering avanzado (títulos en nombres, cabinas)
- [ ] Análisis de SHAP values
- [ ] Cross-validation para robustez
- [ ] Deployment como servicio web

### Detección de Fraude

- [x] Pipeline completo de preprocesamiento
- [x] Implementación de SMOTE para balanceo
- [x] Aplicación web Streamlit multipágina
- [x] 4 notebooks Jupyter documentados
- [x] Persistencia de datos procesados
- [ ] Comparación con otros modelos (XGBoost, LightGBM, Neural Networks)
- [ ] Hyperparameter tuning con RandomizedSearchCV
- [ ] Análisis de SHAP values para interpretabilidad
- [ ] Detección de concept drift y reentrenamiento automático
- [ ] API REST para integración en sistemas de producción
- [ ] Sistema de alertas en tiempo real
- [ ] Análisis de series temporales de fraude
- [ ] Feature engineering avanzado sobre features PCA
- [ ] Ensemble methods (stacking, voting)
- [ ] Monitoreo de performance en producción

### General

- [ ] Tests unitarios completos (pytest)
- [ ] CI/CD pipeline con GitHub Actions
- [ ] Dockerización de los tres proyectos
- [ ] Notebooks Jupyter documentados (completo en fraude-detection)
- [ ] Logging avanzado con `logging` module
- [ ] Documentación con MkDocs o Sphinx
- [ ] Integración con MLflow para tracking de experimentos
