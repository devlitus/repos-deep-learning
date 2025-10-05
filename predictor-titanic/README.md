# 🚢 Predictor de Supervivencia del Titanic

Proyecto de **Machine Learning de clasificación binaria** para predecir la supervivencia de pasajeros del Titanic utilizando Random Forest Classifier.

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de clasificación para predecir si un pasajero del Titanic sobrevivió o no, basándose en características como clase de ticket, edad, sexo, tarifa pagada, entre otras.

**Tipo de problema**: Clasificación Binaria  
**Algoritmo**: Random Forest Classifier  
**Dataset**: Titanic dataset de Seaborn  
**Objetivo**: Maximizar la precisión en la predicción de supervivencia

## 🎯 Características del Modelo

### Variables de Entrada (Features)

- **pclass**: Clase del ticket (1ra, 2da, 3ra clase)
- **sex**: Sexo del pasajero
- **age**: Edad del pasajero
- **fare**: Tarifa pagada
- **embarked**: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)
- **family_size**: Tamaño de la familia (calculado: sibsp + parch + 1)
- **is_alone**: Si viaja solo (calculado: family_size == 1)

### Variable Objetivo (Target)

- **survived**: 0 = No sobrevivió, 1 = Sobrevivió

## 📁 Estructura del Proyecto

```
predictor-titanic/
│
├── data/                    # Directorio para datos (actualmente vacío, se usa dataset de seaborn)
│
├── src/                     # Código fuente modular
│   ├── data_loader.py       # Carga del dataset desde seaborn
│   ├── data_preprocessing.py # Limpieza y preparación de datos
│   ├── train_model.py       # Entrenamiento y evaluación del modelo
│   ├── visualization.py     # Análisis exploratorio y gráficas
│   └── alanizis_year.py     # Script de análisis personalizado
│
├── models/                  # Modelos entrenados guardados
│   └── titanic_random_forest.pkl
│
├── notebooks/               # Jupyter notebooks para exploración
│
├── reports/                 # Reportes y visualizaciones
│   └── figures/
│
├── test/                    # Tests unitarios (vacío)
│
├── config.py               # Configuraciones del proyecto (vacío - pendiente)
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## 🛠️ Instalación

### 1. Clonar el repositorio

```powershell
cd c:\dev\repos-deep-learning\predictor-titanic
```

### 2. Crear entorno virtual (recomendado)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```powershell
pip install -r requirements.txt
```

**Dependencias principales:**

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn (implícito en train_model.py)

## 🚀 Uso del Proyecto

### Flujo de Trabajo Completo

#### 1. Exploración de Datos

```powershell
cd src
python data_loader.py
```

Carga el dataset del Titanic desde seaborn y muestra estadísticas básicas.

#### 2. Preprocesamiento de Datos

```powershell
python data_preprocessing.py
```

Ejecuta el pipeline de limpieza:

- Manejo de valores faltantes (age, embarked, fare)
- Codificación de variables categóricas
- Feature engineering (family_size, is_alone)
- Selección de features relevantes

#### 3. Visualización Exploratoria

```powershell
python visualization.py
```

Genera análisis visual:

- Distribución de supervivencia
- Análisis por clase, sexo, edad
- Correlaciones entre variables
- Matrices de confusión

#### 4. Entrenamiento del Modelo

```powershell
python train_model.py
```

Entrena el modelo Random Forest y muestra métricas:

- Accuracy (Exactitud)
- Precision (Precisión)
- Recall (Sensibilidad)
- F1-Score
- Matriz de confusión
- Classification Report

### Módulos Individuales

Cada módulo puede ejecutarse independientemente desde el directorio `src/`:

```powershell
# Cargar datos
python data_loader.py

# Preprocesar
python data_preprocessing.py

# Visualizar
python visualization.py

# Entrenar modelo
python train_model.py

# Análisis personalizado
python alanizis_year.py
```

## 📊 Pipeline de Datos

```
1. load_data() (data_loader.py)
   └─> Carga dataset desde sns.load_dataset('titanic')

2. preprocess_titanic_data() (data_preprocessing.py)
   ├─> Manejo de valores faltantes
   │   ├─> age: Rellena con mediana
   │   ├─> embarked: Rellena con moda
   │   └─> fare: Rellena con mediana
   ├─> Feature Engineering
   │   ├─> family_size = sibsp + parch + 1
   │   └─> is_alone = (family_size == 1)
   ├─> Codificación de variables categóricas
   │   ├─> sex: One-Hot Encoding (male/female)
   │   └─> embarked: One-Hot Encoding (C/Q/S)
   └─> Selección de features relevantes

3. split_features_target() (data_preprocessing.py)
   └─> Separa X (features) e y (target)

4. train_test_split() (train_model.py)
   └─> Divide en conjuntos de entrenamiento (80%) y prueba (20%)

5. train_random_forest_classifier() (train_model.py)
   └─> Entrena modelo Random Forest

6. evaluate_model() (train_model.py)
   └─> Calcula métricas de rendimiento

7. save_model() (train_model.py)
   └─> Guarda modelo en models/titanic_random_forest.pkl
```

## 🔬 Preprocesamiento de Datos

### Manejo de Valores Faltantes

```python
# Age: Mediana por sexo y clase
df['age'] = df['age'].fillna(df.groupby(['sex', 'pclass'])['age'].transform('median'))

# Embarked: Moda
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Fare: Mediana
df['fare'] = df['fare'].fillna(df['fare'].median())
```

### Feature Engineering

```python
# Tamaño de la familia
df['family_size'] = df['sibsp'] + df['parch'] + 1

# Indicador de viaje solo
df['is_alone'] = (df['family_size'] == 1).astype(int)
```

### Codificación de Variables Categóricas

- **sex**: One-Hot Encoding → `male`, `female`
- **embarked**: One-Hot Encoding → `embarked_C`, `embarked_Q`, `embarked_S`

## 📈 Métricas del Modelo

El modelo Random Forest se evalúa con las siguientes métricas:

- **Accuracy**: Proporción de predicciones correctas
- **Precision**: De los predichos como sobrevivientes, cuántos realmente lo fueron
- **Recall**: De los sobrevivientes reales, cuántos fueron detectados
- **F1-Score**: Media armónica entre Precision y Recall
- **Confusion Matrix**: Matriz de predicciones correctas e incorrectas

Ejemplo de salida:

```
🎯 MÉTRICAS DEL MODELO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Accuracy:  0.8324
📊 Precision: 0.8156
🎯 Recall:    0.7612
⚖️ F1-Score:  0.7875
```

## 💾 Persistencia del Modelo

El modelo entrenado se guarda usando pickle:

```python
import pickle

# Guardar modelo
with open('models/titanic_random_forest.pkl', 'wb') as f:
    pickle.dump(modelo, f)

# Cargar modelo
with open('models/titanic_random_forest.pkl', 'rb') as f:
    modelo = pickle.load(f)
```

## 🎨 Visualizaciones

El módulo `visualization.py` genera gráficas para análisis exploratorio:

- Distribución de supervivencia por clase
- Distribución de supervivencia por sexo
- Histogramas de edad y tarifa
- Mapas de correlación
- Análisis de features categóricas vs supervivencia

## ⚠️ Convenciones Importantes

### ✅ Buenas Prácticas Implementadas

1. **Sin Chained Assignment con inplace**:

   ```python
   # ✅ CORRECTO
   df['age'] = df['age'].fillna(value)

   # ❌ EVITAR (deprecated en pandas 3.0)
   df['age'].fillna(value, inplace=True)
   ```

2. **Asignación Directa**:

   ```python
   # ✅ Compatible con pandas 3.0
   df['nueva_columna'] = df['columna_a'] + df['columna_b']
   ```

3. **Feature Engineering antes de selección**:
   ```python
   # Primero crear features derivadas
   df['family_size'] = df['sibsp'] + df['parch'] + 1
   # Luego seleccionar columnas
   features = ['pclass', 'age', 'family_size']
   ```

## 📝 Notas de Desarrollo

### Pendientes

- [ ] Completar archivo `config.py` con rutas y configuraciones
- [ ] Crear script `main.py` con pipeline completo
- [ ] Implementar tests unitarios en directorio `test/`
- [ ] Agregar módulo `predictor.py` para predicciones con modelo guardado
- [ ] Documentar notebooks de exploración

### Diferencias con predictor-house

- No usa archivo CSV local (dataset viene de seaborn)
- Incluye más feature engineering (family_size, is_alone)
- Preprocesamiento más complejo (múltiples variables categóricas)
- Clasificación binaria vs regresión

## 🔧 Solución de Problemas

### Error: "No module named 'sklearn'"

```powershell
pip install scikit-learn
```

### Error: "No module named 'seaborn'"

```powershell
pip install seaborn
```

### Error al importar desde src/

Asegúrate de ejecutar los scripts desde el directorio correcto:

```powershell
cd c:\dev\repos-deep-learning\predictor-titanic\src
python train_model.py
```

## 📚 Referencias

- **Dataset**: Titanic dataset incluido en Seaborn
- **Algoritmo**: Random Forest Classifier de scikit-learn
- **Documentación sklearn**: https://scikit-learn.org/stable/

## 👤 Autor

Proyecto educativo de Machine Learning - repos-deep-learning

## 📄 Licencia

Ver archivo LICENSE en la raíz del repositorio.

---

**Última actualización**: Octubre 2025
