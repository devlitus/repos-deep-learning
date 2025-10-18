# 🎓 PROYECTO COMPLETO: PREDICCIÓN DE TEMPERATURA CON LSTM

## 📚 RESUMEN EJECUTIVO

Has creado un **sistema completo de Machine Learning** para predecir temperaturas usando Deep Learning.

---

## ✅ LO QUE HEMOS CONSTRUIDO

### 📁 Archivos Creados (8 archivos principales)

1. **`config.py`** - Configuración global
2. **`data/load_data.py`** - Carga y normalización de datos
3. **`src/preprocessing.py`** - Preprocesamiento (secuencias)
4. **`src/model.py`** - Arquitectura LSTM
5. **`src/evaluation.py`** - Métricas de evaluación
6. **`src/visualization.py`** - Gráficas
7. **`train.py`** - Script principal (ejecuta todo)
8. **`README.md`** - Documentación completa

---

## 🧠 CONCEPTOS APRENDIDOS

### 1. Machine Learning Básico

**¿Qué es?**

- Enseñar a computadoras a aprender de datos
- Sin programación explícita de reglas

**Tipos**:

- Supervisado ✅ (este proyecto)
- No supervisado
- Refuerzo

**Pipeline completo**:

```
Datos → Preprocesamiento → Modelo → Entrenamiento → Evaluación → Predicción
```

---

### 2. Deep Learning

**¿Qué son las Redes Neuronales?**

- Capas de neuronas artificiales
- Aprenden representaciones complejas
- Inspiradas en el cerebro humano

**Arquitectura de este proyecto**:

```
Input (60 días)
    ↓
LSTM Layer 1 (50 neuronas) ← Memoria a largo plazo
    ↓
Dropout (20%) ← Previene overfitting
    ↓
LSTM Layer 2 (50 neuronas) ← Más memoria
    ↓
Dropout (20%)
    ↓
Dense (1 neurona) ← Predicción final
    ↓
Output (temperatura día 61)
```

---

### 3. LSTM (Long Short-Term Memory)

**¿Para qué sirve?**

- Procesar secuencias temporales
- Recordar patrones a largo plazo
- Olvidar información irrelevante

**¿Por qué LSTM para temperaturas?**

```
Día 1: 15°C ─┐
Día 2: 16°C  ├─→ LSTM detecta: "tendencia al alza"
Día 3: 17°C ─┘
Día 4: ¿?    → Predice: ~18°C
```

**Ventajas**:

- ✅ Captura dependencias temporales
- ✅ Maneja secuencias largas
- ✅ Evita problema de gradiente desvaneciente

---

### 4. Preprocesamiento de Datos

**Pasos clave**:

1. **Normalización**:

   ```
   Original: 0°C - 26°C
   Normalizado: 0.0 - 1.0

   ¿Por qué? Redes neuronales aprenden mejor con valores pequeños
   ```

2. **Ventanas Deslizantes**:

   ```
   [día1...día60] → día61
   [día2...día61] → día62
   [día3...día62] → día63
   ...

   Creamos 3,590 secuencias de entrenamiento
   ```

3. **División de Datos**:
   ```
   Train (70%): 2,513 secuencias → Para entrenar
   Val (15%):   539 secuencias   → Para ajustar
   Test (15%):  538 secuencias   → Para evaluar
   ```

---

### 5. Métricas de Evaluación

**4 métricas principales**:

| Métrica  | Fórmula                           | Interpretación       |
| -------- | --------------------------------- | -------------------- |
| **RMSE** | √(promedio(errores²))             | Error promedio en °C |
| **MAE**  | promedio(\|errores\|)             | Error absoluto       |
| **MAPE** | promedio(\|error/real\|) × 100    | Error en %           |
| **R²**   | 1 - (error_modelo/error_promedio) | % varianza explicada |

**Ejemplo de buenos resultados**:

```
RMSE: 1.87°C  → Me equivoco ~2°C
MAE:  1.45°C  → Error absoluto de 1.5°C
MAPE: 7.23%   → Error del 7%
R²:   0.8542  → Explico el 85% de variación
```

---

### 6. Visualización de Resultados

**5 gráficas creadas**:

1. **Temperatura Histórica**:

   - 10 años de datos
   - Detecta patrones estacionales

2. **Progreso de Entrenamiento**:

   - Pérdida vs épocas
   - Detecta overfitting

3. **Predicciones vs Realidad**:

   - 2 líneas superpuestas
   - Ver qué tan bien sigue el modelo

4. **Gráfica de Dispersión**:

   - Cada punto = (real, predicción)
   - Detecta sesgos

5. **Análisis de Errores**:
   - Errores en el tiempo
   - Histograma de distribución

---

## 🚀 CÓMO USAR EL PROYECTO

### Opción 1: Ejecución Completa

```bash
# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo (5-10 minutos)
python train.py
```

**Salida esperada**:

```
📂 PASO 1: CARGANDO DATOS
✅ Datos cargados: 3650 días

🔧 PASO 3: PREPROCESANDO DATOS
✅ Secuencias creadas: 3590

🧠 PASO 4: CONSTRUYENDO MODELO LSTM
✅ Modelo construido: 15,551 parámetros

🏋️ PASO 5: ENTRENANDO MODELO
Epoch 1/50 ... loss: 0.0234 - val_loss: 0.0189
Epoch 2/50 ... loss: 0.0156 - val_loss: 0.0145
...
✅ Entrenamiento completado!

📊 PASO 6: EVALUANDO MODELO
RMSE: 1.87°C
R²: 0.8542
✅ Excelente rendimiento!

📈 PASO 7: CREANDO VISUALIZACIONES
✅ 5 gráficas guardadas

💾 PASO 8: GUARDANDO MODELO
✅ Modelo guardado: models/lstm_temperatura.keras

🎉 PIPELINE COMPLETADO EXITOSAMENTE
```

### Opción 2: Uso Programático

```python
# Entrenar
from train import train_model
model, history, metrics = train_model()

# Cargar modelo guardado
from tensorflow import keras
model = keras.models.load_model('models/lstm_temperatura.keras')

# Hacer predicción
import numpy as np
secuencia = np.array([...])  # 60 temperaturas normalizadas
prediccion = model.predict(secuencia)
```

---

## 📊 ARCHIVOS GENERADOS

Después de ejecutar `train.py`:

```
📁 models/
   └── lstm_temperatura.keras    (2-3 MB) - Modelo entrenado

📁 reports/
   ├── temperatura_historica.png - Datos originales
   ├── entrenamiento.png         - Progreso
   ├── predicciones.png          - Comparación
   ├── scatter.png               - Dispersión
   ├── errores.png               - Análisis
   └── metricas.txt              - Reporte numérico
```

---

## 🔍 ANÁLISIS DETALLADO DE CADA MÓDULO

### 1. `config.py`

**Propósito**: Centralizar configuración

```python
SEQUENCE_LENGTH = 60    # Días para predecir
TRAIN_SPLIT = 0.7       # 70% entrenamiento
VAL_SPLIT = 0.15        # 15% validación
TEST_SPLIT = 0.15       # 15% prueba
```

**¿Por qué?**

- Un solo lugar para cambiar parámetros
- Evita valores mágicos dispersos
- Fácil experimentación

---

### 2. `data/load_data.py`

**Funciones**:

- `load_melbourne_data()` → Carga CSV completo

**Proceso**:

```
1. Leer CSV (3650 filas)
2. Verificar valores faltantes
3. Normalizar (0-1)
4. Retornar df, data, scaler
```

**Conceptos clave**:

- **MinMaxScaler**: Normalización lineal
- **DataFrame**: Estructura de pandas
- **NumPy Array**: Array numérico eficiente

---

### 3. `src/preprocessing.py`

**Funciones**:

- `create_sequences()` → Ventanas deslizantes
- `split_data()` → División train/val/test

**Proceso de secuencias**:

```python
Entrada: [1, 2, 3, 4, 5, 6, 7, 8]
sequence_length = 3

Salida:
X = [[1, 2, 3],
     [2, 3, 4],
     [3, 4, 5],
     [4, 5, 6],
     [5, 6, 7]]

y = [4, 5, 6, 7, 8]
```

**¿Por qué ventanas deslizantes?**

- El modelo necesita ver "contexto" (días anteriores)
- Simula predicción secuencial real
- Aumenta cantidad de datos de entrenamiento

---

### 4. `src/model.py`

**Función**: `build_lstm_model()`

**Arquitectura**:

```python
Input: (None, 60, 1)  # batch, timesteps, features
   ↓
LSTM(50, return_sequences=True)  # Primera capa
   ↓
Dropout(0.2)  # 20% de neuronas apagadas
   ↓
LSTM(50)  # Segunda capa
   ↓
Dropout(0.2)
   ↓
Dense(1)  # Salida
   ↓
Output: (None, 1)  # Predicción
```

**Parámetros totales**: ~15,551

**Cálculo**:

```
LSTM 1: 50 × (1 + 50 + 1) × 4 = 10,400
LSTM 2: 50 × (50 + 50 + 1) × 4 = 20,200
Dense:  50 × 1 + 1 = 51
Total ≈ 15,551 parámetros
```

---

### 5. `src/evaluation.py`

**Funciones**:

- `calculate_metrics()` → Calcula 5 métricas
- `print_metrics()` → Muestra formateado
- `evaluate_model()` → Pipeline completo
- `create_evaluation_report()` → Guarda .txt

**Métricas implementadas**:

1. **MSE**: `mean_squared_error()`
2. **RMSE**: `√MSE`
3. **MAE**: `mean_absolute_error()`
4. **MAPE**: `mean(|error/real|) × 100`
5. **R²**: `r2_score()`

---

### 6. `src/visualization.py`

**5 funciones de gráficas**:

```python
plot_temperature_history()   # Histórico
plot_training_history()      # Entrenamiento
plot_predictions()           # Comparación
plot_prediction_scatter()    # Dispersión
plot_errors()                # Errores
```

**Librerías**:

- `matplotlib.pyplot` → Gráficas
- `numpy` → Operaciones

**Configuración**:

- DPI: 300 (alta calidad)
- Formato: PNG
- Tamaño: Variable según tipo

---

### 7. `train.py` (SCRIPT PRINCIPAL)

**Pipeline de 8 pasos**:

```python
def train_model():
    # 1. Cargar datos
    df, data, scaler = load_melbourne_data()

    # 2. Visualizar histórico
    plot_temperature_history(df)

    # 3. Preprocesar
    X, y = create_sequences(data)
    X_train, X_val, X_test, ... = split_data(X, y)

    # 4. Construir modelo
    model = build_lstm_model()

    # 5. Entrenar
    history = model.fit(X_train, y_train, ...)

    # 6. Evaluar
    predictions, metrics = evaluate_model(...)

    # 7. Visualizar resultados
    plot_predictions(...)
    plot_scatter(...)
    plot_errors(...)

    # 8. Guardar
    model.save('models/lstm_temperatura.keras')

    return model, history, metrics
```

---

## 💡 CONCEPTOS TÉCNICOS EXPLICADOS

### ¿Qué es un epoch?

**Epoch** = Una pasada completa por todos los datos de entrenamiento

```
Datos: 2,513 secuencias
Batch size: 32

1 epoch = 2,513 / 32 = ~79 batches
50 epochs = 50 × 79 = 3,950 actualizaciones del modelo
```

---

### ¿Qué es batch_size?

**Batch** = Cantidad de ejemplos procesados antes de actualizar pesos

```
batch_size = 32

El modelo procesa:
[seq1, seq2, ..., seq32] → actualiza pesos
[seq33, seq34, ..., seq64] → actualiza pesos
...
```

**¿Por qué 32?**

- 1: Muy lento (actualiza c/ejemplo)
- 2513: Muy grande (no cabe en memoria)
- 32: Balance perfecto ✅

---

### ¿Qué es Dropout?

**Dropout** = Apagar aleatoriamente neuronas durante entrenamiento

```
Sin Dropout:
○─○─○─○  Todas las neuronas activas
         ↓ Puede memorizar

Con Dropout 20%:
○─●─○─●  20% apagadas aleatoriamente
         ↓ Forzado a generalizar
```

**Ventaja**: Previene overfitting

---

### ¿Qué es return_sequences?

```python
LSTM(50, return_sequences=True)   # Devuelve secuencia completa
LSTM(50, return_sequences=False)  # Devuelve solo último valor
```

**Ejemplo**:

```
Input: [t1, t2, t3, t4, t5]

return_sequences=True:
Output: [h1, h2, h3, h4, h5]  # 5 outputs

return_sequences=False:
Output: [h5]  # Solo último
```

**¿Cuándo usar cada uno?**

- `True`: Cuando tienes otra capa LSTM después
- `False`: Para la última capa LSTM

---

### ¿Qué es shuffle=False?

```python
model.fit(..., shuffle=False)
```

**¿Por qué NO mezclar?**

Series temporales tienen ORDEN:

```
Con shuffle=True (MAL):
Día 50 → Día 2 → Día 100 → Día 5
         ↓ Pierde orden temporal

Con shuffle=False (BIEN):
Día 1 → Día 2 → Día 3 → Día 4
         ↓ Mantiene cronología
```

---

## 🎯 PRÓXIMOS PASOS

### Nivel Principiante

1. **Ejecutar el proyecto**:

   ```bash
   python train.py
   ```

2. **Analizar las gráficas**:

   - ¿El modelo aprende? (entrenamiento.png)
   - ¿Hay overfitting?
   - ¿Buenas predicciones? (predicciones.png)

3. **Experimentos simples**:
   - Cambiar épocas: `epochs=30` o `epochs=100`
   - Cambiar sequence_length: 30, 90, 120

---

### Nivel Intermedio

1. **Mejorar el modelo**:

   ```python
   # Más neuronas
   LSTM(100) en vez de LSTM(50)

   # Más capas
   Agregar 3ra capa LSTM

   # Menos dropout
   Dropout(0.1) en vez de 0.2
   ```

2. **Agregar features**:

   - Día de la semana
   - Mes del año
   - Tendencia

3. **Cross-validation temporal**:
   - Entrenar con diferentes ventanas
   - Promediar resultados

---

### Nivel Avanzado

1. **Arquitecturas avanzadas**:

   ```python
   # Bidirectional LSTM
   Bidirectional(LSTM(50))

   # Attention Mechanism
   attention = Attention()([lstm1, lstm2])

   # Encoder-Decoder
   Seq2Seq architecture
   ```

2. **Hyperparameter Tuning**:

   ```python
   from keras_tuner import RandomSearch

   tuner = RandomSearch(
       build_model,
       objective='val_loss',
       max_trials=50
   )
   ```

3. **Ensemble Methods**:

   - Entrenar 5 modelos
   - Promediar predicciones
   - Mejor robustez

4. **Crear API REST**:

   ```python
   from flask import Flask

   @app.route('/predict', methods=['POST'])
   def predict():
       data = request.json
       prediction = model.predict(data)
       return jsonify({'prediction': prediction})
   ```

---

## 🎓 RESUMEN DE CONCEPTOS

### Machine Learning

- ✅ Aprendizaje supervisado
- ✅ Series temporales
- ✅ Preprocesamiento
- ✅ Train/Val/Test split
- ✅ Overfitting vs Underfitting

### Deep Learning

- ✅ Redes neuronales
- ✅ Capas (LSTM, Dense, Dropout)
- ✅ Funciones de activación
- ✅ Backpropagation (implícito)
- ✅ Optimizadores (Adam)

### LSTM

- ✅ Memoria a largo plazo
- ✅ Gates (forget, input, output)
- ✅ Ventanas deslizantes
- ✅ return_sequences
- ✅ Stateful vs Stateless

### Evaluación

- ✅ RMSE, MAE, MAPE, R²
- ✅ Gráficas de diagnóstico
- ✅ Análisis de errores
- ✅ Interpretación de resultados

### Ingeniería

- ✅ Modularización de código
- ✅ Configuración centralizada
- ✅ Documentación
- ✅ Manejo de errores
- ✅ Reproducibilidad

---

## 🏆 ¡FELICIDADES!

Has completado un proyecto **completo** de Machine Learning:

### Lo que sabes hacer ahora:

✅ Cargar y explorar datos  
✅ Preprocesar series temporales  
✅ Crear ventanas deslizantes  
✅ Construir redes LSTM  
✅ Entrenar modelos de Deep Learning  
✅ Evaluar con múltiples métricas  
✅ Visualizar resultados  
✅ Guardar y cargar modelos  
✅ Interpretar rendimiento  
✅ Estructurar proyecto profesionalmente

---

## 📚 RECURSOS ADICIONALES

### Documentación Oficial

- [TensorFlow/Keras](https://www.tensorflow.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)

### Cursos Recomendados

- Coursera: Deep Learning Specialization (Andrew Ng)
- Fast.ai: Practical Deep Learning
- Stanford CS230: Deep Learning

### Papers Importantes

- [LSTM (Hochreiter & Schmidhuber, 1997)](http://www.bioinf.jku.at/publications/older/2604.pdf)
- [Understanding LSTM Networks (Colah's Blog)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🚀 SIGUIENTE PROYECTO

**Ideas para expandir tus habilidades**:

1. **Predicción de acciones** (finanzas)
2. **Detección de anomalías** (fraude)
3. **Generación de texto** (NLP)
4. **Clasificación de imágenes** (visión)
5. **Sistemas de recomendación** (e-commerce)

---

**¡Éxito en tu viaje de Machine Learning!** 🎉🧠📊
