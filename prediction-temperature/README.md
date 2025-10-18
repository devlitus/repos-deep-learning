# 🌡️ Predicción de Temperatura con LSTM

Sistema completo de Machine Learning para predecir temperaturas mínimas diarias usando redes neuronales LSTM.

---

## 📋 Tabla de Contenidos

1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [¿Qué es Machine Learning?](#-qué-es-machine-learning)
3. [¿Qué son las Redes LSTM?](#-qué-son-las-redes-lstm)
4. [Estructura del Proyecto](#-estructura-del-proyecto)
5. [Instalación](#-instalación)
6. [Uso](#-uso)
7. [Resultados](#-resultados)
8. [Cómo Funciona](#-cómo-funciona)
9. [Métricas de Evaluación](#-métricas-de-evaluación)
10. [Mejoras Futuras](#-mejoras-futuras)

---

## 🎯 Descripción del Proyecto

Este proyecto predice **temperaturas mínimas diarias** en Melbourne, Australia, usando:

- **Datos**: 10 años de temperaturas (1981-1990)
- **Modelo**: Red neuronal LSTM (Long Short-Term Memory)
- **Objetivo**: Predecir la temperatura de mañana basándose en los últimos 60 días

### ¿Por qué es útil?

- 🌾 **Agricultura**: Planificar cultivos y protección ante heladas
- 🏭 **Energía**: Predecir demanda de calefacción/refrigeración
- 🏃 **Eventos**: Planificar actividades al aire libre
- 📚 **Educación**: Aprender Machine Learning con un caso real

---

## 🤖 ¿Qué es Machine Learning?

**Machine Learning** (Aprendizaje Automático) es enseñar a las computadoras a aprender de datos sin programarlas explícitamente.

### Analogía Simple

**Programación Tradicional**:

```
SI temperatura_ayer > 20°C ENTONCES
    temperatura_hoy = temperatura_ayer + ruido
```

**Machine Learning**:

```
Le damos 10 años de temperaturas →
La computadora descubre los patrones →
Puede predecir temperaturas futuras
```

### Tipos de Aprendizaje

1. **Supervisado** (Este proyecto): Aprender de ejemplos con respuestas
2. **No supervisado**: Encontrar patrones sin respuestas
3. **Refuerzo**: Aprender mediante prueba y error

---

## 🧠 ¿Qué son las Redes LSTM?

**LSTM** = Long Short-Term Memory (Memoria de Largo y Corto Plazo)

### ¿Por qué LSTM para temperaturas?

Las temperaturas tienen **memoria**:

- La temperatura de hoy depende de ayer
- Y de la semana pasada
- Y de la estación del año

**LSTM recuerda patrones temporales**:

```
Día 1: 15°C ─┐
Día 2: 16°C  ├─→ LSTM aprende: "está subiendo"
Día 3: 17°C ─┘
Día 4: ¿?°C → Predicción: ~18°C
```

### Ventajas sobre otros modelos

| Modelo           | Memoria        | Predicción           |
| ---------------- | -------------- | -------------------- |
| Regresión Lineal | ❌ No          | Solo promedio        |
| RNN Simple       | ⚠️ Corto plazo | Se olvida del pasado |
| **LSTM**         | ✅ Largo plazo | Recuerda patrones    |

---

## 📁 Estructura del Proyecto

```
prediccion-temperatura/
│
├── data/
│   ├── __init__.py
│   ├── load_data.py                    # Carga y normaliza datos
│   └── daily-min-temperatures.csv      # Dataset (3650 días)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                # Crea secuencias y divide datos
│   ├── model.py                        # Define arquitectura LSTM
│   ├── evaluation.py                   # Calcula métricas (RMSE, MAE, R²)
│   └── visualization.py                # Crea gráficas
│
├── models/
│   └── lstm_temperatura.keras          # Modelo entrenado (guardado)
│
├── reports/
│   ├── temperatura_historica.png       # Datos originales
│   ├── entrenamiento.png               # Progreso del entrenamiento
│   ├── predicciones.png                # Predicciones vs Realidad
│   ├── scatter.png                     # Gráfica de dispersión
│   ├── errores.png                     # Análisis de errores
│   └── metricas.txt                    # Reporte de métricas
│
├── web/
│   └── __init__.py                     # (Futuro: interfaz web)
│
├── config.py                           # Configuración global
├── train.py                            # Script principal 🚀
├── requirements.txt                    # Dependencias
└── README.md                           # Este archivo
```

---

## 🔧 Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Paso 1: Clonar/Descargar el Proyecto

```bash
# Si tienes el proyecto en GitHub
git clone https://github.com/tu-usuario/prediccion-temperatura.git
cd prediccion-temperatura

# O simplemente descarga la carpeta completa
```

### Paso 2: Crear Entorno Virtual (Recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:

- `numpy`: Operaciones matemáticas
- `pandas`: Manejo de datos
- `tensorflow`: Framework de Deep Learning
- `scikit-learn`: Preprocesamiento y métricas
- `matplotlib`: Visualizaciones

### Paso 4: Verificar Instalación

```bash
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

---

## 🚀 Uso

### Entrenamiento Completo

Ejecutar el pipeline completo (5-10 minutos):

```bash
python train.py
```

Esto ejecutará:

1. ✅ Carga de datos
2. ✅ Preprocesamiento
3. ✅ Entrenamiento del modelo (50 épocas)
4. ✅ Evaluación
5. ✅ Generación de gráficas
6. ✅ Guardado del modelo

### Uso Programático

```python
# Entrenar modelo
from train import train_model
model, history, metrics = train_model()

# Hacer predicciones con modelo guardado
from tensorflow import keras
import numpy as np

# Cargar modelo
model = keras.models.load_model('models/lstm_temperatura.keras')

# Predecir (necesitas 60 días normalizados)
secuencia = np.array([...])  # 60 temperaturas normalizadas
prediccion = model.predict(secuencia)
```

### Modificar Hiperparámetros

Edita `config.py` o `train.py`:

```python
# Cambiar cantidad de épocas
epochs = 100  # Default: 50

# Cambiar tamaño de secuencia
sequence_length = 90  # Default: 60

# Cambiar arquitectura
lstm_units_1 = 100  # Default: 50
```

---

## 📊 Resultados

### Métricas Esperadas

Con los parámetros por defecto:

| Métrica  | Valor Esperado | Interpretación          |
| -------- | -------------- | ----------------------- |
| **RMSE** | 1.5 - 2.5°C    | Error promedio en °C    |
| **MAE**  | 1.2 - 2.0°C    | Error absoluto promedio |
| **MAPE** | 5 - 10%        | Error porcentual        |
| **R²**   | 0.75 - 0.90    | % de varianza explicada |

### Ejemplo Real

```
RMSE: 1.87°C
MAE:  1.45°C
MAPE: 7.23%
R²:   0.8542 (85.42% varianza explicada)

Interpretación:
✅ Excelente: El modelo predice con ~2°C de error
✅ Error porcentual aceptable (< 10%)
✅ Explica el 85% de las variaciones
```

### Gráficas Generadas

1. **Temperatura Histórica**: Visualiza 10 años de datos
2. **Progreso de Entrenamiento**: Pérdida vs épocas
3. **Predicciones vs Realidad**: Comparación visual
4. **Dispersión**: Cada punto = (real, predicción)
5. **Análisis de Errores**: Distribución de errores

---

## 🔍 Cómo Funciona

### Pipeline Completo

```
DATOS
↓
1. Carga → CSV con 3650 temperaturas
↓
2. Normalización → Escalar a rango [0, 1]
↓
3. Secuencias → Crear ventanas de 60 días
   [día1...día60] → día61
   [día2...día61] → día62
   ...
↓
4. División → Train (70%) / Val (15%) / Test (15%)
↓
5. Modelo LSTM → Arquitectura:
   - Input: 60 días
   - LSTM 1: 50 neuronas
   - Dropout: 20%
   - LSTM 2: 50 neuronas
   - Dropout: 20%
   - Dense: 1 neurona (predicción)
↓
6. Entrenamiento → 50 épocas, batch_size=32
↓
7. Evaluación → Calcular RMSE, MAE, MAPE, R²
↓
8. Visualización → Generar 5 gráficas
↓
9. Guardado → Modelo .keras + reportes
```

### Ejemplo de Predicción

```python
# Datos de entrada (últimos 60 días)
Input: [15.2, 15.8, 16.1, ..., 18.3, 18.9, 19.2]
        ↓
     [LSTM Layer 1]
        ↓
     [LSTM Layer 2]
        ↓
      [Dense]
        ↓
Output: 19.7°C  (predicción para día 61)
```

---

## 📐 Métricas de Evaluación

### 1. RMSE (Root Mean Squared Error)

**Fórmula**: √(promedio((real - predicción)²))

**Interpretación**:

- RMSE = 2.0°C → "Me equivoco 2°C en promedio"
- Penaliza más los errores grandes

**Escala de calidad**:

- ✅ Excelente: < 1°C
- 👍 Bueno: 1-2°C
- ⚠️ Aceptable: 2-3°C
- ❌ Malo: > 3°C

### 2. MAE (Mean Absolute Error)

**Fórmula**: promedio(|real - predicción|)

**Interpretación**:

- MAE = 1.5°C → "Error absoluto promedio de 1.5°C"
- Todos los errores pesan igual

### 3. MAPE (Mean Absolute Percentage Error)

**Fórmula**: promedio(|real - predicción| / real) × 100

**Interpretación**:

- MAPE = 7% → "Me equivoco un 7% en promedio"
- Fácil de entender

**Escala de calidad**:

- ✅ Excelente: < 5%
- 👍 Bueno: 5-10%
- ⚠️ Aceptable: 10-20%
- ❌ Malo: > 20%

### 4. R² (R-squared)

**Fórmula**: 1 - (errores_modelo / errores_modelo_promedio)

**Interpretación**:

- R² = 0.85 → "Explica el 85% de la variabilidad"
- Compara tu modelo vs predecir siempre el promedio

**Escala de calidad**:

- ✅ Excelente: > 0.90
- 👍 Bueno: 0.70-0.90
- ⚠️ Aceptable: 0.50-0.70
- ❌ Malo: < 0.50

---

## 🚀 Mejoras Futuras

### Técnicas Avanzadas

1. **Arquitectura**:

   - ✅ Bidirectional LSTM
   - ✅ Attention Mechanism
   - ✅ GRU (alternativa a LSTM)

2. **Datos**:

   - ✅ Agregar más features (humedad, presión, viento)
   - ✅ Usar datos de múltiples ciudades
   - ✅ Incorporar datos meteorológicos

3. **Entrenamiento**:
   - ✅ Learning rate scheduling
   - ✅ Early stopping más sofisticado
   - ✅ Cross-validation temporal

### Funcionalidades

1. **Interfaz Web** (Streamlit/Flask):

   ```
   - Subir datos personalizados
   - Visualizar predicciones en tiempo real
   - Comparar múltiples modelos
   ```

2. **API REST**:

   ```python
   POST /predict
   Body: {"last_60_days": [15, 16, ...]}
   Response: {"prediction": 19.7, "confidence": 0.85}
   ```

3. **Dashboard Interactivo**:
   - Gráficas interactivas con Plotly
   - Selección de rango de fechas
   - Exportar reportes en PDF

---

## 📚 Recursos de Aprendizaje

### Machine Learning Básico

- [Coursera: Machine Learning (Andrew Ng)](https://www.coursera.org/learn/machine-learning)
- [StatQuest: Machine Learning](https://www.youtube.com/c/joshstarmer)

### Deep Learning y LSTM

- [Deep Learning Book (Goodfellow)](https://www.deeplearningbook.org/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

### TensorFlow/Keras

- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [Keras API Reference](https://keras.io/api/)

### Series Temporales

- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas!

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/mejora`)
3. Commit tus cambios (`git commit -m 'Añadir mejora'`)
4. Push a la rama (`git push origin feature/mejora`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## 👤 Autor

Proyecto educativo creado para enseñar Machine Learning desde cero.

---

## ❓ Preguntas Frecuentes

### ¿Por qué 60 días como secuencia?

60 días (~2 meses) captura:

- Patrones semanales
- Tendencias mensuales
- Sin ser demasiado largo (overfitting)

Puedes probar con 30, 90 o 120 días.

### ¿Por qué LSTM y no otro modelo?

| Modelo         | Pros                            | Contras                       |
| -------------- | ------------------------------- | ----------------------------- |
| Promedio Móvil | Simple                          | No captura patrones complejos |
| ARIMA          | Bueno para series estacionarias | Asume linealidad              |
| **LSTM**       | Captura patrones no lineales    | Más complejo                  |
| Transformer    | Estado del arte                 | Necesita muchos datos         |

LSTM es el equilibrio perfecto para este problema.

### ¿Cómo mejoro el modelo?

1. **Más épocas**: epochs=100
2. **Más datos**: Agregar más años
3. **Más features**: Humedad, presión, etc.
4. **Mejor arquitectura**: Más capas LSTM
5. **Hyperparameter tuning**: Grid search

### ¿Puedo usar otros datos?

¡Sí! Solo necesitas:

1. Un CSV con fechas y valores numéricos
2. Al menos 1000 observaciones
3. Modificar `load_data.py` para tu formato

---

## 🎉 ¡Felicidades!

Si llegaste hasta aquí, ya sabes cómo funciona un sistema completo de Machine Learning.

**Has aprendido**:

- ✅ Carga y preprocesamiento de datos
- ✅ Redes neuronales LSTM
- ✅ Entrenamiento y evaluación
- ✅ Visualización de resultados
- ✅ Despliegue de modelos

**Próximos pasos**:

1. Ejecuta `python train.py`
2. Analiza las gráficas
3. Experimenta con parámetros
4. ¡Crea tu propio proyecto!

---

**¿Preguntas o problemas?**

- Abre un Issue en GitHub
- Consulta la documentación de TensorFlow
- Busca en Stack Overflow

¡Buena suerte con tu aprendizaje! 🚀📊🧠
