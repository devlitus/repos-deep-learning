"""
═══════════════════════════════════════════════════════════════
CONFIGURACIÓN DEL PROYECTO - PREDICCIÓN DE TEMPERATURA
═══════════════════════════════════════════════════════════════

Este archivo contiene TODOS los parámetros del proyecto.
Si quieres experimentar, cambia los valores aquí.
"""

# ═══════════════════════════════════════════════════════════════
# 1. DATOS
# ═══════════════════════════════════════════════════════════════

# URL del dataset (temperaturas de Melbourne 1981-1990)
DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'

# Rutas donde se guardan archivos
DATA_RAW = 'data/daily-min-temperatures.csv'          # CSV original
DATA_PROCESSED = 'data/processed_data.npz'            # Datos procesados
MODEL_PATH = 'models/lstm_temperature_model.h5'       # Modelo entrenado
SCALER_PATH = 'models/scaler.pkl'                     # Normalizador guardado


# ═══════════════════════════════════════════════════════════════
# 2. PARÁMETROS DE PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════

# TIME_STEPS: ¿Cuántos días pasados usamos para predecir el siguiente?
# Ejemplo: Si TIME_STEPS=60, usamos los últimos 60 días para predecir el día 61
TIME_STEPS = 60

# TRAIN_SPLIT: ¿Qué porcentaje usamos para entrenar?
# 0.8 = 80% entrenamiento, 20% prueba
TRAIN_SPLIT = 0.8


# ═══════════════════════════════════════════════════════════════
# 3. ARQUITECTURA DEL MODELO LSTM
# ═══════════════════════════════════════════════════════════════

# LSTM_UNITS: Cantidad de "neuronas" en cada capa LSTM
# Más neuronas = más capacidad de aprender (pero más lento y riesgo de sobreajuste)
LSTM_UNITS_1 = 50  # Primera capa
LSTM_UNITS_2 = 50  # Segunda capa

# DROPOUT_RATE: Porcentaje de neuronas que "apagamos" durante entrenamiento
# ¿Para qué? Evita que el modelo memorice (overfitting)
# Valor típico: 0.2 (20%)
DROPOUT_RATE = 0.2


# ═══════════════════════════════════════════════════════════════
# 4. PARÁMETROS DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════

# EPOCHS: ¿Cuántas veces el modelo ve TODOS los datos?
# Más épocas = más aprendizaje (pero puede sobreajustarse)
EPOCHS = 50

# BATCH_SIZE: ¿Cuántos ejemplos procesa a la vez?
# Valores típicos: 16, 32, 64
# Más grande = más rápido pero usa más memoria
BATCH_SIZE = 32


# ═══════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN
# ═══════════════════════════════════════════════════════════════

# Tamaño de las gráficas (ancho, alto) en pulgadas
FIGSIZE = (15, 5)


# ═══════════════════════════════════════════════════════════════
# NOTAS IMPORTANTES
# ═══════════════════════════════════════════════════════════════
"""
Si quieres experimentar:

- TIME_STEPS más grande (ej: 90) → Usa más historia, pero tarda más
- TIME_STEPS más pequeño (ej: 30) → Más rápido, pero menos contexto
- LSTM_UNITS mayor (ej: 100) → Más potente, pero más lento
- EPOCHS mayor (ej: 100) → Más entrenamiento, pero ojo con overfitting
- BATCH_SIZE mayor (ej: 64) → Más rápido, necesita más RAM
"""