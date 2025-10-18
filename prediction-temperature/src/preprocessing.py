"""
═══════════════════════════════════════════════════════════════
MÓDULO: PREPROCESAMIENTO DE DATOS
═══════════════════════════════════════════════════════════════

Propósito: Transformar datos crudos en formato listo para LSTM

Funciones:
1. normalize_data() → Escala temperaturas entre 0 y 1
2. create_sequences() → Crea ventanas de tiempo para entrenar
3. split_train_test() → Divide en entrenamiento y prueba
4. reshape_for_lstm() → Da formato que LSTM necesita
5. save_scaler() / load_scaler() → Guarda/carga el normalizador
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from src.config import TIME_STEPS, TRAIN_SPLIT, SCALER_PATH


def normalize_data(data):
    """
    Normaliza los datos entre 0 y 1
    
    ═══════════════════════════════════════════════════════════
    ¿POR QUÉ NORMALIZAR?
    ═══════════════════════════════════════════════════════════
    
    Las redes neuronales funcionan mejor con números pequeños.
    Si usamos temperaturas directas (0°C - 26°C), el modelo
    aprende más lento y menos preciso.
    
    ═══════════════════════════════════════════════════════════
    ¿CÓMO FUNCIONA?
    ═══════════════════════════════════════════════════════════
    
    Fórmula: valor_normalizado = (valor - mínimo) / (máximo - mínimo)
    
    Ejemplo:
    Temperatura: 13°C
    Rango: 0°C a 26°C
    Normalizado: (13 - 0) / (26 - 0) = 0.5
    
    Resultado: 13°C se convierte en 0.5
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ ES MinMaxScaler?
    ═══════════════════════════════════════════════════════════
    
    Es una herramienta de scikit-learn que:
    1. Calcula el mínimo y máximo de los datos
    2. Guarda estos valores (para revertir después)
    3. Aplica la fórmula de normalización
    
    Args:
        data: Array con temperaturas originales
              Ejemplo: [[20.7], [17.9], [18.8], ...]
    
    Returns:
        tuple: (datos_normalizados, scaler)
               - datos_normalizados: Array con valores 0-1
               - scaler: Objeto que guarda min/max para revertir
    """
    # Crear el normalizador
    # feature_range=(0, 1) significa "escala entre 0 y 1"
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # fit_transform hace 2 cosas:
    # 1. fit: Aprende el mín y máx de los datos
    # 2. transform: Aplica la normalización
    scaled_data = scaler.fit_transform(data)
    
    # Mostrar información
    print(f"✅ Datos normalizados")
    print(f"   Original   → min={data.min():.2f}°C, max={data.max():.2f}°C")
    print(f"   Normalizado → min={scaled_data.min():.2f}, max={scaled_data.max():.2f}")
    print(f"   Ejemplo: {data[0][0]:.2f}°C → {scaled_data[0][0]:.4f}")
    
    return scaled_data, scaler


def create_sequences(data, time_steps=TIME_STEPS):
    """
    Crea secuencias de tiempo para entrenar LSTM
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ SON SECUENCIAS?
    ═══════════════════════════════════════════════════════════
    
    LSTM necesita "ventanas" de datos consecutivos.
    No le damos temperaturas sueltas, sino GRUPOS de días.
    
    ═══════════════════════════════════════════════════════════
    EJEMPLO VISUAL
    ═══════════════════════════════════════════════════════════
    
    Datos originales (10 días):
    [20, 21, 19, 22, 23, 24, 22, 21, 20, 19]
    
    Con time_steps=3, creamos:
    
    Secuencia 1:  [20, 21, 19] → Predice: 22
    Secuencia 2:  [21, 19, 22] → Predice: 23
    Secuencia 3:  [19, 22, 23] → Predice: 24
    Secuencia 4:  [22, 23, 24] → Predice: 22
    ...
    
    ═══════════════════════════════════════════════════════════
    ¿POR QUÉ time_steps=60?
    ═══════════════════════════════════════════════════════════
    
    Usamos 60 días (2 meses) porque:
    - Da suficiente contexto temporal
    - Captura patrones semanales y mensuales
    - No es tan largo que dificulte el entrenamiento
    
    ═══════════════════════════════════════════════════════════
    PROCESO PASO A PASO
    ═══════════════════════════════════════════════════════════
    
    1. Recorremos los datos desde la posición 'time_steps'
    2. Para cada posición i:
       - X: toma los últimos time_steps días (i-60 hasta i)
       - y: toma el día actual (i)
    3. Guardamos cada par (X, y) para entrenar
    
    Args:
        data: Array normalizado de temperaturas
        time_steps: Cantidad de días a usar como entrada
    
    Returns:
        tuple: (X, y)
               - X: Secuencias de entrada (3D)
               - y: Valores a predecir (1D)
    """
    X, y = [], []
    
    # Recorrer desde time_steps hasta el final
    # ¿Por qué desde time_steps? Porque necesitamos time_steps días previos
    for i in range(time_steps, len(data)):
        # X: últimos time_steps días
        # data[i-time_steps:i, 0] toma filas desde (i-60) hasta (i-1)
        # El :i no incluye i, por eso es "hasta i-1"
        X.append(data[i-time_steps:i, 0])
        
        # y: el día actual (lo que queremos predecir)
        y.append(data[i, 0])
    
    # Convertir listas a arrays de NumPy (más eficiente)
    X, y = np.array(X), np.array(y)
    
    # Mostrar información
    print(f"✅ Secuencias creadas")
    print(f"   X shape: {X.shape} → ({X.shape[0]} muestras, {X.shape[1]} días)")
    print(f"   y shape: {y.shape} → ({y.shape[0]} valores a predecir)")
    print(f"   Lógica: Usar {time_steps} días → Predecir día siguiente")
    print(f"   Total de ejemplos: {len(X)}")
    
    return X, y


def split_data(X, y, train_split=0.6, val_split=0.2):
    """
    Alias para split_train_val_test - Divide datos en 3 conjuntos: TRAIN, VALIDATION y TEST
    """
    return split_train_val_test(X, y, train_split, val_split)


def split_train_val_test(X, y, train_split=0.6, val_split=0.2):
    """
    Divide datos en 3 conjuntos: TRAIN, VALIDATION y TEST
    
    ═══════════════════════════════════════════════════════════
    ¿POR QUÉ 3 CONJUNTOS?
    ═══════════════════════════════════════════════════════════
    
    Analogía del estudiante:
    - TRAIN       = Ejercicios del libro (aprendes aquí)
    - VALIDATION  = Exámenes de práctica (ajustas tu estudio)
    - TEST        = Examen final (evaluación honesta)
    
    ═══════════════════════════════════════════════════════════
    DIVISIÓN TEMPORAL (sin mezclar)
    ═══════════════════════════════════════════════════════════
    
    |------ TRAIN (60%) ------|-- VAL (20%) --|-- TEST (20%) --|
    Días 1-2190              Días 2191-2920   Días 2921-3650
    
    IMPORTANTE: División cronológica, NO aleatoria
    ¿Por qué? Queremos predecir el FUTURO, no el pasado.
    
    ═══════════════════════════════════════════════════════════
    PROPÓSITO DE CADA CONJUNTO
    ═══════════════════════════════════════════════════════════
    
    1. TRAIN (60%):
       - Aquí el modelo APRENDE patrones
       - Ajusta sus pesos internos
       
    2. VALIDATION (20%):
       - Evalúa el modelo DURANTE el entrenamiento
       - Detecta overfitting temprano
       - Ayuda a decidir cuándo parar (EarlyStopping)
       - El modelo NO entrena con estos datos
       
    3. TEST (20%):
       - Evaluación FINAL
       - Datos que el modelo NUNCA vio
       - Mide el rendimiento real
       - Solo se usa AL FINAL
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        X: Secuencias de entrada
        y: Valores a predecir
        train_split: Porcentaje para entrenamiento (default: 0.6 = 60%)
        val_split: Porcentaje para validación (default: 0.2 = 20%)
                   El resto (20%) será para test
    
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Calcular índices de división
    train_size = int(len(X) * train_split)
    val_size = int(len(X) * val_split)
    
    # División cronológica:
    # 1. TRAIN: desde inicio hasta train_size
    X_train = X[:train_size]
    y_train = y[:train_size]
    
    # 2. VALIDATION: desde train_size hasta train_size + val_size
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    
    # 3. TEST: desde train_size + val_size hasta el final
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    # Calcular porcentajes reales
    test_split = 1 - train_split - val_split
    
    # Mostrar información
    print(f"✅ Datos divididos en 3 conjuntos")
    print(f"   TRAIN      : {len(X_train)} muestras ({train_split*100:.0f}%)")
    print(f"   VALIDATION : {len(X_val)} muestras ({val_split*100:.0f}%)")
    print(f"   TEST       : {len(X_test)} muestras ({test_split*100:.0f}%)")
    print(f"   Total      : {len(X)} muestras")
    print(f"\n   📅 División temporal:")
    print(f"      Train: días 1-{train_size}")
    print(f"      Val  : días {train_size+1}-{train_size+val_size}")
    print(f"      Test : días {train_size+val_size+1}-{len(X)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def reshape_for_lstm(X_train, X_val, X_test):
    """
    Reformatea datos para que LSTM los pueda procesar
    
    ═══════════════════════════════════════════════════════════
    ¿POR QUÉ RESHAPE?
    ═══════════════════════════════════════════════════════════
    
    LSTM necesita datos en formato 3D:
    (número_de_muestras, pasos_de_tiempo, características)
    
    ═══════════════════════════════════════════════════════════
    EXPLICACIÓN DIMENSIONAL
    ═══════════════════════════════════════════════════════════
    
    Antes del reshape:
    X_train.shape = (2154, 60)
    - 2154 muestras
    - 60 días cada una
    
    Después del reshape:
    X_train.shape = (2154, 60, 1)
    - 2154 muestras
    - 60 pasos de tiempo (días)
    - 1 característica (solo temperatura)
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ ES UNA CARACTERÍSTICA?
    ═══════════════════════════════════════════════════════════
    
    Una característica = una variable que medimos
    
    En nuestro caso: solo temperatura (1 característica)
    
    Si tuviéramos más datos:
    [temperatura, humedad, viento] = 3 características
    
    Args:
        X_train: Datos de entrenamiento (2D)
        X_val: Datos de validación (2D)
        X_test: Datos de prueba (2D)
    
    Returns:
        tuple: (X_train, X_val, X_test) en formato 3D
    """
    # reshape(samples, time_steps, features)
    # -1 en el primer parámetro = "calcula automáticamente"
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"✅ Datos reformateados para LSTM")
    print(f"   X_train: {X_train.shape} → (muestras, tiempo, características)")
    print(f"   X_val  : {X_val.shape}")
    print(f"   X_test : {X_test.shape}")
    
    return X_train, X_val, X_test


def save_scaler(scaler):
    """
    Guarda el scaler en disco
    
    ¿Por qué guardar el scaler?
    Porque lo necesitaremos después para:
    1. Normalizar nuevos datos antes de predecir
    2. Desnormalizar las predicciones (convertir 0.5 → 13°C)
    
    Args:
        scaler: Objeto MinMaxScaler entrenado
    """
    with open(SCALER_PATH, 'wb') as f:
        # pickle = librería para guardar objetos de Python
        pickle.dump(scaler, f)
    print(f"✅ Scaler guardado en {SCALER_PATH}")


def load_scaler():
    """
    Carga el scaler desde disco
    
    Returns:
        MinMaxScaler: Objeto scaler con min/max guardados
    """
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ Scaler cargado desde {SCALER_PATH}")
    return scaler


# ═══════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 PROBANDO MÓDULO DE PREPROCESAMIENTO")
    print("="*70 + "\n")
    
    # Crear datos de ejemplo (10 temperaturas)
    test_data = np.array([[20], [21], [19], [22], [23], [24], [22], [21], [20], [19]])
    
    print("1️⃣ NORMALIZACIÓN:")
    print("-"*70)
    scaled, scaler = normalize_data(test_data)
    print(f"   Primer valor: {test_data[0][0]}°C → {scaled[0][0]:.4f}\n")
    
    print("2️⃣ CREAR SECUENCIAS:")
    print("-"*70)
    X, y = create_sequences(scaled, time_steps=3)
    print(f"   Secuencia 0: {X[0]} → predice {y[0]:.4f}\n")
    
    print("3️⃣ DIVIDIR DATOS (Train/Val/Test):")
    print("-"*70)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, train_split=0.6, val_split=0.2)
    
    print("\n4️⃣ RESHAPE PARA LSTM:")
    print("-"*70)
    X_train, X_val, X_test = reshape_for_lstm(X_train, X_val, X_test)
    
    print("\n" + "="*70)
    print("✅ MÓDULO PROBADO CORRECTAMENTE")
    print("="*70)