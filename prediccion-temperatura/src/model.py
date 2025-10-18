"""
═══════════════════════════════════════════════════════════════
MÓDULO: MODELO LSTM
═══════════════════════════════════════════════════════════════

Propósito: Construir y entrenar la red neuronal LSTM

Funciones:
1. build_lstm_model() → Construye la arquitectura del modelo
2. train_model() → Entrena el modelo con los datos
3. get_model_summary() → Muestra resumen del modelo
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from src.config import (
    LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE,
    EPOCHS, BATCH_SIZE, MODEL_PATH, TIME_STEPS
)


def build_lstm_model(time_steps=TIME_STEPS):
    """
    Construye el modelo LSTM capa por capa
    
    ═══════════════════════════════════════════════════════════
    ARQUITECTURA DEL MODELO
    ═══════════════════════════════════════════════════════════
    
    Nuestra red tiene 5 capas:
    
    1. LSTM Layer 1 (50 neuronas)
       ├─ Procesa la secuencia de 60 días
       └─ Aprende patrones temporales complejos
       
    2. Dropout (20%)
       ├─ "Apaga" aleatoriamente 20% de neuronas
       └─ Evita que el modelo memorice (overfitting)
       
    3. LSTM Layer 2 (50 neuronas)
       ├─ Refina los patrones aprendidos
       └─ Captura relaciones más abstractas
       
    4. Dropout (20%)
       └─ Más regularización
       
    5. Dense (1 neurona)
       └─ Capa de salida: predice la temperatura
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ ES Sequential?
    ═══════════════════════════════════════════════════════════
    
    Sequential = modelo donde las capas se apilan una tras otra
    
    Analogía: Como una tubería donde el agua (datos) fluye:
    
    Entrada → [Capa 1] → [Capa 2] → [Capa 3] → Salida
    
    ═══════════════════════════════════════════════════════════
    ¿POR QUÉ 2 CAPAS LSTM?
    ═══════════════════════════════════════════════════════════
    
    Una sola capa LSTM:
    - Aprende patrones simples
    - Ejemplo: "Si sube 3 días → sigue subiendo"
    
    Dos capas LSTM (stacked):
    - Primera capa: Aprende patrones básicos
    - Segunda capa: Aprende patrones de patrones
    - Ejemplo: "Detecta ciclos semanales dentro de tendencias mensuales"
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        time_steps: Longitud de la secuencia de entrada (60 días)
    
    Returns:
        model: Modelo LSTM compilado y listo para entrenar
    """
    
    # Inicializar el modelo secuencial
    # Sequential = capas en secuencia (una tras otra)
    model = Sequential()
    
    # ═══════════════════════════════════════════════════════════
    # CAPA 1: Primera LSTM
    # ═══════════════════════════════════════════════════════════
    
    model.add(LSTM(
        units=LSTM_UNITS_1,           # 50 neuronas LSTM
        return_sequences=True,        # ← IMPORTANTE: explicación abajo
        input_shape=(time_steps, 1)   # (60 días, 1 característica)
    ))
    
    """
    ¿QUÉ ES return_sequences=True?
    
    Controla QUÉ devuelve la capa LSTM:
    
    return_sequences=False (default):
    - Solo devuelve el OUTPUT del último paso
    - Salida: (batch_size, units)
    - Ejemplo: (32, 50) → 32 muestras, 50 valores
    
    return_sequences=True:
    - Devuelve TODOS los outputs de cada paso
    - Salida: (batch_size, time_steps, units)
    - Ejemplo: (32, 60, 50) → 32 muestras, 60 pasos, 50 valores por paso
    
    ¿CUÁNDO usar True?
    - Cuando hay OTRA capa LSTM después
    - La siguiente capa LSTM necesita la secuencia completa
    
    Analogía:
    - False = "Dame solo el resumen final del libro"
    - True = "Dame el resumen de cada capítulo"
    
    En nuestro caso:
    Primera LSTM → return_sequences=True → pasa 60 salidas a segunda LSTM
    Segunda LSTM → return_sequences=False → da solo la predicción final
    """
    
    # ═══════════════════════════════════════════════════════════
    # CAPA 2: Dropout (Regularización)
    # ═══════════════════════════════════════════════════════════
    
    model.add(Dropout(DROPOUT_RATE))  # 0.2 = 20%
    
    """
    ¿QUÉ HACE DROPOUT?
    
    Durante el ENTRENAMIENTO:
    - Apaga aleatoriamente el 20% de las neuronas
    - Cada época apaga neuronas DIFERENTES
    
    Ejemplo visual:
    Época 1: ⚫⚫⚪⚫⚫⚪⚫⚫⚪⚫  (⚪ = apagada)
    Época 2: ⚫⚪⚫⚫⚪⚫⚫⚪⚫⚫  (diferentes neuronas)
    
    Durante la PREDICCIÓN:
    - TODAS las neuronas activas (0% dropout)
    
    ¿POR QUÉ ES ÚTIL?
    
    Sin Dropout (puede memorizar):
    - Neurona 1: "Si día 27 = 15°C → día 28 = 16°C"
    - Memoriza ejemplos específicos ❌
    - En datos nuevos: falla
    
    Con Dropout (aprende robusto):
    - Cada neurona debe ser útil por sí sola
    - No puede depender de otras neuronas específicas
    - Aprende patrones generales ✅
    - En datos nuevos: funciona bien
    
    Analogía:
    Estudiando para un examen:
    - Sin Dropout: Memorizas respuestas exactas
    - Con Dropout: Entiendes los conceptos
    """
    
    # ═══════════════════════════════════════════════════════════
    # CAPA 3: Segunda LSTM
    # ═══════════════════════════════════════════════════════════
    
    model.add(LSTM(
        units=LSTM_UNITS_2,        # 50 neuronas LSTM
        return_sequences=False     # ← Solo output final
    ))
    
    """
    ¿POR QUÉ return_sequences=False aquí?
    
    Es la ÚLTIMA capa LSTM:
    - Ya no hay más capas LSTM después
    - Solo necesitamos UNA predicción final
    - No necesitamos la secuencia completa
    
    Salida de esta capa: (batch_size, 50)
    Ejemplo: (32, 50) → 32 muestras, 50 valores cada una
    """
    
    # ═══════════════════════════════════════════════════════════
    # CAPA 4: Dropout (Segunda regularización)
    # ═══════════════════════════════════════════════════════════
    
    model.add(Dropout(DROPOUT_RATE))  # Otra vez 20%
    
    """
    ¿POR QUÉ otro Dropout?
    
    Más capas de Dropout = más regularización
    - Evita aún más el overfitting
    - Especialmente útil con redes profundas (2+ capas)
    """
    
    # ═══════════════════════════════════════════════════════════
    # CAPA 5: Dense (Capa de Salida)
    # ═══════════════════════════════════════════════════════════
    
    model.add(Dense(units=1))
    
    """
    ¿QUÉ ES Dense?
    
    Dense = Capa totalmente conectada (Fully Connected)
    - Cada neurona conectada a TODAS las del paso anterior
    
    ¿POR QUÉ units=1?
    
    Queremos predecir UN solo valor: la temperatura de mañana
    
    Si quisiéramos predecir los próximos 7 días:
    - units=7 (una salida por cada día)
    
    Entrada a Dense: 50 valores (de la LSTM anterior)
    Salida de Dense: 1 valor (temperatura predicha)
    
    Cálculo interno:
    output = suma(50_valores × 50_pesos) + bias
    """
    
    # ═══════════════════════════════════════════════════════════
    # COMPILAR EL MODELO
    # ═══════════════════════════════════════════════════════════
    
    model.compile(
        optimizer='adam',              # Algoritmo de optimización
        loss='mean_squared_error'      # Función de pérdida
    )
    
    """
    ¿QUÉ ES COMPILAR?
    
    Compilar = Configurar cómo el modelo va a aprender
    
    Necesitas especificar:
    1. Optimizer (optimizador)
    2. Loss (función de pérdida)
    
    ───────────────────────────────────────────────────────────
    1. OPTIMIZER: 'adam'
    ───────────────────────────────────────────────────────────
    
    ¿Qué hace?
    - Ajusta los pesos del modelo para reducir el error
    
    ¿Cómo funciona?
    Piensa en bajar una montaña con los ojos vendados:
    
    SGD (optimizador simple):
    - Das pasos del mismo tamaño siempre
    - Puedes pasarte el valle o tardar mucho
    
    Adam (optimizador inteligente):
    - Ajusta el tamaño del paso automáticamente
    - Pasos grandes cuando estás lejos del óptimo
    - Pasos pequeños cuando estás cerca
    - Recuerda la dirección de pasos anteriores
    
    ¿Por qué Adam?
    ✅ Funciona bien en la mayoría de casos
    ✅ Converge más rápido
    ✅ No necesitas ajustar mucho el learning rate
    
    ───────────────────────────────────────────────────────────
    2. LOSS: 'mean_squared_error' (MSE)
    ───────────────────────────────────────────────────────────
    
    ¿Qué es?
    - Una función que mide "qué tan mal" predecimos
    
    Fórmula:
    MSE = promedio((predicción - real)²)
    
    Ejemplo:
    Predicciones: [20, 21, 19]
    Reales:       [21, 22, 20]
    Errores:      [-1, -1, -1]
    Errores²:     [1,  1,  1]
    MSE = (1 + 1 + 1) / 3 = 1.0
    
    ¿Por qué elevar al cuadrado?
    - Penaliza más los errores grandes
    - Error de 2°C pesa 4 veces más que error de 1°C
    - Error de 10°C pesa 100 veces más que error de 1°C
    
    ¿Por qué MSE para temperaturas?
    ✅ Penaliza errores grandes (importante en clima)
    ✅ Función suave (buena para gradientes)
    ✅ Estándar para problemas de regresión
    
    Otras opciones (no las usamos):
    - MAE: Mean Absolute Error (penaliza igual todos los errores)
    - Huber: Híbrido entre MSE y MAE
    """
    
    # Mostrar información
    print("✅ Modelo LSTM creado")
    print(f"\n   📐 Arquitectura:")
    print(f"      1. LSTM Layer 1    : {LSTM_UNITS_1} unidades (return_sequences=True)")
    print(f"      2. Dropout         : {DROPOUT_RATE*100}%")
    print(f"      3. LSTM Layer 2    : {LSTM_UNITS_2} unidades (return_sequences=False)")
    print(f"      4. Dropout         : {DROPOUT_RATE*100}%")
    print(f"      5. Dense (Output)  : 1 unidad")
    print(f"\n   ⚙️  Configuración:")
    print(f"      Optimizer: Adam")
    print(f"      Loss: Mean Squared Error (MSE)")
    
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    """
    Entrena el modelo LSTM con los datos
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ PASA DURANTE EL ENTRENAMIENTO?
    ═══════════════════════════════════════════════════════════
    
    El entrenamiento es un proceso iterativo:
    
    CADA ÉPOCA (epoch):
    ├─ 1. Forward Pass (hacia adelante):
    │     - El modelo ve los datos
    │     - Hace predicciones
    │     - Ejemplo: predice [20.5, 21.3, 19.8...]
    │
    ├─ 2. Calcular Loss (pérdida):
    │     - Compara predicciones vs realidad
    │     - Calcula MSE
    │     - Ejemplo: MSE = 2.5°C
    │
    ├─ 3. Backward Pass (hacia atrás):
    │     - Calcula gradientes (derivadas)
    │     - Determina cómo ajustar cada peso
    │     - "Si bajo este peso 0.01, el error baja 0.1"
    │
    ├─ 4. Actualizar Pesos:
    │     - Adam optimizer ajusta todos los pesos
    │     - Las 3 puertas LSTM aprenden
    │     - Dense layer aprende
    │
    └─ 5. Validación:
          - Evalúa en datos de validación
          - Si no mejora → EarlyStopping considera parar
    
    REPITE esto EPOCHS veces (50 por default)
    
    ═══════════════════════════════════════════════════════════
    CALLBACKS: Funciones que se ejecutan durante entrenamiento
    ═══════════════════════════════════════════════════════════
    
    Son como "asistentes" que vigilan el entrenamiento
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        model: Modelo LSTM construido
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
    
    Returns:
        history: Objeto con historial del entrenamiento
                 (pérdidas por época, métricas, etc.)
    """
    
    # Crear carpeta models si no existe
    os.makedirs('models', exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # CALLBACK 1: EarlyStopping
    # ═══════════════════════════════════════════════════════════
    
    early_stop = EarlyStopping(
        monitor='val_loss',           # Vigilar la pérdida de validación
        patience=10,                  # Esperar 10 épocas sin mejora
        restore_best_weights=True,    # Restaurar mejores pesos
        verbose=1                     # Mostrar mensajes
    )
    
    """
    ¿QUÉ HACE EarlyStopping?
    
    Detiene el entrenamiento si deja de mejorar
    
    ───────────────────────────────────────────────────────────
    PARÁMETROS:
    ───────────────────────────────────────────────────────────
    
    monitor='val_loss':
    - Vigila la pérdida en datos de VALIDACIÓN
    - No la pérdida de entrenamiento (esa siempre baja)
    
    ¿Por qué validación?
    - Si val_loss baja → modelo mejora en datos nuevos ✅
    - Si val_loss sube pero train_loss baja → overfitting ❌
    
    patience=10:
    - "Dame 10 oportunidades para mejorar"
    - Si en 10 épocas no hay mejora → STOP
    
    Ejemplo:
    Época 20: val_loss = 2.5 (mejor hasta ahora)
    Época 21: val_loss = 2.6 (peor, contador = 1)
    Época 22: val_loss = 2.7 (peor, contador = 2)
    ...
    Época 30: val_loss = 2.8 (contador = 10) → ¡STOP!
    
    restore_best_weights=True:
    - Al parar, restaura los pesos de la mejor época
    - No te quedas con los pesos finales (que pueden ser peores)
    
    Ejemplo:
    Época 20: val_loss = 2.5 ← MEJOR
    Época 30: val_loss = 2.8 ← Final (peor)
    EarlyStopping restaura pesos de época 20 ✅
    
    ───────────────────────────────────────────────────────────
    ¿POR QUÉ ES IMPORTANTE?
    ───────────────────────────────────────────────────────────
    
    Sin EarlyStopping:
    - Entrenas 50 épocas completas
    - Puede que en época 25 ya era óptimo
    - Épocas 26-50 = desperdicio de tiempo
    - Peor: puede sobreajustarse (overfitting)
    
    Con EarlyStopping:
    - Para automáticamente cuando es óptimo
    - Ahorra tiempo ⏱️
    - Evita overfitting ✅
    - Mejores resultados 📈
    
    Analogía:
    Como estudiar para un examen:
    - Sin EarlyStopping: Estudias hasta las 3am aunque ya lo sepas
    - Con EarlyStopping: Paras cuando ya dominas el tema
    """
    
    # ═══════════════════════════════════════════════════════════
    # CALLBACK 2: ModelCheckpoint
    # ═══════════════════════════════════════════════════════════
    
    checkpoint = ModelCheckpoint(
        MODEL_PATH,                   # Ruta donde guardar
        monitor='val_loss',           # Vigilar val_loss
        save_best_only=True,          # Solo guardar si mejora
        verbose=1                     # Mostrar mensajes
    )
    
    """
    ¿QUÉ HACE ModelCheckpoint?
    
    Guarda el modelo en disco cuando mejora
    
    ───────────────────────────────────────────────────────────
    PARÁMETROS:
    ───────────────────────────────────────────────────────────
    
    save_best_only=True:
    - Solo guarda cuando val_loss mejora
    - No sobreescribe con versiones peores
    
    Ejemplo:
    Época 10: val_loss = 3.0 → GUARDAR ✅
    Época 11: val_loss = 3.2 → No guardar
    Época 12: val_loss = 2.8 → GUARDAR ✅ (mejor que 3.0)
    Época 13: val_loss = 2.9 → No guardar
    
    Resultado final: Modelo guardado con val_loss = 2.8
    
    ───────────────────────────────────────────────────────────
    ¿POR QUÉ ES ÚTIL?
    ───────────────────────────────────────────────────────────
    
    Seguridad:
    - Si el entrenamiento se interrumpe (corte de luz, etc.)
    - Ya tienes el mejor modelo guardado
    
    Comparación:
    - Puedes entrenar múltiples veces
    - Comparar cuál modelo fue mejor
    
    Reproducibilidad:
    - El modelo guardado puede usarse después
    - Sin necesidad de re-entrenar
    """
    
    # ═══════════════════════════════════════════════════════════
    # INICIAR ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════
    
    print("\n🚀 Iniciando entrenamiento...")
    print("="*70)
    print(f"   📊 Configuración:")
    print(f"      Épocas máximas    : {EPOCHS}")
    print(f"      Batch size        : {BATCH_SIZE}")
    print(f"      Datos training    : {len(X_train)} muestras")
    print(f"      Datos validation  : {len(X_val)} muestras")
    print(f"\n   🎯 Callbacks:")
    print(f"      EarlyStopping     : Paciencia de 10 épocas")
    print(f"      ModelCheckpoint   : Guarda mejor modelo")
    print("="*70 + "\n")
    
    """
    ═══════════════════════════════════════════════════════════
    EXPLICACIÓN DE PARÁMETROS DE model.fit()
    ═══════════════════════════════════════════════════════════
    """
    
    history = model.fit(
        X_train, y_train,                    # Datos de entrenamiento
        epochs=EPOCHS,                       # Máximo de épocas
        batch_size=BATCH_SIZE,              # Tamaño de lote
        validation_data=(X_val, y_val),     # Datos de validación
        callbacks=[early_stop, checkpoint],  # Callbacks
        verbose=1                            # Mostrar progreso
    )
    
    """
    ───────────────────────────────────────────────────────────
    PARÁMETRO: batch_size=32
    ───────────────────────────────────────────────────────────
    
    ¿Qué es un batch (lote)?
    
    No procesamos TODAS las muestras a la vez
    Las dividimos en grupos pequeños (batches)
    
    Ejemplo con 3200 muestras y batch_size=32:
    - Batch 1: Muestras 1-32   → Procesa → Actualiza pesos
    - Batch 2: Muestras 33-64  → Procesa → Actualiza pesos
    - Batch 3: Muestras 65-96  → Procesa → Actualiza pesos
    - ...
    - Batch 100: Muestras 3169-3200 → Procesa → Actualiza pesos
    
    1 ÉPOCA = Procesar todos los batches una vez
    
    ───────────────────────────────────────────────────────────
    ¿POR QUÉ BATCHES?
    ───────────────────────────────────────────────────────────
    
    Ventaja 1: MEMORIA
    - No cabe todo en RAM/GPU a la vez
    - Batches pequeños → menos memoria
    
    Ventaja 2: VELOCIDAD
    - GPU procesa batches en paralelo
    - Más eficiente que uno por uno
    
    Ventaja 3: MEJOR APRENDIZAJE
    - Actualizar pesos frecuentemente (cada batch)
    - Converge más rápido
    - Mejor generalización
    
    ───────────────────────────────────────────────────────────
    COMPARACIÓN DE TAMAÑOS DE BATCH:
    ───────────────────────────────────────────────────────────
    
    batch_size=1 (Stochastic Gradient Descent):
    ❌ Muy lento
    ❌ Actualizaciones muy ruidosas
    ✅ Puede escapar de mínimos locales
    
    batch_size=32 (RECOMENDADO):
    ✅ Buen balance velocidad/memoria
    ✅ Actualizaciones estables
    ✅ Funciona bien en la mayoría de casos
    
    batch_size=3200 (Batch completo):
    ❌ Requiere mucha memoria
    ✅ Actualizaciones muy precisas
    ❌ Puede quedar atrapado en mínimos locales
    ❌ Más lento
    
    ───────────────────────────────────────────────────────────
    PARÁMETRO: verbose=1
    ───────────────────────────────────────────────────────────
    
    Controla cuánta información mostrar:
    
    verbose=0: Silencioso (nada)
    verbose=1: Barra de progreso + métricas (RECOMENDADO)
    verbose=2: Una línea por época (sin barra)
    
    Ejemplo de salida con verbose=1:
    Epoch 1/50
    100/100 [==============================] - 5s - loss: 0.0234 - val_loss: 0.0198
    """
    
    print("\n✅ Entrenamiento completado")
    print(f"   Total de épocas ejecutadas: {len(history.history['loss'])}")
    print(f"   Mejor val_loss: {min(history.history['val_loss']):.6f}")
    
    return history


def get_model_summary(model):
    """
    Muestra un resumen detallado del modelo
    
    Incluye:
    - Arquitectura de capas
    - Parámetros (pesos) por capa
    - Total de parámetros entrenables
    
    Args:
        model: Modelo LSTM
    """
    print("\n" + "="*70)
    print("📋 RESUMEN DEL MODELO")
    print("="*70)
    model.summary()
    print("="*70)
    
    """
    ¿QUÉ MUESTRA model.summary()?
    
    Ejemplo de salida:
    
    Layer (type)                 Output Shape              Param #   
    =================================================================
    lstm (LSTM)                  (None, 60, 50)            10400     
    dropout (Dropout)            (None, 60, 50)            0         
    lstm_1 (LSTM)                (None, 50)                20200     
    dropout_1 (Dropout)          (None, 50)                0         
    dense (Dense)                (None, 1)                 51        
    =================================================================
    Total params: 30,651
    Trainable params: 30,651
    Non-trainable params: 0
    
    ───────────────────────────────────────────────────────────
    EXPLICACIÓN:
    ───────────────────────────────────────────────────────────
    
    Output Shape:
    - (None, 60, 50) = (batch_size, time_steps, units)
    - None = puede variar según el batch
    
    Param # (Parámetros):
    - Cantidad de pesos que aprende esa capa
    
    ¿Por qué LSTM tiene 10,400 parámetros?
    
    LSTM tiene 4 sets de pesos (3 puertas + candidato):
    - Forget Gate: pesos
    - Input Gate: pesos  
    - Output Gate: pesos
    - Candidate: pesos
    
    Fórmula para LSTM:
    params = 4 × (units × (units + input_dim + 1))
    params = 4 × (50 × (50 + 1 + 1))
    params = 4 × (50 × 52)
    params = 10,400 ✅
    
    ¿Por qué Dense tiene 51 parámetros?
    params = (input_size × output_size) + bias
    params = (50 × 1) + 1
    params = 51 ✅
    
    ¿Por qué Dropout tiene 0 parámetros?
    - Dropout NO tiene pesos
    - Solo apaga neuronas aleatoriamente
    - No aprende nada
    """


# ═══════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 PROBANDO MÓDULO DEL MODELO")
    print("="*70 + "\n")
    
    # Construir modelo
    model = build_lstm_model()
    
    # Mostrar resumen
    get_model_summary(model)
    
    print("\n💡 Nota: Para entrenar el modelo, usa el script train.py")
    print("="*70)