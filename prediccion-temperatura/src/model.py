"""
MÓDULO: MODELO LSTM MEJORADO (3 CAPAS)
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
    Construye modelo LSTM con 3 capas
    
    Arquitectura:
    1. LSTM (50 neuronas) → Patrones básicos
    2. LSTM (50 neuronas) → Patrones intermedios
    3. LSTM (50 neuronas) → Patrones complejos
    4. Dense (1 neurona)  → Predicción final
    """
    
    model = Sequential()
    
    # ════════════════════════════════════════════
    # CAPA 1: Primera LSTM
    # ════════════════════════════════════════════
    model.add(LSTM(
        units=50,                      # 50 neuronas
        return_sequences=True,         # TRUE ← hay otra LSTM después
        input_shape=(time_steps, 1)    # (60 días, 1 feature)
    ))
    model.add(Dropout(0.2))            # 20% dropout
    
    # ════════════════════════════════════════════
    # CAPA 2: Segunda LSTM (NUEVA)
    # ════════════════════════════════════════════
    model.add(LSTM(
        units=50,                      # 50 neuronas
        return_sequences=True          # TRUE ← hay OTRA LSTM después
    ))
    model.add(Dropout(0.2))
    
    # ════════════════════════════════════════════
    # CAPA 3: Tercera LSTM (ÚLTIMA)
    # ════════════════════════════════════════════
    model.add(LSTM(
        units=50                       # 50 neuronas
        # return_sequences=False (defecto)
        # FALSE ← es la última LSTM
    ))
    model.add(Dropout(0.2))
    
    # ════════════════════════════════════════════
    # CAPA FINAL: Dense (Predicción)
    # ════════════════════════════════════════════
    model.add(Dense(1))
    
    # Compilar modelo
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model


def get_model_summary(model):
    """Muestra resumen del modelo"""
    print("="*70)
    print("📐 ARQUITECTURA DEL MODELO")
    print("="*70 + "\n")
    
    model.summary()
    
    print("\n" + "="*70)
    print(f"✅ Modelo construido con {model.count_params():,} parámetros")
    print("="*70)