"""
MÓDULO: MODELO LSTM MEJORADO (3 CAPAS)
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
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
    1. Bidirectional LSTM (64 neuronas) → Patrones básicos
    2. Bidirectional LSTM (64 neuronas) → Patrones intermedios
    3. Dense (16 neuronas) + Dense (1 neurona) → Predicción final
    """

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=(time_steps, 1)),
        Dropout(0.3),
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
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