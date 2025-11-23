"""
Script de diagnóstico para verificar predicciones del modelo LSTM
"""

import numpy as np
from pathlib import Path
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Rutas
BASE_DIR = Path(__file__).parent
MODEL_LSTM = BASE_DIR / 'models' / 'lstm_sentiment_model.keras'
TOKENIZER_FILE = BASE_DIR / 'models' / 'tokenizer.pkl'

# Cargar modelo y tokenizer
print("Cargando modelo...")
model = load_model(str(MODEL_LSTM))
print("✅ Modelo cargado")

print("\nCargando tokenizer...")
with open(TOKENIZER_FILE, 'rb') as f:
    tokenizer = pickle.load(f)
print("✅ Tokenizer cargado")

# Textos de prueba
test_texts = [
    "This movie is excellent! Best film ever!",  # Debería ser POSITIVO
    "Terrible waste of time. The plot was boring and confusing.",  # Debería ser NEGATIVO
    "I absolutely loved this movie!",  # Debería ser POSITIVO
    "Worst movie I've ever seen. Awful.",  # Debería ser NEGATIVO
]

expected_labels = ["POSITIVO", "NEGATIVO", "POSITIVO", "NEGATIVO"]

print("\n" + "="*70)
print("PROBANDO PREDICCIONES")
print("="*70)

for i, (text, expected) in enumerate(zip(test_texts, expected_labels), 1):
    # Tokenizar y pad
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=300, padding='post', truncating='post')

    # Predecir
    probability = model.predict(padded, verbose=0)[0][0]

    # Interpretar
    if probability > 0.5:
        prediction = "POSITIVO"
        confidence = probability
    else:
        prediction = "NEGATIVO"
        confidence = 1 - probability

    # Verificar
    is_correct = "✅" if prediction == expected else "❌"

    print(f"\n{i}. {is_correct} {prediction} (confianza: {confidence:.2%})")
    print(f"   Texto: \"{text}\"")
    print(f"   Esperado: {expected}")
    print(f"   Probabilidad raw: {probability:.4f}")

    # Debug: mostrar secuencia
    print(f"   Secuencia (primeras 10 palabras): {sequence[0][:10]}")

print("\n" + "="*70)
