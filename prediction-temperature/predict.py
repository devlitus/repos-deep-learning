"""
SCRIPT DE PREDICCIÓN: Usar el modelo entrenado para hacer predicciones

Este script carga el modelo LSTM entrenado y realiza predicciones de temperatura.

Uso:
    python predict.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys
import io

# Agregar encoding UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("\n" + "="*70)
print("SISTEMA DE PREDICCION DE TEMPERATURA")
print("="*70 + "\n")

# Importar módulos del proyecto
print("[INFO] Importando módulos...")

try:
    from tensorflow.keras.models import load_model
    from data.load_data import load_melbourne_data
    from src.preprocessing import create_sequences, split_data
    from src.evaluation import evaluate_model
    print("[OK] Modulos importados correctamente\n")
except Exception as e:
    print(f"[ERROR] No se pudieron importar módulos: {e}")
    sys.exit(1)


def load_trained_model(model_path='models/lstm_temperatura.keras'):
    """Cargar el modelo entrenado"""
    try:
        model = load_model(model_path)
        print(f"[OK] Modelo cargado desde: {model_path}")
        return model
    except FileNotFoundError:
        print(f"[ERROR] Modelo no encontrado en: {model_path}")
        print("   Primero ejecuta: python train.py")
        return None


def predict_test_set():
    """Hacer predicciones en el conjunto de prueba"""

    print("\n" + "="*70)
    print("PREDICCIONES EN CONJUNTO DE PRUEBA")
    print("="*70 + "\n")

    # Cargar datos
    print("[INFO] Cargando datos...")
    df, data_normalized, scaler = load_melbourne_data()

    # Crear secuencias
    print("[INFO] Creando secuencias...")
    X, y = create_sequences(data_normalized, time_steps=60)

    # Dividir datos
    print("[INFO] Dividiendo datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"[OK] Conjunto de prueba: {len(X_test)} muestras\n")

    # Cargar modelo
    model = load_trained_model()
    if model is None:
        return

    # Hacer predicciones
    print("[INFO] Realizando predicciones en conjunto de prueba...")
    predictions = model.predict(X_test, verbose=0)

    # Desnormalizar resultados
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler.inverse_transform(predictions).flatten()

    # Calcular métricas
    print("\n" + "="*70)
    print("RESULTADOS DE PREDICCIONES")
    print("="*70 + "\n")

    # MAE
    mae = np.mean(np.abs(y_test_original - predictions_original))
    print(f"[METRICA] MAE (Error Absoluto Medio):  {mae:.4f} C")

    # RMSE
    rmse = np.sqrt(np.mean((y_test_original - predictions_original) ** 2))
    print(f"[METRICA] RMSE (Error Cuadrático):     {rmse:.4f} C")

    # MAPE
    mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
    print(f"[METRICA] MAPE (Error Porcentual):     {mape:.2f}%")

    # R²
    ss_res = np.sum((y_test_original - predictions_original) ** 2)
    ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"[METRICA] R² (Coeficiente Determinacion): {r2:.4f}\n")

    # Mostrar algunas predicciones
    print("="*70)
    print("MUESTRA DE PREDICCIONES (Primeras 10)")
    print("="*70 + "\n")

    print(f"{'Indice':<8} {'Real (C)':<12} {'Predicho (C)':<12} {'Error (C)':<12}")
    print("-" * 50)

    for i in range(min(10, len(y_test_original))):
        real = y_test_original[i]
        pred = predictions_original[i]
        error = pred - real
        print(f"{i:<8} {real:<12.2f} {pred:<12.2f} {error:<12.2f}")

    print("\n" + "="*70)
    print("ANALISIS DE ERRORES")
    print("="*70 + "\n")

    errors = predictions_original - y_test_original

    print(f"Error mínimo:     {np.min(errors):.4f} C")
    print(f"Error máximo:     {np.max(errors):.4f} C")
    print(f"Error promedio:   {np.mean(errors):.4f} C")
    print(f"Desviación std:   {np.std(errors):.4f} C")

    # Predicciones dentro de ±1 C
    within_1c = np.sum(np.abs(errors) <= 1.0) / len(errors) * 100
    print(f"\nPredicciones dentro de ±1 C:   {within_1c:.1f}%")

    # Predicciones dentro de ±0.5 C
    within_05c = np.sum(np.abs(errors) <= 0.5) / len(errors) * 100
    print(f"Predicciones dentro de ±0.5 C: {within_05c:.1f}%")

    print("\n" + "="*70)
    print("FIN DE PREDICCIONES")
    print("="*70 + "\n")

    return model, scaler


def predict_next_temperatures(model, scaler, X_test, y_test):
    """Predecir usando el conjunto de prueba"""

    print("\n" + "="*70)
    print("PREDICCION EN CONJUNTO DE PRUEBA (Muestra extendida)")
    print("="*70 + "\n")

    print(f"[INFO] Haciendo predicciones en {len(X_test)} muestras de prueba\n")

    predictions = model.predict(X_test, verbose=0)

    # Desnormalizar
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_original = scaler.inverse_transform(predictions).flatten()

    print(f"{'Muestra':<10} {'Real (C)':<12} {'Predicho (C)':<12} {'Error (C)':<12}")
    print("-" * 50)

    # Mostrar primeras 10
    for i in range(min(10, len(y_test_original))):
        real = y_test_original[i]
        pred = predictions_original[i]
        error = pred - real
        print(f"{i:<10} {real:<12.2f} {pred:<12.2f} {error:<12.2f}")

    # Mostrar últimas 10
    print("\n... (muestras en el medio omitidas) ...\n")
    for i in range(max(0, len(y_test_original)-10), len(y_test_original)):
        real = y_test_original[i]
        pred = predictions_original[i]
        error = pred - real
        print(f"{i:<10} {real:<12.2f} {pred:<12.2f} {error:<12.2f}")

    print("\n[OK] Predicciones completadas!\n")

    return predictions_original


if __name__ == "__main__":
    try:
        # Hacer predicciones en conjunto de prueba
        model, scaler = predict_test_set()

        if model is not None:
            # Cargar datos para más predicciones
            df, data_normalized, _ = load_melbourne_data()
            X, y = create_sequences(data_normalized, time_steps=60)
            X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

            # Hacer predicciones extendidas en conjunto de prueba
            predictions = predict_next_temperatures(model, scaler, X_test, y_test)

            print("[SUCCESS] Proceso completado sin errores!\n")

    except KeyboardInterrupt:
        print("\n\n[AVISO] Proceso interrumpido por el usuario\n")

    except Exception as e:
        print(f"\n\n[ERROR] Problema durante predicciones:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
