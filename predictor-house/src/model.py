# src/model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from config import TEST_SIZE, RANDOM_STATE, MODEL_FILE

def split_data(X, y):
    """Divide datos en entrenamiento y prueba"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\n=== DIVISIÓN DE DATOS ===")
    print(f"Entrenamiento: {X_train.shape[0]} casas")
    print(f"Prueba: {X_test.shape[0]} casas")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Entrena el modelo de Regresión Lineal"""
    print("\n=== ENTRENANDO MODELO ===")
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("✓ Modelo entrenado exitosamente")
    
    # Mostrar coeficientes
    print("\nCoeficientes aprendidos:")
    for feature, coef in zip(X_train.columns, modelo.coef_):
        print(f"  {feature}: {coef:.2f}")
    print(f"\nIntercepto: {modelo.intercept_:.2f}")
    
    return modelo

def evaluate_model(modelo, X_test, y_test):
    """Evalúa el modelo con métricas"""
    print("\n=== EVALUACIÓN DEL MODELO ===")
    
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    print(f"\nEl modelo explica el {r2*100:.2f}% de la variación en los precios")
    
    # Comparación detallada
    comparacion = pd.DataFrame({
        'Real': y_test.values,
        'Predicción': y_pred,
        'Diferencia': y_test.values - y_pred
    })
    print("\nComparación Real vs Predicción:")
    print(comparacion)
    
    return y_pred, {'mae': mae, 'rmse': rmse, 'r2': r2}

def save_model(modelo):
    """Guarda el modelo entrenado"""
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"\n✓ Modelo guardado en: {MODEL_FILE}")

def load_model():
    """Carga el modelo guardado"""
    with open(MODEL_FILE, 'rb') as f:
        modelo = pickle.load(f)
    print(f"✓ Modelo cargado desde: {MODEL_FILE}")
    return modelo