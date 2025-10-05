# main_kaggle.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
from config import (
    KAGGLE_TRAIN_FILE, 
    KAGGLE_FEATURES, 
    KAGGLE_TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    KAGGLE_MODEL_FILE,
    REPORTS_DIR
)
import os

def main():
    """Pipeline con dataset de Kaggle"""
    
    # 1. Cargar datos
    print("=== CARGANDO DATOS DE KAGGLE ===")
    df = pd.read_csv(KAGGLE_TRAIN_FILE)
    print(f"✓ Datos cargados: {df.shape[0]} casas, {df.shape[1]} columnas")
    
    # 2. Preparar datos
    print("\n=== PREPARANDO DATOS ===")
    X = df[KAGGLE_FEATURES]
    y = df[KAGGLE_TARGET]
    print(f"Características: {X.shape[1]}")
    print(f"Target: {KAGGLE_TARGET}")
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"\nEntrenamiento: {X_train.shape[0]} casas")
    print(f"Prueba: {X_test.shape[0]} casas")
    
    # 4. Entrenar modelo
    print("\n=== ENTRENANDO MODELO ===")
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("✓ Modelo entrenado")
    
    # Mostrar coeficientes
    print("\nCaracterísticas más influyentes:")
    coef_df = pd.DataFrame({
        'Feature': KAGGLE_FEATURES,
        'Coeficiente': modelo.coef_
    }).sort_values('Coeficiente', key=abs, ascending=False)
    print(coef_df)
    
    # 5. Evaluar modelo
    print("\n=== EVALUACIÓN ===")
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    print(f"\nEl modelo explica el {r2*100:.2f}% de la variación")
    
    # 6. Visualizar predicciones
    print("\n=== GENERANDO GRÁFICA ===")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('Precio Real')
    plt.ylabel('Precio Predicho')
    plt.title(f'Predicciones vs Reales (R²={r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join(REPORTS_DIR, 'kaggle_predictions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {filepath}")
    plt.show()
    
    # 7. Guardar modelo
    with open(KAGGLE_MODEL_FILE, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"\n✓ Modelo guardado: {KAGGLE_MODEL_FILE}")
    
    print("\n🎉 Pipeline completado con dataset de Kaggle")

if __name__ == "__main__":
    main()