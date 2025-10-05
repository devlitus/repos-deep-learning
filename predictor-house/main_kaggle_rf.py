# main_kaggle_rf.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # Nuevo algoritmo
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
from config import (
    KAGGLE_TRAIN_FILE, 
    KAGGLE_FEATURES, 
    KAGGLE_TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR,
    REPORTS_DIR
)
import os

def main():
    """Pipeline con Random Forest"""
    
    # 1. Cargar datos
    print("=== RANDOM FOREST CON DATASET KAGGLE ===")
    df = pd.read_csv(KAGGLE_TRAIN_FILE)
    print(f"✓ Datos cargados: {df.shape[0]} casas")
    
    # 2. Preparar datos
    X = df[KAGGLE_FEATURES]
    y = df[KAGGLE_TARGET]
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Entrenamiento: {X_train.shape[0]} casas")
    print(f"Prueba: {X_test.shape[0]} casas")
    
    # 4. Entrenar Random Forest
    print("\n=== ENTRENANDO RANDOM FOREST ===")
    modelo = RandomForestRegressor(
        n_estimators=100,      # 100 árboles
        max_depth=15,          # Profundidad máxima
        random_state=RANDOM_STATE,
        n_jobs=-1              # Usar todos los cores del CPU
    )
    modelo.fit(X_train, y_train)
    print("✓ Modelo entrenado")
    
    # Importancia de características
    print("\nImportancia de características:")
    importance_df = pd.DataFrame({
        'Feature': KAGGLE_FEATURES,
        'Importancia': modelo.feature_importances_
    }).sort_values('Importancia', ascending=False)
    print(importance_df)
    
    # 5. Evaluar
    print("\n=== EVALUACIÓN ===")
    y_pred = modelo.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: ${mae:,.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R²: {r2:.4f}")
    print(f"\nEl modelo explica el {r2*100:.2f}% de la variación")
    
    # 6. Comparar con Regresión Lineal
    print("\n=== COMPARACIÓN ===")
    print(f"Regresión Lineal R²: 0.7980")
    print(f"Random Forest R²: {r2:.4f}")
    mejora = ((r2 - 0.7980) / 0.7980) * 100
    print(f"Mejora: {mejora:+.2f}%")
    
    # 7. Visualizar
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('Precio Real')
    plt.ylabel('Precio Predicho')
    plt.title(f'Random Forest - Predicciones (R²={r2:.4f})')
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join(REPORTS_DIR, 'kaggle_rf_predictions.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada: {filepath}")
    plt.show()
    
    # 8. Guardar modelo
    model_file = os.path.join(MODELS_DIR, 'modelo_kaggle_rf.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"✓ Modelo guardado: {model_file}")
    
    print("\n🎉 Pipeline completado con Random Forest")

if __name__ == "__main__":
    main()