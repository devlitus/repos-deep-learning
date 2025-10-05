# src/predictor.py
import pandas as pd
from config import FEATURES

def predict_new_houses(modelo, casas_nuevas_data):
    """
    Predice precios de casas nuevas
    
    Args:
        modelo: Modelo entrenado
        casas_nuevas_data: Lista de listas o DataFrame con las características
    
    Returns:
        predicciones: Array con los precios predichos
    """
    # Convertir a DataFrame si es necesario
    if not isinstance(casas_nuevas_data, pd.DataFrame):
        casas_nuevas = pd.DataFrame(casas_nuevas_data, columns=FEATURES)
    else:
        casas_nuevas = casas_nuevas_data
    
    print("\n=== PREDICIENDO CASAS NUEVAS ===")
    print(casas_nuevas)
    
    predicciones = modelo.predict(casas_nuevas)
    
    print("\nPredicciones:")
    for i, precio in enumerate(predicciones):
        print(f"Casa {i+1}: ${precio:,.2f}")
    
    return predicciones