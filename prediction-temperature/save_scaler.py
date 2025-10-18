"""
Script para guardar el scaler después del entrenamiento.

Esto es necesario para que la aplicación web pueda normalizar
nuevas temperaturas usando el mismo scaler que el entrenamiento.
"""

import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

# Cargar datos directamente sin emojis
print("Cargando datos...")

# Leer CSV
import pandas as pd
csv_path = 'data/daily-min-temperatures.csv'
df = pd.read_csv(csv_path)

# Extraer temperaturas
data = df['Temp'].values.reshape(-1, 1)

print("Datos cargados: {} valores".format(len(data)))
print("Rango: {:.2f} a {:.2f} grados".format(data.min(), data.max()))

# Crear y entrenar scaler
print("Creando scaler...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data)

# Guardar scaler
scaler_path = 'models/scaler.pkl'
os.makedirs('models', exist_ok=True)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print("OK - Scaler guardado en: {}".format(scaler_path))
print("Puedes ejecutar: streamlit run web/app.py")
