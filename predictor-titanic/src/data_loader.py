import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar el dataset (viene incluido en seaborn)
titanic = sns.load_dataset('titanic')

# Ver las primeras filas
print(titanic.head())

# Información general del dataset
print("\n=== INFORMACIÓN GENERAL ===")
print(titanic.info())

# Estadísticas descriptivas
print("\n=== ESTADÍSTICAS ===")
print(titanic.describe())

# Dimensiones
print(f"\nFilas: {titanic.shape[0]}, Columnas: {titanic.shape[1]}")