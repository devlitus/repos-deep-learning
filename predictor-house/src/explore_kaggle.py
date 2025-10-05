# explore_kaggle.py
import pandas as pd
import sys
import os

# Añadir el directorio padre al path para poder importar config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import KAGGLE_TRAIN_FILE

# Cargar datos
print("Cargando dataset de Kaggle...")
df = pd.read_csv(KAGGLE_TRAIN_FILE)

print(f"\n=== INFORMACIÓN GENERAL ===")
print(f"Dimensiones: {df.shape}")
print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")

print(f"\n=== PRIMERAS FILAS ===")
print(df.head())

print(f"\n=== TIPOS DE DATOS ===")
print(df.dtypes)

print(f"\n=== COLUMNAS DISPONIBLES ===")
print(df.columns.tolist())

print(f"\n=== VALORES FALTANTES ===")
faltantes = df.isnull().sum()
print(faltantes[faltantes > 0].sort_values(ascending=False))

print(f"\n=== ESTADÍSTICAS DEL PRECIO (Target) ===")
print(df['SalePrice'].describe())


print(f"\n=== CARACTERÍSTICAS RECOMENDADAS ===")

# Características numéricas importantes con pocos faltantes
features_numericas = [
    'GrLivArea',        # Área habitable sobre el suelo (m²)
    'TotalBsmtSF',      # Área total del sótano (m²)
    'OverallQual',      # Calidad general (1-10)
    'OverallCond',      # Condición general (1-10)
    'YearBuilt',        # Año de construcción
    'YearRemodAdd',     # Año de remodelación
    'GarageCars',       # Capacidad del garaje (# de autos)
    'GarageArea',       # Área del garaje (m²)
    'FullBath',         # Baños completos
    'BedroomAbvGr',     # Habitaciones sobre el suelo
    'TotRmsAbvGrd',     # Total de habitaciones sobre el suelo
    'Fireplaces',       # Número de chimeneas
]

print("\nCaracterísticas seleccionadas:")
for feat in features_numericas:
    faltantes = df[feat].isnull().sum()
    print(f"  {feat}: {faltantes} valores faltantes")

# Verificar correlación con el precio
print(f"\n=== CORRELACIÓN CON EL PRECIO ===")
correlaciones = df[features_numericas + ['SalePrice']].corr()['SalePrice'].sort_values(ascending=False)
print(correlaciones)