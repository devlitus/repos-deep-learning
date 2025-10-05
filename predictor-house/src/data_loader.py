# src/data_loader.py
import pandas as pd
from config import RAW_DATA_FILE, FEATURES, TARGET

def load_data():
    """Carga los datos desde el CSV"""
    print("Cargando datos...")
    df = pd.read_csv(RAW_DATA_FILE)
    df.columns = df.columns.str.strip()
    print(f"✓ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def explore_data(df):
    """Muestra información básica del dataset"""
    print("\n=== EXPLORACIÓN DE DATOS ===")
    print("\nPrimeras filas:")
    print(df.head())
    
    print("\nInformación:")
    print(df.info())
    
    print("\nEstadísticas:")
    print(df.describe())
    
    print("\nValores faltantes:")
    print(df.isnull().sum())
    
    return df

def prepare_data(df):
    """Separa características (X) y objetivo (y)"""
    X = df[FEATURES]
    y = df[TARGET]
    return X, y