"""
Módulo para carga y exploración inicial del dataset del Titanic
Sigue el patrón estándar: load_data() → explore_data() → prepare_data()
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from data_preprocessing import preprocess_titanic_data


def load_data():
    """
    Carga el dataset del Titanic desde seaborn
    
    Retorna:
    --------
    df : DataFrame
        Dataset original del Titanic (891 filas × 15 columnas)
    """
    # Cargar el dataset (viene incluido en seaborn)
    titanic = sns.load_dataset('titanic')
    return titanic


def explore_data(df):
    """
    Realiza exploración inicial del dataset
    Imprime información general, estadísticas y dimensiones
    
    Parámetros:
    -----------
    df : DataFrame
        Dataset del Titanic
        
    Retorna:
    --------
    df : DataFrame
        El mismo dataset (sin modificaciones)
    """
    # Ver las primeras filas
    print(df.head())

    # Información general del dataset
    print("\n=== INFORMACIÓN GENERAL ===")
    print(df.info())

    # Estadísticas descriptivas
    print("\n=== ESTADÍSTICAS ===")
    print(df.describe())

    # Dimensiones
    print(f"\nFilas: {df.shape[0]}, Columnas: {df.shape[1]}")
    
    return df


def prepare_data(df):
    """
    Preprocesa el dataset del Titanic
    Wrapper que llama a preprocess_titanic_data()
    
    Parámetros:
    -----------
    df : DataFrame
        Dataset original del Titanic
        
    Retorna:
    --------
    df_clean : DataFrame
        Dataset preprocesado y limpio
    """
    return preprocess_titanic_data(df)


# ============================================
# MODO TEST: Ejecutar si se llama directamente
# ============================================
if __name__ == "__main__":
    print("\n🧪 EJECUTANDO DATA LOADER EN MODO TEST\n")
    
    # Cargar datos
    titanic = load_data()
    print(f"✅ Dataset cargado: {titanic.shape}")
    
    # Explorar datos
    titanic = explore_data(titanic)
    
    print("\n✅ Exploración completada exitosamente")