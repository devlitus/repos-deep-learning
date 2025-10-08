"""
Módulo para preprocesamiento de datos
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from config.config import (
    RANDOM_STATE, 
    TEST_SIZE, 
    VALIDATION_SIZE,
    TARGET_COLUMN,
    AMOUNT_COLUMN,
    TIME_COLUMN,
    PROCESSED_DATA_DIR
)


def scale_features(df):
    """
    Escala las features Amount y Time usando StandardScaler
    
    StandardScaler transforma los datos para que tengan:
    - Media = 0
    - Desviación estándar = 1
    
    Propósito: Los modelos de ML funcionan mejor cuando las variables
    están en la misma escala.
    
    Args:
        df (pd.DataFrame): Dataset original
        
    Returns:
        pd.DataFrame: Dataset con features escaladas
        StandardScaler: Scaler ajustado (para usar en predicciones futuras)
    """
    df_scaled = df.copy()
    
    # Inicializar el scaler
    scaler = StandardScaler()
    
    # Escalar Amount y Time
    df_scaled[[AMOUNT_COLUMN, TIME_COLUMN]] = scaler.fit_transform(
        df[[AMOUNT_COLUMN, TIME_COLUMN]]
    )
    
    print("✅ Features escaladas: Amount y Time")
    print(f"   • Amount - Media: {df_scaled[AMOUNT_COLUMN].mean():.6f}, Std: {df_scaled[AMOUNT_COLUMN].std():.6f}")
    print(f"   • Time - Media: {df_scaled[TIME_COLUMN].mean():.6f}, Std: {df_scaled[TIME_COLUMN].std():.6f}")
    
    return df_scaled, scaler


def split_data(df, test_size=TEST_SIZE, validation_size=VALIDATION_SIZE, random_state=RANDOM_STATE):
    """
    Divide el dataset en train, validation y test
    
    División estratificada: Mantiene la misma proporción de clases
    en cada conjunto.
    
    Args:
        df (pd.DataFrame): Dataset completo
        test_size (float): Proporción para test (default: 0.2 = 20%)
        validation_size (float): Proporción para validation del train (default: 0.2 = 20%)
        random_state (int): Semilla aleatoria para reproducibilidad
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    print("\n" + "="*60)
    print("📊 DIVISIÓN DE DATOS")
    print("="*60)
    
    # Separar features (X) y target (y)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    # Primera división: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Mantener proporción de clases
    )
    
    # Segunda división: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_temp
    )
    
    # Mostrar información
    print(f"\n1️⃣ TRAIN SET:")
    print(f"   • Total: {len(X_train):,} muestras ({len(X_train)/len(df)*100:.1f}%)")
    print(f"   • Legítimas: {(y_train == 0).sum():,}")
    print(f"   • Fraudes: {(y_train == 1).sum():,}")
    
    print(f"\n2️⃣ VALIDATION SET:")
    print(f"   • Total: {len(X_val):,} muestras ({len(X_val)/len(df)*100:.1f}%)")
    print(f"   • Legítimas: {(y_val == 0).sum():,}")
    print(f"   • Fraudes: {(y_val == 1).sum():,}")
    
    print(f"\n3️⃣ TEST SET:")
    print(f"   • Total: {len(X_test):,} muestras ({len(X_test)/len(df)*100:.1f}%)")
    print(f"   • Legítimas: {(y_test == 0).sum():,}")
    print(f"   • Fraudes: {(y_test == 1).sum():,}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, sampling_strategy=0.5, random_state=RANDOM_STATE):
    """
    Aplica SMOTE para balancear las clases en el conjunto de entrenamiento
    
    SMOTE (Synthetic Minority Over-sampling Technique):
    - Crea ejemplos sintéticos de la clase minoritaria (fraudes)
    - NO duplica ejemplos existentes, sino que genera nuevos
    - Interpola entre ejemplos cercanos de la clase minoritaria
    
    Propósito: Ayudar al modelo a aprender mejor los patrones de fraude
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        sampling_strategy (float): Ratio deseado minority/majority (default: 0.5)
        random_state (int): Semilla aleatoria
        
    Returns:
        tuple: (X_train_balanced, y_train_balanced)
    """
    print("\n" + "="*60)
    print("⚖️ APLICANDO SMOTE (BALANCEO DE CLASES)")
    print("="*60)
    
    print(f"\n📌 ANTES de SMOTE:")
    print(f"   • Legítimas: {(y_train == 0).sum():,}")
    print(f"   • Fraudes: {(y_train == 1).sum():,}")
    print(f"   • Ratio: 1:{(y_train == 0).sum() / (y_train == 1).sum():.0f}")
    
    # Aplicar SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"\n📌 DESPUÉS de SMOTE:")
    print(f"   • Legítimas: {(y_train_balanced == 0).sum():,}")
    print(f"   • Fraudes: {(y_train_balanced == 1).sum():,}")
    print(f"   • Ratio: 1:{(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.1f}")
    
    print(f"\n✅ Se generaron {(y_train_balanced == 1).sum() - (y_train == 1).sum():,} ejemplos sintéticos de fraudes")
    
    return X_train_balanced, y_train_balanced


def preprocess_pipeline(df, apply_balancing=True):
    """
    Pipeline completo de preprocesamiento
    
    Pasos:
    1. Escalar features (Amount y Time)
    2. Dividir en train/val/test
    3. Aplicar SMOTE (opcional)
    
    Args:
        df (pd.DataFrame): Dataset original
        apply_balancing (bool): Si True, aplica SMOTE
        
    Returns:
        dict: Diccionario con todos los conjuntos de datos y el scaler
    """
    print("\n" + "="*60)
    print("🔧 PIPELINE DE PREPROCESAMIENTO")
    print("="*60)
    
    # 1. Escalar features
    print("\n1️⃣ Escalando features...")
    df_scaled, scaler = scale_features(df)
    
    # 2. Dividir datos
    print("\n2️⃣ Dividiendo datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_scaled)
    
    # 3. Aplicar SMOTE (solo en train)
    if apply_balancing:
        print("\n3️⃣ Aplicando balanceo...")
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    else:
        print("\n3️⃣ Balanceo desactivado")
        X_train_balanced = X_train
        y_train_balanced = y_train
    
    print("\n" + "="*60)
    print("✅ PREPROCESAMIENTO COMPLETADO")
    print("="*60)
    
    return {
        'X_train': X_train_balanced,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train_balanced,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'X_train_original': X_train,  # Sin SMOTE
        'y_train_original': y_train   # Sin SMOTE
    }


if __name__ == "__main__":
    # Prueba del módulo
    from src.data.load import load_data
    
    print("🚀 Iniciando preprocesamiento de datos...")
    
    # Cargar datos
    data = load_data()
    
    if data is not None:
        # Ejecutar pipeline completo
        processed_data = preprocess_pipeline(data, apply_balancing=True)
        
        print("\n📦 Datos preprocesados disponibles:")
        print(f"   • X_train: {processed_data['X_train'].shape}")
        print(f"   • X_val: {processed_data['X_val'].shape}")
        print(f"   • X_test: {processed_data['X_test'].shape}")