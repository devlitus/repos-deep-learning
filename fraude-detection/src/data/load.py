"""
Módulo para carga y exploración de datos
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from config.config import RAW_DATA_FILE


def load_data():
    """
    Carga el dataset de transacciones de tarjetas de crédito

    Returns:
        pd.DataFrame: Dataset cargado, o None si hay error
    """
    print("\n" + "="*60)
    print("📂 CARGANDO DATOS")
    print("="*60)

    try:
        print(f"\n📍 Buscando archivo: {RAW_DATA_FILE}")

        if not RAW_DATA_FILE.exists():
            print(f"\n❌ ERROR: No se encuentra el archivo de datos")
            print(f"   Ruta esperada: {RAW_DATA_FILE}")
            print("\n💡 Solución:")
            print("   1. Descarga el dataset de Kaggle:")
            print("      https://www.kaggle.com/mlg-ulb/creditcardfraud")
            print(f"   2. Coloca el archivo 'creditcard.csv' en: {RAW_DATA_FILE.parent}")
            return None

        print("⏳ Cargando dataset (puede tardar unos segundos)...")
        df = pd.read_csv(RAW_DATA_FILE)

        print(f"\n✅ Dataset cargado exitosamente!")
        print(f"   • Filas: {len(df):,}")
        print(f"   • Columnas: {len(df.columns)}")
        print(f"   • Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        return df

    except Exception as e:
        print(f"\n❌ ERROR al cargar el dataset: {e}")
        return None


def explore_data(df):
    """
    Muestra información básica sobre el dataset

    Args:
        df (pd.DataFrame): Dataset a explorar
    """
    print("\n" + "="*60)
    print("🔍 EXPLORACIÓN INICIAL DE DATOS")
    print("="*60)

    # Información general
    print("\n1️⃣ INFORMACIÓN GENERAL:")
    print(f"   • Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"   • Columnas: {list(df.columns)}")

    # Valores nulos
    print("\n2️⃣ VALORES NULOS:")
    null_counts = df.isnull().sum()
    if null_counts.sum() == 0:
        print("   ✅ No hay valores nulos en el dataset")
    else:
        print(null_counts[null_counts > 0])

    # Distribución de clases
    print("\n3️⃣ DISTRIBUCIÓN DE CLASES:")
    class_counts = df['Class'].value_counts()
    class_percentages = df['Class'].value_counts(normalize=True) * 100

    print(f"   • Transacciones Legítimas (Class=0): {class_counts[0]:,} ({class_percentages[0]:.4f}%)")
    print(f"   • Transacciones Fraudulentas (Class=1): {class_counts[1]:,} ({class_percentages[1]:.4f}%)")
    print(f"   • Ratio de desbalanceo: 1:{class_counts[0]/class_counts[1]:.0f}")

    # Estadísticas de Amount
    print("\n4️⃣ ESTADÍSTICAS DE 'AMOUNT':")
    print(f"   • Mínimo: ${df['Amount'].min():.2f}")
    print(f"   • Máximo: ${df['Amount'].max():.2f}")
    print(f"   • Media: ${df['Amount'].mean():.2f}")
    print(f"   • Mediana: ${df['Amount'].median():.2f}")

    # Primeras filas
    print("\n5️⃣ PRIMERAS 3 FILAS:")
    print(df.head(3).to_string())

    print("\n" + "="*60)


if __name__ == "__main__":
    # Prueba del módulo
    print("🚀 Iniciando carga de datos...")

    data = load_data()

    if data is not None:
        explore_data(data)
        print("\n✅ Módulo de carga funcionando correctamente")
    else:
        print("\n❌ No se pudo cargar el dataset")
