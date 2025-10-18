"""
═══════════════════════════════════════════════════════════════
MÓDULO: CARGA DE DATOS
═══════════════════════════════════════════════════════════════

Propósito: Descargar y cargar el dataset de temperaturas

Funciones:
1. download_data() → Descarga el CSV si no existe
2. load_data() → Carga el CSV en un DataFrame
3. get_temperature_data() → Extrae solo las temperaturas
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_URL, DATA_RAW


def download_data():
    """
    Descarga el dataset de temperaturas desde internet
    
    ¿Qué hace?
    1. Verifica si la carpeta 'data/' existe, si no, la crea
    2. Verifica si el archivo ya existe
    3. Si no existe, lo descarga desde la URL
    4. Lo guarda como CSV
    
    Returns:
        bool: True si todo salió bien, False si hubo error
    """
    try:
        # Crear carpeta 'data' si no existe
        # exist_ok=True significa "no des error si ya existe"
        os.makedirs('data', exist_ok=True)
        
        # Verificar si el archivo ya fue descargado
        if not os.path.exists(DATA_RAW):
            print(f"📥 Descargando datos desde {DATA_URL}...")
            
            # pd.read_csv() lee un CSV desde internet o desde el disco
            df = pd.read_csv(DATA_URL)
            
            # Guardar el DataFrame como CSV en nuestro disco
            # index=False significa "no guardes el índice numérico"
            df.to_csv(DATA_RAW, index=False)
            
            print(f"✅ Datos descargados en {DATA_RAW}")
        else:
            print(f"✅ Datos ya existen en {DATA_RAW}")
        
        return True
        
    except Exception as e:
        # Si algo falla, captura el error y lo muestra
        print(f"❌ Error al descargar datos: {e}")
        return False


def load_data():
    """
    Carga el dataset desde el archivo CSV
    
    ¿Qué hace?
    Lee el CSV y lo convierte en un DataFrame (tabla) de pandas
    
    ¿Qué es un DataFrame?
    Es como una hoja de Excel en Python: tiene filas y columnas
    
    Returns:
        pd.DataFrame: Tabla con los datos, o None si hay error
    """
    try:
        # Leer el CSV
        df = pd.read_csv(DATA_RAW)
        
        # Mostrar información básica
        print(f"📊 Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        return df
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None


def get_temperature_data(df):
    """
    Extrae solo la columna de temperaturas

    ¿Por qué?
    El CSV tiene 2 columnas: 'Date' y 'Temp'
    Solo necesitamos 'Temp' para entrenar

    ¿Qué hace?
    1. Extrae la columna 'Temp'
    2. La convierte en un array de NumPy
    3. La reshape a formato (n, 1) que necesita el modelo

    Args:
        df: DataFrame con los datos

    Returns:
        np.array: Array con las temperaturas en formato (n, 1)
                  Ejemplo: [[20], [21], [19], ...]
    """
    # .values convierte columna de pandas a array de numpy
    # .reshape(-1, 1) convierte [20, 21, 19] a [[20], [21], [19]]
    # -1 significa "calcula automáticamente según los datos"
    # 1 significa "una columna"
    data = df['Temp'].values.reshape(-1, 1)

    # Mostrar estadísticas básicas
    print(f"🌡️  Temperaturas extraídas: {len(data)} valores")
    print(f"   Rango: {data.min():.2f}°C a {data.max():.2f}°C")
    print(f"   Promedio: {data.mean():.2f}°C")

    return data


def load_melbourne_data():
    """
    Función de conveniencia que carga y prepara los datos de temperatura

    ¿Qué hace?
    1. Descarga los datos si no existen
    2. Carga el DataFrame
    3. Extrae las temperaturas
    4. Normaliza los datos usando MinMaxScaler

    Returns:
        tuple: (df, data_normalized, scaler)
            - df: DataFrame original con datos
            - data_normalized: Array normalizado en rango [0, 1]
            - scaler: Objeto MinMaxScaler para desnormalizar después
    """
    # Paso 1: Descargar datos
    download_data()

    # Paso 2: Cargar datos
    df = load_data()

    if df is None:
        raise ValueError("No se pudieron cargar los datos")

    # Paso 3: Extraer temperaturas
    data = get_temperature_data(df)

    # Paso 4: Normalizar datos
    print("\n📊 Normalizando datos al rango [0, 1]...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)

    print("✅ Datos normalizados")

    # Renombrar columna para compatibilidad
    df = df.rename(columns={'Temp': 'Temp'})

    return df, data_normalized, scaler


# ═══════════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA
# ═══════════════════════════════════════════════════════════════
# Este código solo se ejecuta si corremos este archivo directamente
# No se ejecuta cuando lo importamos en otro archivo

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 PROBANDO MÓDULO DE CARGA DE DATOS")
    print("="*70 + "\n")
    
    # Probar descarga
    download_data()
    
    # Probar carga
    df = load_data()
    
    if df is not None:
        print("\n📋 Primeras 5 filas del dataset:")
        print(df.head())
        
        print("\n📋 Últimas 5 filas del dataset:")
        print(df.tail())
        
        print("\n📋 Información del dataset:")
        print(df.info())
        
        # Probar extracción de temperaturas
        data = get_temperature_data(df)
        print(f"\n🔢 Forma del array de temperaturas: {data.shape}")
        print(f"   Primeros 5 valores: {data[:5].flatten()}")