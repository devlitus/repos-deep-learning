import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess_titanic_data(df):
    """
    Preprocesa el dataset del Titanic:
    - Rellena valores faltantes
    - Convierte variables categóricas a numéricas
    - Selecciona las columnas relevantes
    
    Parámetros:
    -----------
    df : DataFrame
        Dataset original del Titanic
        
    Retorna:
    --------
    df_clean : DataFrame
        Dataset limpio y preprocesado
    """
    
    print("=" * 60)
    print("🧹 INICIANDO PREPROCESAMIENTO DE DATOS")
    print("=" * 60)
    
    # Hacer una copia para no modificar el original
    df_clean = df.copy()
    
    # ============================================
    # 1. VALORES FALTANTES
    # ============================================
    print("\n📋 PASO 1: Manejo de valores faltantes")
    print("-" * 60)
    
    # Mostrar valores faltantes antes
    print("\nValores faltantes ANTES:")
    missing_before = df_clean.isnull().sum()
    print(missing_before[missing_before > 0])
    
    # Rellenar EDAD con la mediana
    mediana_edad = df_clean['age'].median()
    df_clean['age'] = df_clean['age'].fillna(mediana_edad)
    print(f"\n✅ Edad: Rellenados {missing_before['age']} valores con mediana ({mediana_edad:.1f} años)")
    
    # Rellenar EMBARKED con el valor más frecuente
    moda_embarked = df_clean['embarked'].mode()[0]
    df_clean['embarked'] = df_clean['embarked'].fillna(moda_embarked)
    print(f"✅ Embarked: Rellenados {missing_before['embarked']} valores con moda ('{moda_embarked}')")
    
    # Rellenar FARE con la mediana (por si acaso)
    if df_clean['fare'].isnull().sum() > 0:
        mediana_fare = df_clean['fare'].median()
        df_clean['fare'] = df_clean['fare'].fillna(mediana_fare)
        print(f"✅ Fare: Rellenados valores con mediana ({mediana_fare:.2f})")
    
    # Mostrar valores faltantes después
    print("\nValores faltantes DESPUÉS:")
    missing_after = df_clean.isnull().sum()
    print(missing_after[missing_after > 0] if missing_after.sum() > 0 else "¡Ninguno! ✅")
    
    # ============================================
    # 2. CONVERTIR VARIABLES CATEGÓRICAS A NUMÉRICAS
    # ============================================
    print("\n📋 PASO 2: Conversión de variables categóricas a numéricas")
    print("-" * 60)
    
    # SEX: male/female → 0/1
    df_clean['sex'] = df_clean['sex'].map({'male': 0, 'female': 1})
    print("✅ Sex: 'male' → 0, 'female' → 1")
    
    # EMBARKED: C/Q/S → 0/1/2
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    df_clean['embarked'] = df_clean['embarked'].map(embarked_mapping)
    print("✅ Embarked: 'C' → 0, 'Q' → 1, 'S' → 2")
    
    # ============================================
    # 3. FEATURE ENGINEERING (crear nuevas variables)
    # ============================================
    print("\n📋 PASO 3: Feature Engineering")
    print("-" * 60)
    
    # Crear: FAMILY_SIZE (tamaño de familia a bordo)
    df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1
    print("✅ Creada variable 'family_size' = sibsp + parch + 1")
    
    # Crear: IS_ALONE (viaja solo o acompañado)
    df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)
    print("✅ Creada variable 'is_alone' (1 si viaja solo, 0 si no)")
    
    # ============================================
    # 4. SELECCIONAR COLUMNAS RELEVANTES
    # ============================================
    print("\n📋 PASO 4: Selección de columnas")
    print("-" * 60)
    
    # Columnas que usaremos para el modelo
    columnas_seleccionadas = [
        'survived',      # Variable objetivo
        'pclass',        # Clase del ticket
        'sex',           # Sexo (0=male, 1=female)
        'age',           # Edad
        'sibsp',         # Hermanos/cónyuges
        'parch',         # Padres/hijos
        'fare',          # Precio del ticket
        'embarked',      # Puerto de embarque
        'family_size',   # Tamaño de familia
        'is_alone'       # Viaja solo
    ]
    
    df_clean = df_clean[columnas_seleccionadas]
    
    print(f"✅ Seleccionadas {len(columnas_seleccionadas)} columnas:")
    for col in columnas_seleccionadas:
        print(f"   - {col}")
    
    # ============================================
    # 5. VERIFICACIÓN FINAL
    # ============================================
    print("\n📋 PASO 5: Verificación final")
    print("-" * 60)
    
    print(f"✅ Dimensiones finales: {df_clean.shape[0]} filas x {df_clean.shape[1]} columnas")
    print(f"✅ Valores faltantes totales: {df_clean.isnull().sum().sum()}")
    print(f"✅ Tipos de datos correctos: {df_clean.dtypes.value_counts().to_dict()}")
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESAMIENTO COMPLETADO")
    print("=" * 60)
    
    return df_clean


# ============================================
# FUNCIÓN PARA SEPARAR X (features) y y (target)
# ============================================
def split_features_target(df):
    """
    Separa el dataset en features (X) y target (y)
    
    Parámetros:
    -----------
    df : DataFrame
        Dataset preprocesado
        
    Retorna:
    --------
    X : DataFrame
        Features (variables predictoras)
    y : Series
        Target (variable objetivo)
    """
    
    X = df.drop('survived', axis=1)
    y = df['survived']
    
    print("\n📊 Separación de datos:")
    print(f"   X (features): {X.shape}")
    print(f"   y (target): {y.shape}")
    
    return X, y


# ============================================
# FUNCIÓN PRINCIPAL PARA TESTEAR
# ============================================
if __name__ == "__main__":
    import seaborn as sns
    
    print("\n🧪 EJECUTANDO PREPROCESAMIENTO EN MODO TEST\n")
    
    # Cargar datos
    titanic = sns.load_dataset('titanic')
    print(f"Dataset original: {titanic.shape}")
    
    # Preprocesar
    titanic_clean = preprocess_titanic_data(titanic)
    
    # Mostrar primeras filas del dataset limpio
    print("\n📊 PRIMERAS 5 FILAS DEL DATASET LIMPIO:")
    print(titanic_clean.head())
    
    # Mostrar estadísticas
    print("\n📊 ESTADÍSTICAS DEL DATASET LIMPIO:")
    print(titanic_clean.describe())
    
    # Separar X e y
    X, y = split_features_target(titanic_clean)
    
    print("\n✅ Preprocesamiento completado exitosamente")