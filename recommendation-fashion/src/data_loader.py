"""
Módulo de Carga y Preprocesamiento de Datos
Carga el dataset de Amazon Fashion Reviews y realiza limpieza y exploración
"""

import sys
import io
import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# Configurar UTF-8 para Windows
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_FILE,
    COL_USER_ID,
    COL_PRODUCT_ID,
    COL_RATING,
    COL_REVIEW_TEXT,
    COL_SUMMARY,
    COL_TIMESTAMP,
    COL_VERIFIED,
    MIN_USER_RATINGS,
    MIN_PRODUCT_RATINGS,
    MIN_RATING_SCORE,
    MAX_RATING_SCORE,
    SAMPLE_SIZE,
    SAMPLE_RANDOM_STATE
)

warnings.filterwarnings('ignore')

def print_header(text):
    """Imprime un encabezado decorativo"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)

def print_step(text):
    """Imprime un paso del proceso"""
    print(f"\n{'─' * 80}")
    print(f"📍 {text}")
    print("─" * 80)

def print_success(text):
    """Imprime mensaje de éxito"""
    print(f"✅ {text}")

def print_info(text):
    """Imprime información"""
    print(f"ℹ️  {text}")

def load_data(filepath=None):
    """
    Carga el dataset de reviews desde archivo JSON (línea por línea)

    Args:
        filepath: Ruta al archivo JSON. Si es None, usa DATASET_FILE de config

    Returns:
        pd.DataFrame: DataFrame con las reviews cargadas
    """
    print_header("📥 CARGANDO DATASET DE FASHION REVIEWS")

    if filepath is None:
        filepath = DATASET_FILE

    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"El archivo {filepath} no existe.\n"
            "Ejecuta primero: python download_fashion.py"
        )

    print_step("Leyendo archivo JSON")
    print(f"  Archivo: {Path(filepath).name}")

    # Leer JSON línea por línea
    reviews = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                reviews.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Error en línea {i+1}: {e}")
                continue

            # Mostrar progreso cada 100k líneas
            if (i + 1) % 100000 == 0:
                print(f"  Procesadas {i + 1:,} líneas...")

    df = pd.DataFrame(reviews)

    # Aplicar sampling si está configurado
    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        print_step(f"Aplicando muestreo aleatorio: {SAMPLE_SIZE:,} reviews")
        df = df.sample(n=SAMPLE_SIZE, random_state=SAMPLE_RANDOM_STATE)
        df = df.reset_index(drop=True)

    print_success(f"Dataset cargado: {len(df):,} reviews")
    print(f"  Columnas: {', '.join(df.columns.tolist())}")

    return df

def explore_data(df):
    """
    Explora y muestra estadísticas descriptivas del dataset

    Args:
        df: DataFrame con las reviews
    """
    print_header("🔍 ANÁLISIS EXPLORATORIO DE DATOS")

    # Información general
    print_step("Información General")
    print(f"  Total de reviews: {len(df):,}")
    print(f"  Columnas: {len(df.columns)}")
    print(f"  Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Tipos de datos
    print_step("Tipos de Datos")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    # Valores faltantes
    print_step("Valores Faltantes")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print_success("No hay valores faltantes")
    else:
        for col, count in missing.items():
            if count > 0:
                pct = (count / len(df)) * 100
                print(f"  {col}: {count:,} ({pct:.2f}%)")

    # Estadísticas de usuarios
    print_step("Estadísticas de Usuarios")
    n_users = df[COL_USER_ID].nunique()
    reviews_per_user = df.groupby(COL_USER_ID).size()

    print(f"  Usuarios únicos: {n_users:,}")
    print(f"  Reviews por usuario:")
    print(f"    - Promedio: {reviews_per_user.mean():.2f}")
    print(f"    - Mediana: {reviews_per_user.median():.0f}")
    print(f"    - Mínimo: {reviews_per_user.min()}")
    print(f"    - Máximo: {reviews_per_user.max()}")
    print(f"    - Desv. estándar: {reviews_per_user.std():.2f}")

    # Estadísticas de productos
    print_step("Estadísticas de Productos")
    n_products = df[COL_PRODUCT_ID].nunique()
    reviews_per_product = df.groupby(COL_PRODUCT_ID).size()

    print(f"  Productos únicos: {n_products:,}")
    print(f"  Reviews por producto:")
    print(f"    - Promedio: {reviews_per_product.mean():.2f}")
    print(f"    - Mediana: {reviews_per_product.median():.0f}")
    print(f"    - Mínimo: {reviews_per_product.min()}")
    print(f"    - Máximo: {reviews_per_product.max()}")
    print(f"    - Desv. estándar: {reviews_per_product.std():.2f}")

    # Estadísticas de ratings
    print_step("Estadísticas de Ratings")
    print(f"  Rating promedio: {df[COL_RATING].mean():.2f}")
    print(f"  Mediana: {df[COL_RATING].median():.2f}")
    print(f"  Desviación estándar: {df[COL_RATING].std():.2f}")
    print(f"  Mínimo: {df[COL_RATING].min():.2f}")
    print(f"  Máximo: {df[COL_RATING].max():.2f}")

    print("\n  Distribución de ratings:")
    rating_dist = df[COL_RATING].value_counts().sort_index()
    for rating in sorted(rating_dist.index)[:10]:  # Primeros 10
        count = rating_dist[rating]
        pct = (count / len(df)) * 100
        bar = '█' * int(pct)
        print(f"    {rating:.1f}: {count:5,} ({pct:5.1f}%) {bar}")

    # Sparsity de la matriz
    print_step("Análisis de Sparsity")
    total_interactions = len(df)
    possible_interactions = n_users * n_products
    sparsity = (1 - (total_interactions / possible_interactions)) * 100

    print(f"  Interacciones actuales: {total_interactions:,}")
    print(f"  Interacciones posibles: {possible_interactions:,}")
    print(f"  Sparsity: {sparsity:.4f}%")
    print(f"  Densidad: {100 - sparsity:.4f}%")

    if sparsity > 99.5:
        print("  ⚠️  Matriz muy dispersa - considerar filtrado agresivo")
    elif sparsity > 99:
        print("  ℹ️  Matriz dispersa - filtrado recomendado")
    else:
        print("  ✅ Densidad aceptable para collaborative filtering")

    # Información adicional si existe
    if COL_VERIFIED in df.columns:
        verified_count = df[COL_VERIFIED].sum()
        verified_pct = (verified_count / len(df)) * 100
        print_step("Reviews Verificadas")
        print(f"  Verificadas: {verified_count:,} ({verified_pct:.1f}%)")
        print(f"  No verificadas: {len(df) - verified_count:,} ({100 - verified_pct:.1f}%)")

def prepare_data(df):
    """
    Preprocesa el dataset: limpieza, filtrado y preparación para modelado

    Args:
        df: DataFrame original con las reviews

    Returns:
        pd.DataFrame: DataFrame limpio y preparado
    """
    print_header("🧹 PREPROCESAMIENTO DE DATOS")

    # Crear copia para no modificar el original
    df_clean = df.copy()
    print_info(f"Dataset original: {len(df_clean):,} reviews")

    # Paso 1: Verificar y renombrar columnas si es necesario
    print_step("Verificando estructura de columnas")
    required_cols = {COL_USER_ID, COL_PRODUCT_ID, COL_RATING}
    if not required_cols.issubset(df_clean.columns):
        missing = required_cols - set(df_clean.columns)
        raise ValueError(f"Columnas faltantes: {missing}")
    print_success("Todas las columnas requeridas están presentes")

    # Paso 2: Eliminar duplicados
    print_step("Eliminando duplicados")
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=[COL_USER_ID, COL_PRODUCT_ID], keep='first')
    duplicates_removed = initial_count - len(df_clean)

    if duplicates_removed > 0:
        print(f"  Duplicados eliminados: {duplicates_removed:,}")
        print_success(f"Reviews después de eliminar duplicados: {len(df_clean):,}")
    else:
        print_success("No se encontraron duplicados")

    # Paso 3: Filtrar ratings fuera de rango
    print_step("Filtrando ratings fuera de rango")
    initial_count = len(df_clean)
    df_clean = df_clean[
        (df_clean[COL_RATING] >= MIN_RATING_SCORE) &
        (df_clean[COL_RATING] <= MAX_RATING_SCORE)
    ]
    invalid_ratings = initial_count - len(df_clean)

    if invalid_ratings > 0:
        print(f"  Ratings inválidos eliminados: {invalid_ratings:,}")
        print_success(f"Reviews válidas: {len(df_clean):,}")
    else:
        print_success(f"Todos los ratings están en rango [{MIN_RATING_SCORE}, {MAX_RATING_SCORE}]")

    # Paso 4: Filtrar usuarios con pocas reviews
    print_step(f"Filtrando usuarios con menos de {MIN_USER_RATINGS} reviews")
    user_counts = df_clean[COL_USER_ID].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index

    initial_users = df_clean[COL_USER_ID].nunique()
    df_clean = df_clean[df_clean[COL_USER_ID].isin(valid_users)]
    final_users = df_clean[COL_USER_ID].nunique()

    print(f"  Usuarios antes: {initial_users:,}")
    print(f"  Usuarios después: {final_users:,}")
    print(f"  Usuarios eliminados: {initial_users - final_users:,}")
    print_success(f"Reviews restantes: {len(df_clean):,}")

    # Paso 5: Filtrar productos con pocas reviews
    print_step(f"Filtrando productos con menos de {MIN_PRODUCT_RATINGS} reviews")
    product_counts = df_clean[COL_PRODUCT_ID].value_counts()
    valid_products = product_counts[product_counts >= MIN_PRODUCT_RATINGS].index

    initial_products = df_clean[COL_PRODUCT_ID].nunique()
    df_clean = df_clean[df_clean[COL_PRODUCT_ID].isin(valid_products)]
    final_products = df_clean[COL_PRODUCT_ID].nunique()

    print(f"  Productos antes: {initial_products:,}")
    print(f"  Productos después: {final_products:,}")
    print(f"  Productos eliminados: {initial_products - final_products:,}")
    print_success(f"Reviews finales: {len(df_clean):,}")

    # Paso 6: Re-indexar IDs para eficiencia
    print_step("Re-indexando IDs de usuarios y productos")

    # Crear mapeos de IDs originales a índices numéricos
    user_id_map = {uid: idx for idx, uid in enumerate(df_clean[COL_USER_ID].unique())}
    product_id_map = {pid: idx for idx, pid in enumerate(df_clean[COL_PRODUCT_ID].unique())}

    # Crear columnas con IDs numéricos
    df_clean['user_idx'] = df_clean[COL_USER_ID].map(user_id_map)
    df_clean['product_idx'] = df_clean[COL_PRODUCT_ID].map(product_id_map)

    print_success(f"IDs re-indexados: {len(user_id_map):,} usuarios, {len(product_id_map):,} productos")

    # Paso 7: Ordenar por timestamp si existe
    if COL_TIMESTAMP in df_clean.columns:
        print_step("Ordenando por timestamp")
        df_clean = df_clean.sort_values(COL_TIMESTAMP)
        df_clean = df_clean.reset_index(drop=True)
        print_success("Dataset ordenado cronológicamente")

    # Resumen final
    print_step("Resumen del Preprocesamiento")
    print(f"  Reviews originales: {len(df):,}")
    print(f"  Reviews finales: {len(df_clean):,}")
    print(f"  Reducción: {(1 - len(df_clean)/len(df)) * 100:.1f}%")
    print(f"\n  Usuarios finales: {df_clean[COL_USER_ID].nunique():,}")
    print(f"  Productos finales: {df_clean[COL_PRODUCT_ID].nunique():,}")
    print(f"  Rating promedio: {df_clean[COL_RATING].mean():.2f}")

    # Calcular nueva sparsity
    n_users = df_clean[COL_USER_ID].nunique()
    n_products = df_clean[COL_PRODUCT_ID].nunique()
    sparsity = (1 - (len(df_clean) / (n_users * n_products))) * 100
    print(f"  Sparsity final: {sparsity:.4f}%")

    print_success("¡Preprocesamiento completado!")

    # Guardar mapeos como atributos del DataFrame (útil para después)
    df_clean.attrs['user_id_map'] = user_id_map
    df_clean.attrs['product_id_map'] = product_id_map
    df_clean.attrs['user_id_reverse_map'] = {v: k for k, v in user_id_map.items()}
    df_clean.attrs['product_id_reverse_map'] = {v: k for k, v in product_id_map.items()}

    return df_clean

def get_user_item_matrix(df, fill_value=0):
    """
    Crea la matriz usuario-producto (user-item matrix)

    Args:
        df: DataFrame con las reviews (debe tener user_idx, product_idx, y rating)
        fill_value: Valor para rellenar interacciones faltantes (default: 0)

    Returns:
        pd.DataFrame: Matriz usuario-producto con ratings
    """
    print_step("Creando matriz usuario-producto")

    # Crear matriz pivotada
    user_item_matrix = df.pivot_table(
        index='user_idx',
        columns='product_idx',
        values=COL_RATING,
        fill_value=fill_value
    )

    print(f"  Dimensiones: {user_item_matrix.shape[0]:,} usuarios × {user_item_matrix.shape[1]:,} productos")
    print(f"  Total elementos: {user_item_matrix.size:,}")
    print(f"  Elementos no-cero: {(user_item_matrix != 0).sum().sum():,}")

    sparsity = (1 - (user_item_matrix != 0).sum().sum() / user_item_matrix.size) * 100
    print(f"  Sparsity: {sparsity:.4f}%")

    print_success("Matriz usuario-producto creada")

    return user_item_matrix

if __name__ == '__main__':
    """Ejecutar el módulo directamente para testing"""

    # Cargar datos
    df = load_data()

    # Explorar datos
    explore_data(df)

    # Preparar datos
    df_clean = prepare_data(df)

    # Crear matriz usuario-producto
    user_item_matrix = get_user_item_matrix(df_clean)

    print_header("✅ PROCESAMIENTO COMPLETADO")
    print("\nDataFrame limpio disponible en: df_clean")
    print("Matriz usuario-producto disponible en: user_item_matrix")
