"""
Módulo de Análisis de Dispersión (Sparsity)
Analiza la dispersión de la matriz usuario-producto y sus implicaciones
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    COL_USER_ID, COL_PRODUCT_ID, COL_RATING,
    REPORTS_DIR, PLOT_STYLE, PLOT_DPI
)

warnings.filterwarnings('ignore')

# Ruta del reporte
REPORT_SPARSITY = REPORTS_DIR / 'sparsity_analysis.png'


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(text):
    print(f"\n{'─' * 80}")
    print(f"📍 {text}")
    print("─" * 80)


def calculate_sparsity(rating_matrix):
    """
    Calcula el porcentaje de dispersión de la matriz usuario-producto.

    Args:
        rating_matrix: DataFrame (usuarios x productos) con ratings

    Returns:
        float: Porcentaje de dispersión (0-100)
    """
    total_cells = rating_matrix.shape[0] * rating_matrix.shape[1]
    filled_cells = (rating_matrix != 0).sum().sum()
    sparsity = (1 - filled_cells / total_cells) * 100
    return sparsity


def analyze_sparsity(df):
    """
    Realiza un análisis completo de dispersión del dataset.

    Args:
        df: DataFrame con las reviews

    Returns:
        dict: Estadísticas de dispersión
    """
    print_step("Análisis de Dispersión de la Matriz")

    n_users = df[COL_USER_ID].nunique()
    n_products = df[COL_PRODUCT_ID].nunique()
    n_interactions = len(df)
    total_possible = n_users * n_products
    sparsity = (1 - n_interactions / total_possible) * 100
    density = 100 - sparsity

    print(f"\n  📊 Dimensiones de la Matriz:")
    print(f"    Usuarios: {n_users:,}")
    print(f"    Productos: {n_products:,}")
    print(f"    Celdas totales: {total_possible:,}")

    print(f"\n  📈 Métricas de Dispersión:")
    print(f"    Interacciones reales: {n_interactions:,}")
    print(f"    Celdas vacías: {total_possible - n_interactions:,}")
    print(f"    Sparsity: {sparsity:.4f}%")
    print(f"    Densidad: {density:.4f}%")

    # Análisis de ratings por usuario
    reviews_per_user = df.groupby(COL_USER_ID).size()
    print(f"\n  👥 Reviews por Usuario:")
    print(f"    Media: {reviews_per_user.mean():.2f}")
    print(f"    Mediana: {reviews_per_user.median():.1f}")
    print(f"    P25: {reviews_per_user.quantile(0.25):.1f}")
    print(f"    P75: {reviews_per_user.quantile(0.75):.1f}")

    # Análisis de ratings por producto
    reviews_per_product = df.groupby(COL_PRODUCT_ID).size()
    print(f"\n  👕 Reviews por Producto:")
    print(f"    Media: {reviews_per_product.mean():.2f}")
    print(f"    Mediana: {reviews_per_product.median():.1f}")
    print(f"    P25: {reviews_per_product.quantile(0.25):.1f}")
    print(f"    P75: {reviews_per_product.quantile(0.75):.1f}")

    # Interpretación
    print(f"\n  💡 Interpretación:")
    if sparsity > 99.5:
        print(f"    ⚠️  Matriz MUY dispersa ({sparsity:.2f}%)")
        print(f"    → Filtrado agresivo recomendado antes de CF")
    elif sparsity > 99:
        print(f"    ℹ️  Matriz dispersa ({sparsity:.2f}%)")
        print(f"    → Filtrado moderado recomendado")
    else:
        print(f"    ✅ Densidad aceptable ({density:.2f}%)")

    stats = {
        'n_users': n_users,
        'n_products': n_products,
        'n_interactions': n_interactions,
        'total_possible': total_possible,
        'sparsity': sparsity,
        'density': density,
        'reviews_per_user': reviews_per_user,
        'reviews_per_product': reviews_per_product
    }

    return stats


def plot_sparsity(rating_matrix, df):
    """
    Genera visualización de la dispersión de la matriz.
    Guarda en reports/sparsity_analysis.png

    Args:
        rating_matrix: DataFrame (usuarios x productos)
        df: DataFrame con las reviews
    """
    print_step("Generando gráfico de análisis de sparsity")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    reviews_per_user = df.groupby(COL_USER_ID).size()
    reviews_per_product = df.groupby(COL_PRODUCT_ID).size()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Histograma de reviews por usuario
    ax1 = axes[0, 0]
    reviews_per_user.hist(bins=30, ax=ax1, color='coral', edgecolor='black', alpha=0.7)
    ax1.axvline(reviews_per_user.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {reviews_per_user.mean():.1f}')
    ax1.set_xlabel('Reviews por Usuario')
    ax1.set_ylabel('Cantidad de Usuarios')
    ax1.set_title('Distribución de Reviews por Usuario', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Histograma de reviews por producto
    ax2 = axes[0, 1]
    reviews_per_product.hist(bins=30, ax=ax2, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(reviews_per_product.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {reviews_per_product.mean():.1f}')
    ax2.set_xlabel('Reviews por Producto')
    ax2.set_ylabel('Cantidad de Productos')
    ax2.set_title('Distribución de Reviews por Producto', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Muestra de la matriz (heatmap)
    ax3 = axes[1, 0]
    sample_size = min(20, rating_matrix.shape[0])
    sample_cols = min(30, rating_matrix.shape[1])
    matrix_sample = rating_matrix.iloc[:sample_size, :sample_cols]
    sns.heatmap(matrix_sample, cmap='YlOrRd', ax=ax3, cbar_kws={'label': 'Rating'})
    ax3.set_title(f'Muestra de la Matriz ({sample_size}x{sample_cols})', fontweight='bold')
    ax3.set_xlabel('Productos')
    ax3.set_ylabel('Usuarios')

    # 4. Gráfico de densidad vs dispersión
    ax4 = axes[1, 1]
    sparsity = calculate_sparsity(rating_matrix)
    density = 100 - sparsity
    labels = ['Datos\n(Densidad)', 'Vacío\n(Sparsity)']
    sizes = [density, sparsity]
    colors_pie = ['#06A77D', '#D62828']
    explode = (0.05, 0)
    ax4.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%',
            colors=colors_pie, startangle=90, textprops={'fontsize': 11})
    ax4.set_title('Densidad vs Dispersión de la Matriz', fontweight='bold')

    plt.tight_layout()
    plt.savefig(REPORT_SPARSITY, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {REPORT_SPARSITY.name}")


def run_sparsity_analysis(df, rating_matrix):
    """
    Ejecuta el análisis completo de dispersión.

    Args:
        df: DataFrame con las reviews
        rating_matrix: DataFrame (usuarios x productos)

    Returns:
        dict: Resultados del análisis
    """
    print_header("🕳️  ANÁLISIS DE DISPERSIÓN (SPARSITY)")

    stats = analyze_sparsity(df)
    plot_sparsity(rating_matrix, df)

    print(f"\n  ✅ Análisis de sparsity completado")

    return stats


if __name__ == '__main__':
    from data_loader import load_data, prepare_data, get_user_item_matrix

    df = load_data()
    df_clean = prepare_data(df)
    rating_matrix = get_user_item_matrix(df_clean)
    results = run_sparsity_analysis(df_clean, rating_matrix)
