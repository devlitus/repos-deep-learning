"""
Módulo de Análisis Exploratorio de Datos (EDA)
Analiza distribuciones de ratings, usuarios y productos del dataset de Fashion Reviews
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
    REPORTS_DIR, REPORT_RATINGS_DIST, REPORT_TOP_USERS, REPORT_TOP_PRODUCTS,
    PLOT_STYLE, PLOT_DPI
)

warnings.filterwarnings('ignore')


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(text):
    print(f"\n{'─' * 80}")
    print(f"📍 {text}")
    print("─" * 80)


def analyze_ratings(df):
    """
    Analiza la distribución de ratings del dataset.

    Args:
        df: DataFrame con las reviews

    Returns:
        dict: Estadísticas de ratings
    """
    print_step("Análisis de Distribución de Ratings")

    rating_counts = df[COL_RATING].value_counts().sort_index()
    rating_percentages = (rating_counts / len(df) * 100).round(2)

    print("\n⭐ Distribución de Calificaciones:")
    print("─" * 60)
    for rating in sorted(df[COL_RATING].unique()):
        count = rating_counts.get(rating, 0)
        percentage = rating_percentages.get(rating, 0)
        bar = '█' * int(percentage / 2)
        print(f"  {rating:.0f} estrellas: {count:>6,} ({percentage:>6.2f}%) {bar}")

    stats = {
        'mean': df[COL_RATING].mean(),
        'median': df[COL_RATING].median(),
        'std': df[COL_RATING].std(),
        'min': df[COL_RATING].min(),
        'max': df[COL_RATING].max(),
        'counts': rating_counts.to_dict()
    }

    print(f"\n  Rating promedio: {stats['mean']:.2f}/5.0")
    print(f"  Rating mediano: {stats['median']:.1f}/5.0")
    print(f"  Desviación estándar: {stats['std']:.2f}")

    return stats


def analyze_users(df):
    """
    Analiza la actividad de los usuarios.

    Args:
        df: DataFrame con las reviews

    Returns:
        dict: Estadísticas de usuarios
    """
    print_step("Análisis de Actividad de Usuarios")

    reviews_per_user = df.groupby(COL_USER_ID).size()
    n_users = df[COL_USER_ID].nunique()

    print(f"\n  👥 Usuarios únicos: {n_users:,}")
    print(f"\n  Reviews por usuario:")
    print(f"    Min: {reviews_per_user.min()} reviews")
    print(f"    Max: {reviews_per_user.max()} reviews")
    print(f"    Media: {reviews_per_user.mean():.2f} reviews")
    print(f"    Mediana: {reviews_per_user.median():.1f} reviews")
    print(f"    P75: {reviews_per_user.quantile(0.75):.1f}")
    print(f"    P90: {reviews_per_user.quantile(0.90):.1f}")

    # Top 10 usuarios más activos
    top_users = reviews_per_user.nlargest(10)
    print(f"\n  🏆 Top 10 Usuarios Más Activos:")
    for i, (user, count) in enumerate(top_users.items(), 1):
        print(f"    {i:2d}. Usuario {user}: {count} reviews")

    stats = {
        'n_users': n_users,
        'reviews_per_user': reviews_per_user,
        'mean': reviews_per_user.mean(),
        'median': reviews_per_user.median(),
        'min': reviews_per_user.min(),
        'max': reviews_per_user.max(),
        'top_users': top_users
    }

    return stats


def analyze_products(df):
    """
    Analiza la popularidad de los productos.

    Args:
        df: DataFrame con las reviews

    Returns:
        dict: Estadísticas de productos
    """
    print_step("Análisis de Popularidad de Productos")

    reviews_per_product = df.groupby(COL_PRODUCT_ID).size()
    rating_per_product = df.groupby(COL_PRODUCT_ID)[COL_RATING].mean()
    n_products = df[COL_PRODUCT_ID].nunique()

    print(f"\n  👕 Productos únicos: {n_products:,}")
    print(f"\n  Reviews por producto:")
    print(f"    Min: {reviews_per_product.min()} reviews")
    print(f"    Max: {reviews_per_product.max()} reviews")
    print(f"    Media: {reviews_per_product.mean():.2f} reviews")
    print(f"    Mediana: {reviews_per_product.median():.1f} reviews")

    # Top 10 productos más reseñados
    top_products = reviews_per_product.nlargest(10)
    print(f"\n  🏆 Top 10 Productos Más Reseñados:")
    for i, (product, count) in enumerate(top_products.items(), 1):
        avg_rating = rating_per_product[product]
        print(f"    {i:2d}. Producto {product}: {count:2d} reviews (Rating: {avg_rating:.2f}⭐)")

    stats = {
        'n_products': n_products,
        'reviews_per_product': reviews_per_product,
        'rating_per_product': rating_per_product,
        'mean': reviews_per_product.mean(),
        'median': reviews_per_product.median(),
        'min': reviews_per_product.min(),
        'max': reviews_per_product.max(),
        'top_products': top_products
    }

    return stats


def plot_rating_distribution(df):
    """
    Genera gráfico de distribución de ratings (bar + pie chart).
    Guarda en reports/rating_distribution.png

    Args:
        df: DataFrame con las reviews
    """
    print_step("Generando gráfico de distribución de ratings")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    rating_counts = df[COL_RATING].value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de barras
    ax1 = axes[0]
    rating_counts.plot(kind='bar', ax=ax1, color='steelblue', edgecolor='black')
    ax1.set_title('Número de Reviews por Calificación', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Calificación (estrellas)')
    ax1.set_ylabel('Cantidad de Reviews')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

    for i, v in enumerate(rating_counts):
        ax1.text(i, v + 50, str(v), ha='center', fontweight='bold')

    # Gráfico de pastel
    ax2 = axes[1]
    colors = ['#FF6B6B', '#FFA06B', '#FFD93D', '#A8D93D', '#6BCA6B']
    ax2.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%',
            colors=colors[:len(rating_counts)], startangle=90,
            wedgeprops={'edgecolor': 'black'})
    ax2.set_title('Porcentaje de Reviews por Calificación', fontsize=12, fontweight='bold')

    plt.tight_layout()

    output_path = REPORT_RATINGS_DIST
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {output_path.name}")


def plot_user_distribution(df):
    """
    Genera gráfico de distribución de reviews por usuario (histogram + boxplot).
    Guarda en reports/top_users.png

    Args:
        df: DataFrame con las reviews
    """
    print_step("Generando gráfico de distribución de usuarios")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    reviews_per_user = df.groupby(COL_USER_ID).size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    ax1 = axes[0]
    reviews_per_user.hist(bins=20, ax=ax1, color='coral', edgecolor='black')
    ax1.axvline(reviews_per_user.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {reviews_per_user.mean():.2f}')
    ax1.axvline(reviews_per_user.median(), color='green', linestyle='--', linewidth=2,
                label=f'Mediana: {reviews_per_user.median():.1f}')
    ax1.set_xlabel('Reviews por Usuario')
    ax1.set_ylabel('Cantidad de Usuarios')
    ax1.set_title('Distribución: Reviews por Usuario', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Box plot
    ax2 = axes[1]
    ax2.boxplot(reviews_per_user, vert=True)
    ax2.set_ylabel('Reviews por Usuario')
    ax2.set_title('Box Plot: Reviews por Usuario', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_xticklabels(['Reviews/Usuario'])

    plt.tight_layout()

    output_path = REPORT_TOP_USERS
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {output_path.name}")


def plot_product_distribution(df):
    """
    Genera gráfico de distribución de reviews por producto (histogram + top-15 bar).
    Guarda en reports/top_products.png

    Args:
        df: DataFrame con las reviews
    """
    print_step("Generando gráfico de distribución de productos")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    reviews_per_product = df.groupby(COL_PRODUCT_ID).size()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    ax1 = axes[0]
    reviews_per_product.hist(bins=30, ax=ax1, color='skyblue', edgecolor='black')
    ax1.axvline(reviews_per_product.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {reviews_per_product.mean():.2f}')
    ax1.axvline(reviews_per_product.median(), color='green', linestyle='--', linewidth=2,
                label=f'Mediana: {reviews_per_product.median():.1f}')
    ax1.set_xlabel('Reviews por Producto')
    ax1.set_ylabel('Cantidad de Productos')
    ax1.set_title('Distribución: Reviews por Producto', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Top 15 productos
    ax2 = axes[1]
    top_15 = reviews_per_product.nlargest(15)
    top_15.plot(kind='barh', ax=ax2, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Cantidad de Reviews')
    ax2.set_title('Top 15 Productos Más Reseñados', fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = REPORT_TOP_PRODUCTS
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {output_path.name}")


def run_full_eda(df):
    """
    Ejecuta el análisis exploratorio completo.

    Args:
        df: DataFrame con las reviews

    Returns:
        dict: Resultados de todos los análisis
    """
    print_header("📊 ANÁLISIS EXPLORATORIO DE DATOS (EDA)")

    results = {}

    # Análisis estadístico
    results['ratings'] = analyze_ratings(df)
    results['users'] = analyze_users(df)
    results['products'] = analyze_products(df)

    # Generar gráficos
    plot_rating_distribution(df)
    plot_user_distribution(df)
    plot_product_distribution(df)

    # Resumen
    print_step("Resumen del EDA")
    n_reviews = len(df)
    n_users = results['users']['n_users']
    n_products = results['products']['n_products']
    sparsity = (1 - n_reviews / (n_users * n_products)) * 100

    print(f"\n  📌 INSIGHTS CLAVE:")
    print(f"\n  1️⃣  DATOS DISPERSOS: {sparsity:.1f}% de la matriz está vacía")
    print(f"  2️⃣  USUARIOS: Promedio {results['users']['mean']:.1f} reviews/usuario")
    print(f"  3️⃣  PRODUCTOS: Promedio {results['products']['mean']:.1f} reviews/producto")
    print(f"  4️⃣  RATINGS: Promedio {results['ratings']['mean']:.2f}/5.0")

    print(f"\n  ✅ EDA completado - {3} gráficos generados en reports/")

    return results


if __name__ == '__main__':
    from data_loader import load_data

    df = load_data()
    results = run_full_eda(df)
