"""
Análisis Exploratorio del Dataset Amazon Fashion Reviews
Paso 1: Cargar y entender los datos de reviews de ropa
"""
import sys
import io
import json
from pathlib import Path

# Configurar codificación UTF-8 para la salida en consola (Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener la ruta base del proyecto (directorio recommendation-fashion)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw'
REPORTS_DIR = PROJECT_DIR / 'reports'
SRC_DIR = PROJECT_DIR / 'src'

# Crear directorios si no existen
SRC_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# PASO 1: CARGAR LOS DATOS
# =============================================================================

print("=" * 60)
print("📊 ANÁLISIS EXPLORATORIO - AMAZON FASHION REVIEWS")
print("=" * 60)

json_file = DATA_DIR / 'fashion_reviews.json'

# Verificar que el archivo existe
if not json_file.exists():
    print(f"\n❌ Error: No se encontró {json_file}")
    print(f"Por favor, ejecuta primero: python download_fashion.py")
    sys.exit(1)

print("\n📥 Cargando datos JSON...")
print(f"Archivo: {json_file}")

# Cargar datos JSON (línea por línea)
reviews = []
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError:
                continue

    print(f"✅ {len(reviews):,} reviews cargadas exitosamente!")

    # Convertir a DataFrame
    df = pd.DataFrame(reviews)

    # Seleccionar columnas relevantes
    relevant_cols = ['reviewerID', 'asin', 'overall', 'reviewText', 'summary', 'unixReviewTime']
    df = df[[col for col in relevant_cols if col in df.columns]]

    # Renombrar columnas para consistencia
    df.columns = ['user_id', 'product_id', 'rating', 'review_text', 'summary', 'timestamp']

except Exception as e:
    print(f"\n❌ Error cargando JSON: {e}")
    print("Por favor verifica que el archivo JSON esté bien formado")
    sys.exit(1)

# =============================================================================
# PASO 2: EXPLORACIÓN BÁSICA
# =============================================================================

print("\n" + "=" * 60)
print("📋 INFORMACIÓN BÁSICA DEL DATASET")
print("=" * 60)

n_reviews = len(df)
n_unique_users = df['user_id'].nunique()
n_unique_products = df['product_id'].nunique()
sparsity = 1 - (n_reviews / (n_unique_users * n_unique_products))

print(f"\n👕 Productos únicos: {n_unique_products:,}")
print(f"👥 Usuarios únicos: {n_unique_users:,}")
print(f"⭐ Reviews totales: {n_reviews:,}")
print(f"📊 Dispersión de matriz: {sparsity*100:.2f}%")

print("\n📊 PRIMERAS FILAS:")
print(df[['user_id', 'product_id', 'rating', 'summary']].head(10))

print("\n📊 INFORMACIÓN DEL DATASET:")
print(df.info())

print("\n📈 ESTADÍSTICAS DE RATINGS:")
print(df['rating'].describe())

# =============================================================================
# PASO 3: ANÁLISIS DE RATINGS
# =============================================================================

print("\n" + "=" * 60)
print("⭐ ANÁLISIS DE CALIFICACIONES")
print("=" * 60)

# Distribución de ratings
rating_counts = df['rating'].value_counts().sort_index()
print("\n📊 Distribución de calificaciones:")
for rating, count in rating_counts.items():
    percentage = (count / len(df)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {int(rating)} estrellas: {count:>7,} ({percentage:5.2f}%) {bar}")

# Rating promedio
avg_rating = df['rating'].mean()
print(f"\n⭐ Rating promedio: {avg_rating:.2f}/5.0")

# =============================================================================
# PASO 4: ANÁLISIS POR USUARIO Y PRODUCTO
# =============================================================================

print("\n" + "=" * 60)
print("📊 ANÁLISIS POR USUARIO Y PRODUCTO")
print("=" * 60)

user_counts = df['user_id'].value_counts()
product_counts = df['product_id'].value_counts()

print(f"\n👥 USUARIOS:")
print(f"  - Mínimo de reviews por usuario: {user_counts.min()}")
print(f"  - Promedio de reviews por usuario: {user_counts.mean():.2f}")
print(f"  - Máximo de reviews por usuario: {user_counts.max()}")
print(f"  - Mediana: {user_counts.median():.0f}")

print(f"\n👕 PRODUCTOS:")
print(f"  - Mínimo de reviews por producto: {product_counts.min()}")
print(f"  - Promedio de reviews por producto: {product_counts.mean():.2f}")
print(f"  - Máximo de reviews por producto: {product_counts.max()}")
print(f"  - Mediana: {product_counts.median():.0f}")

# Top 10 productos más reseñados
print(f"\n🏆 TOP 10 PRODUCTOS MÁS RESEÑADOS:")
top_products = product_counts.head(10)
for idx, (prod_id, count) in enumerate(top_products.items(), 1):
    avg_rating = df[df['product_id'] == prod_id]['rating'].mean()
    print(f"  {idx}. Producto {prod_id}: {count:,} reviews (rating promedio: {avg_rating:.2f})")

# Top 10 usuarios más activos
print(f"\n👤 TOP 10 USUARIOS MÁS ACTIVOS:")
top_users = user_counts.head(10)
for idx, (user_id, count) in enumerate(top_users.items(), 1):
    print(f"  {idx}. Usuario {user_id}: {count:,} reviews")

# =============================================================================
# PASO 5: VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 60)
print("📈 GENERANDO VISUALIZACIONES")
print("=" * 60)

# 1. Distribución de ratings
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
rating_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
axes[0].set_title('Distribución de Calificaciones', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Rating (estrellas)', fontsize=12)
axes[0].set_ylabel('Número de Reviews', fontsize=12)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(True, alpha=0.3)

# Gráfico de pastel
colors = ['#d62728', '#ff7f0e', '#ffbb78', '#2ca02c', '#1f77b4']
axes[1].pie(rating_counts.values, labels=rating_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
axes[1].set_title('Proporción de Calificaciones', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'rating_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: rating_distribution.png")
plt.close()

# 2. Distribución de reviews por usuario
fig, ax = plt.subplots(figsize=(12, 6))
user_counts.head(30).plot(kind='barh', ax=ax, color='coral', edgecolor='black')
ax.set_title('Top 30 Usuarios más Activos', fontsize=14, fontweight='bold')
ax.set_xlabel('Número de Reviews', fontsize=12)
ax.set_ylabel('Usuario', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'top_users.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: top_users.png")
plt.close()

# 3. Distribución de reviews por producto
fig, ax = plt.subplots(figsize=(12, 6))
product_counts.head(30).plot(kind='barh', ax=ax, color='lightgreen', edgecolor='black')
ax.set_title('Top 30 Productos más Reseñados', fontsize=14, fontweight='bold')
ax.set_xlabel('Número de Reviews', fontsize=12)
ax.set_ylabel('Producto ID', fontsize=12)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'top_products.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: top_products.png")
plt.close()

# 4. Histogramas de distribución
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Reviews por usuario
axes[0].hist(user_counts.values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0].set_title('Distribución: Reviews por Usuario', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Número de Reviews', fontsize=12)
axes[0].set_ylabel('Frecuencia', fontsize=12)
axes[0].set_xscale('log')
axes[0].grid(True, alpha=0.3)

# Reviews por producto
axes[1].hist(product_counts.values, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[1].set_title('Distribución: Reviews por Producto', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Número de Reviews', fontsize=12)
axes[1].set_ylabel('Frecuencia', fontsize=12)
axes[1].set_xscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'distribution_histogram.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: distribution_histogram.png")
plt.close()

# =============================================================================
# PASO 6: RESUMEN Y ESTADÍSTICAS
# =============================================================================

print("\n" + "=" * 60)
print("📌 RESUMEN ESTADÍSTICO FINAL")
print("=" * 60)

summary_stats = {
    'Total de Reviews': f"{n_reviews:,}",
    'Usuarios Únicos': f"{n_unique_users:,}",
    'Productos Únicos': f"{n_unique_products:,}",
    'Dispersión de Matriz': f"{sparsity*100:.2f}%",
    'Rating Promedio': f"{avg_rating:.2f}",
    'Rating Mínimo': f"{df['rating'].min():.1f}",
    'Rating Máximo': f"{df['rating'].max():.1f}",
    'Reviews/Usuario (promedio)': f"{user_counts.mean():.2f}",
    'Reviews/Producto (promedio)': f"{product_counts.mean():.2f}",
}

for key, value in summary_stats.items():
    print(f"{key:.<40} {value}")

print("\n" + "=" * 60)
print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
print("=" * 60)
print("\nPróximos pasos:")
print("1. Ejecutar: python src/user_based_collaborative_filtering.py")
print("2. Ejecutar: python src/item_based_collaborative_filtering.py")
print("3. Ejecutar: python src/matrix_factorization_svd.py")
print("4. Ver resultados en la carpeta reports/")
