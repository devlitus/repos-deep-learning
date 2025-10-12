"""
Análisis Exploratorio del Dataset MovieLens 100K
Paso 1: Cargar y entender los datos
"""
import sys
import io
from pathlib import Path

# Configurar codificación UTF-8 para la salida en consola (Windows)
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Obtener la ruta base del proyecto (directorio amazone)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw' / 'ml-100k'
SRC_DIR = PROJECT_DIR / 'src'

# Crear directorios si no existen
SRC_DIR.mkdir(parents=True, exist_ok=True)

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# =============================================================================
# PASO 1: CARGAR LOS DATOS
# =============================================================================

print("=" * 60)
print("📊 ANÁLISIS EXPLORATORIO - MOVIELENS 100K")
print("=" * 60)

# Cargar ratings (calificaciones)
# Formato: user_id | item_id | rating | timestamp
ratings = pd.read_csv(
    DATA_DIR / 'u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

# Cargar información de películas
# Formato: movie_id | title | release_date | ... | géneros
movies = pd.read_csv(
    DATA_DIR / 'u.item',
    sep='|',
    encoding='latin-1',
    names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

# Cargar información de usuarios
users = pd.read_csv(
    DATA_DIR / 'u.user',
    sep='|',
    names=['user_id', 'age', 'gender', 'occupation', 'zip_code']
)

print("\n✅ Datos cargados exitosamente!")

# =============================================================================
# PASO 2: EXPLORACIÓN BÁSICA
# =============================================================================

print("\n" + "=" * 60)
print("📋 INFORMACIÓN BÁSICA DEL DATASET")
print("=" * 60)

print(f"\n🎬 Películas: {movies.shape[0]:,}")
print(f"👥 Usuarios: {users.shape[0]:,}")
print(f"⭐ Ratings totales: {ratings.shape[0]:,}")

print("\n📊 PRIMERAS FILAS DE RATINGS:")
print(ratings.head())

print("\n📊 INFORMACIÓN DE RATINGS:")
print(ratings.info())

print("\n📈 ESTADÍSTICAS DE RATINGS:")
print(ratings.describe())

# =============================================================================
# PASO 3: ANÁLISIS DE RATINGS
# =============================================================================

print("\n" + "=" * 60)
print("⭐ ANÁLISIS DE CALIFICACIONES")
print("=" * 60)

# Distribución de ratings
rating_counts = ratings['rating'].value_counts().sort_index()
print("\n📊 Distribución de calificaciones:")
for rating, count in rating_counts.items():
    percentage = (count / len(ratings)) * 100
    print(f"Rating {rating}: {count:,} ({percentage:.1f}%)")

# Rating promedio
avg_rating = ratings['rating'].mean()
print(f"\n📊 Rating promedio: {avg_rating:.2f}")

# =============================================================================
# PASO 4: ANÁLISIS DE USUARIOS
# =============================================================================

print("\n" + "=" * 60)
print("👥 ANÁLISIS DE USUARIOS")
print("=" * 60)

# Ratings por usuario
ratings_per_user = ratings.groupby('user_id').size()
print(f"\n📊 Ratings por usuario:")
print(f"  - Promedio: {ratings_per_user.mean():.1f}")
print(f"  - Mínimo: {ratings_per_user.min()}")
print(f"  - Máximo: {ratings_per_user.max()}")
print(f"  - Mediana: {ratings_per_user.median():.1f}")

# Usuario más activo
most_active_user = ratings_per_user.idxmax()
print(f"\n🏆 Usuario más activo: {most_active_user} con {ratings_per_user.max()} ratings")

# =============================================================================
# PASO 5: ANÁLISIS DE PELÍCULAS
# =============================================================================

print("\n" + "=" * 60)
print("🎬 ANÁLISIS DE PELÍCULAS")
print("=" * 60)

# Ratings por película
ratings_per_movie = ratings.groupby('item_id').size()
print(f"\n📊 Ratings por película:")
print(f"  - Promedio: {ratings_per_movie.mean():.1f}")
print(f"  - Mínimo: {ratings_per_movie.min()}")
print(f"  - Máximo: {ratings_per_movie.max()}")
print(f"  - Mediana: {ratings_per_movie.median():.1f}")

# Películas más calificadas
most_rated_movies = ratings.groupby('item_id').size().sort_values(ascending=False).head(10)
print(f"\n🏆 TOP 10 PELÍCULAS MÁS CALIFICADAS:")
for idx, (movie_id, count) in enumerate(most_rated_movies.items(), 1):
    movie_title = movies[movies['item_id'] == movie_id]['title'].values[0]
    print(f"{idx:2d}. {movie_title}: {count} ratings")

# =============================================================================
# PASO 6: VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 60)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 60)

# Crear figura con subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Análisis Exploratorio - MovieLens 100K', fontsize=16, fontweight='bold')

# 1. Distribución de ratings
axes[0, 0].bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frecuencia')
axes[0, 0].set_title('Distribución de Calificaciones')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Distribución de ratings por usuario
axes[0, 1].hist(ratings_per_user, bins=50, color='coral', edgecolor='black')
axes[0, 1].set_xlabel('Número de ratings')
axes[0, 1].set_ylabel('Número de usuarios')
axes[0, 1].set_title('Distribución de Ratings por Usuario')
axes[0, 1].axvline(ratings_per_user.mean(), color='red', linestyle='--', label=f'Media: {ratings_per_user.mean():.1f}')
axes[0, 1].legend()

# 3. Distribución de ratings por película
axes[1, 0].hist(ratings_per_movie, bins=50, color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('Número de ratings')
axes[1, 0].set_ylabel('Número de películas')
axes[1, 0].set_title('Distribución de Ratings por Película')
axes[1, 0].axvline(ratings_per_movie.mean(), color='red', linestyle='--', label=f'Media: {ratings_per_movie.mean():.1f}')
axes[1, 0].legend()

# 4. Rating promedio por película (top 20 más calificadas)
top_20_movies = most_rated_movies.head(20).index
top_20_avg_rating = ratings[ratings['item_id'].isin(top_20_movies)].groupby('item_id')['rating'].mean().sort_values(ascending=False)
axes[1, 1].barh(range(len(top_20_avg_rating)), top_20_avg_rating.values, color='plum')
axes[1, 1].set_yticks(range(len(top_20_avg_rating)))
axes[1, 1].set_yticklabels([movies[movies['item_id'] == idx]['title'].values[0][:30] for idx in top_20_avg_rating.index], fontsize=8)
axes[1, 1].set_xlabel('Rating Promedio')
axes[1, 1].set_title('Rating Promedio - Top 20 Películas Más Calificadas')
axes[1, 1].invert_yaxis()

plt.tight_layout()
output_file = SRC_DIR / 'exploratory_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Visualización guardada en: {output_file}")

plt.show()

print("\n" + "=" * 60)
print("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
print("=" * 60)
print("\n📝 CONCLUSIONES INICIALES:")
print(f"1. Tenemos {ratings.shape[0]:,} ratings de {users.shape[0]} usuarios sobre {movies.shape[0]:,} películas")
print(f"2. Rating promedio: {avg_rating:.2f}")
print(f"3. Los usuarios califican en promedio {ratings_per_user.mean():.1f} películas")
print(f"4. Las películas reciben en promedio {ratings_per_movie.mean():.1f} calificaciones")
print("\n🎯 Siguiente paso: Entender la matriz de ratings y sparsity")