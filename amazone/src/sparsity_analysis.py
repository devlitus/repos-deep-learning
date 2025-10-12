"""
Análisis de Sparsity (Dispersión) del Dataset MovieLens 100K
Paso 2: Entender la matriz de ratings y el problema de datos faltantes
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("=" * 70)
print("🔍 ANÁLISIS DE SPARSITY - MOVIELENS 100K")
print("=" * 70)

# =============================================================================
# PASO 1: CARGAR DATOS
# =============================================================================

ratings = pd.read_csv(
    'data/raw/ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

print("\n✅ Datos cargados")

# =============================================================================
# PASO 2: CREAR LA MATRIZ DE RATINGS
# =============================================================================

print("\n" + "=" * 70)
print("📊 CREANDO MATRIZ DE RATINGS (USER-ITEM)")
print("=" * 70)

# Crear matriz user-item usando pivot
# Filas = usuarios, Columnas = películas, Valores = ratings
ratings_matrix = ratings.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
)

print(f"\n📐 Dimensiones de la matriz:")
print(f"   - Usuarios (filas): {ratings_matrix.shape[0]}")
print(f"   - Películas (columnas): {ratings_matrix.shape[1]}")
print(f"   - Total de celdas: {ratings_matrix.shape[0] * ratings_matrix.shape[1]:,}")

print(f"\n📊 Muestra de la matriz (primeros 5 usuarios x 5 películas):")
print(ratings_matrix.iloc[:5, :5])
print("\nNota: NaN significa que el usuario NO ha calificado esa película")

# =============================================================================
# PASO 3: CALCULAR SPARSITY
# =============================================================================

print("\n" + "=" * 70)
print("🕳️  CÁLCULO DE SPARSITY")
print("=" * 70)

# Total de celdas en la matriz
total_cells = ratings_matrix.shape[0] * ratings_matrix.shape[1]

# Celdas con ratings (no NaN)
filled_cells = ratings_matrix.notna().sum().sum()

# Celdas vacías (NaN)
empty_cells = ratings_matrix.isna().sum().sum()

# Porcentaje de sparsity
sparsity = (empty_cells / total_cells) * 100

print(f"\n📊 Estadísticas de la matriz:")
print(f"   - Total de celdas posibles: {total_cells:,}")
print(f"   - Celdas con ratings: {filled_cells:,} ({(filled_cells/total_cells)*100:.2f}%)")
print(f"   - Celdas vacías (sin rating): {empty_cells:,} ({sparsity:.2f}%)")
print(f"\n🎯 SPARSITY: {sparsity:.2f}%")

print(f"\n💡 Interpretación:")
print(f"   Solo el {(filled_cells/total_cells)*100:.2f}% de las interacciones posibles existen.")
print(f"   La matriz está {sparsity:.2f}% vacía.")
print(f"   Esto significa que cada usuario solo ha visto ~{(filled_cells/ratings_matrix.shape[0]):.0f} películas")
print(f"   de las {ratings_matrix.shape[1]:,} disponibles.")

# =============================================================================
# PASO 4: VISUALIZAR SPARSITY
# =============================================================================

print("\n" + "=" * 70)
print("📊 GENERANDO VISUALIZACIONES DE SPARSITY")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Análisis de Sparsity - MovieLens 100K', fontsize=16, fontweight='bold')

# 1. Visualización de la matriz (muestra de 50x50)
ax1 = axes[0, 0]
sample_matrix = ratings_matrix.iloc[:50, :50].notna().astype(int)
im = ax1.imshow(sample_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
ax1.set_xlabel('Películas (IDs)')
ax1.set_ylabel('Usuarios (IDs)')
ax1.set_title('Matriz de Ratings (50x50 muestra)\nVerde=Rating existe, Rojo=Sin rating')
plt.colorbar(im, ax=ax1, label='1=Rating, 0=Vacío')

# 2. Distribución de ratings por usuario
ax2 = axes[0, 1]
ratings_per_user = ratings_matrix.notna().sum(axis=1)
ax2.hist(ratings_per_user, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(ratings_per_user.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {ratings_per_user.mean():.1f}')
ax2.axvline(ratings_per_user.median(), color='orange', linestyle='--', 
            linewidth=2, label=f'Mediana: {ratings_per_user.median():.1f}')
ax2.set_xlabel('Número de películas calificadas')
ax2.set_ylabel('Número de usuarios')
ax2.set_title('Distribución de Actividad por Usuario')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Distribución de ratings por película
ax3 = axes[1, 0]
ratings_per_movie = ratings_matrix.notna().sum(axis=0)
ax3.hist(ratings_per_movie, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax3.axvline(ratings_per_movie.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {ratings_per_movie.mean():.1f}')
ax3.axvline(ratings_per_movie.median(), color='orange', linestyle='--', 
            linewidth=2, label=f'Mediana: {ratings_per_movie.median():.1f}')
ax3.set_xlabel('Número de usuarios que calificaron')
ax3.set_ylabel('Número de películas')
ax3.set_title('Distribución de Popularidad por Película')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Gráfico de sparsity general
ax4 = axes[1, 1]
categories = ['Ratings\nExistentes', 'Celdas\nVacías']
values = [filled_cells, empty_cells]
colors = ['#2ecc71', '#e74c3c']
bars = ax4.bar(categories, values, color=colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('Número de celdas')
ax4.set_title(f'Sparsity General: {sparsity:.2f}%')
ax4.set_ylim(0, total_cells * 1.1)

# Añadir etiquetas con porcentajes
for bar, value in zip(bars, values):
    height = bar.get_height()
    percentage = (value / total_cells) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:,}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()

# Guardar visualización
import os
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/sparsity_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización guardada en: reports/sparsity_analysis.png")

plt.show()

# =============================================================================
# PASO 5: ANÁLISIS DE USUARIOS Y PELÍCULAS CON POCOS DATOS
# =============================================================================

print("\n" + "=" * 70)
print("⚠️  ANÁLISIS DE DATOS ESCASOS")
print("=" * 70)

# Usuarios con pocos ratings
users_low_ratings = (ratings_per_user < 20).sum()
print(f"\n👥 Usuarios con menos de 20 ratings: {users_low_ratings} ({(users_low_ratings/len(ratings_per_user))*100:.1f}%)")

# Películas con pocos ratings
movies_low_ratings = (ratings_per_movie < 5).sum()
print(f"🎬 Películas con menos de 5 ratings: {movies_low_ratings} ({(movies_low_ratings/len(ratings_per_movie))*100:.1f}%)")

print(f"\n💡 Implicación:")
print(f"   - Los usuarios con pocos ratings son difíciles de recomendar")
print(f"   - Las películas con pocos ratings tienen poca información para ser recomendadas")
print(f"   - Este es el PROBLEMA PRINCIPAL que los sistemas de recomendación deben resolver")

# =============================================================================
# CONCLUSIONES
# =============================================================================

print("\n" + "=" * 70)
print("✅ ANÁLISIS DE SPARSITY COMPLETADO")
print("=" * 70)

print("\n📝 CONCLUSIONES:")
print(f"1. La matriz de ratings tiene {sparsity:.2f}% de sparsity (muy dispersa)")
print(f"2. Cada usuario solo califica ~{(filled_cells/ratings_matrix.shape[0]):.0f} de {ratings_matrix.shape[1]:,} películas")
print(f"3. Hay usuarios y películas con muy pocos datos")
print(f"4. Los sistemas de recomendación deben 'predecir' las celdas vacías")

print("\n🎯 SIGUIENTE PASO:")
print("   Ahora que entendemos el problema de sparsity, vamos a:")
print("   1. Aprender qué es Collaborative Filtering")
print("   2. Implementar un sistema User-Based")
print("   3. Ver cómo predicen las celdas vacías")

print("\n💡 CONCEPTO CLAVE - SPARSITY:")
print("   Sparsity = Porcentaje de datos faltantes en la matriz")
print("   Propósito: Medir qué tan vacía está nuestra matriz de ratings")
print("   Por qué importa: A mayor sparsity, más difícil es hacer buenas recomendaciones")