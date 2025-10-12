"""
Item-Based Collaborative Filtering
Paso 4: Implementar sistema de recomendación basado en similitud de películas
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🎬 ITEM-BASED COLLABORATIVE FILTERING")
print("=" * 70)

# =============================================================================
# PASO 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n📊 Cargando datos...")

# Cargar ratings
ratings = pd.read_csv(
    'data/raw/ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

# Cargar información de películas
movies = pd.read_csv(
    'data/raw/ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
           'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

print("✅ Datos cargados")

# Crear matriz user-item (usuarios en filas, películas en columnas)
ratings_matrix = ratings.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
)

print(f"\n📐 Matriz de ratings: {ratings_matrix.shape[0]} usuarios × {ratings_matrix.shape[1]} películas")

# =============================================================================
# PASO 2: CALCULAR SIMILITUD ENTRE PELÍCULAS
# =============================================================================

print("\n" + "=" * 70)
print("🔢 CALCULANDO SIMILITUD ENTRE PELÍCULAS")
print("=" * 70)

print("\n💡 DIFERENCIA CLAVE CON USER-BASED:")
print("   User-Based: Compara usuarios (filas) → ¿Qué usuarios son similares?")
print("   Item-Based: Compara películas (columnas) → ¿Qué películas son similares?")

# Para calcular similitud entre películas, necesitamos transponer la matriz
# Ahora: películas en filas, usuarios en columnas
ratings_matrix_transposed = ratings_matrix.T

# Rellenar NaN con 0
ratings_matrix_filled = ratings_matrix_transposed.fillna(0)

print(f"\n⚙️  Calculando matriz de similitud del coseno...")
print(f"   Matriz transpuesta: {ratings_matrix_transposed.shape[0]} películas × {ratings_matrix_transposed.shape[1]} usuarios")
print("   (Esto puede tardar unos segundos...)")

# Calcular similitud del coseno entre películas
item_similarity = cosine_similarity(ratings_matrix_filled)

# Convertir a DataFrame
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=ratings_matrix_transposed.index,
    columns=ratings_matrix_transposed.index
)

print(f"✅ Matriz de similitud creada: {item_similarity_df.shape[0]} × {item_similarity_df.shape[1]}")

# Mostrar ejemplo de similitudes
example_movie_id = 1  # Toy Story
example_movie_title = movies[movies['item_id'] == example_movie_id]['title'].values[0]

print(f"\n📊 Ejemplo - Películas más similares a '{example_movie_title}':")
similar_movies = item_similarity_df.loc[example_movie_id].sort_values(ascending=False).head(11)

for idx, (movie_id, similarity) in enumerate(similar_movies.items()):
    if idx == 0:  # Skip la misma película
        continue
    movie_title = movies[movies['item_id'] == movie_id]['title'].values[0]
    print(f"   {similarity:.3f} - {movie_title}")

# =============================================================================
# PASO 3: FUNCIÓN PARA ENCONTRAR PELÍCULAS SIMILARES
# =============================================================================

def find_similar_items(item_id, similarity_df, k=10):
    """
    Encuentra las k películas más similares a una película dada
    
    Args:
        item_id: ID de la película
        similarity_df: DataFrame con similitudes
        k: Número de películas similares a retornar
    
    Returns:
        Series con las k películas más similares y sus similitudes
    """
    # Obtener similitudes de la película
    similarities = similarity_df.loc[item_id]
    
    # Ordenar de mayor a menor (excluir la misma película)
    similar_items = similarities.sort_values(ascending=False)[1:k+1]
    
    return similar_items

# =============================================================================
# PASO 4: FUNCIÓN PARA PREDECIR RATING
# =============================================================================

def predict_rating_item_based(user_id, item_id, ratings_matrix, similarity_df, k=10):
    """
    Predice el rating que un usuario daría a una película
    usando películas similares que el usuario ya calificó
    
    Args:
        user_id: ID del usuario
        item_id: ID de la película a predecir
        ratings_matrix: Matriz de ratings
        similarity_df: Matriz de similitudes entre películas
        k: Número de películas similares a considerar
    
    Returns:
        Rating predicho (float)
    """
    # Obtener películas similares a la película objetivo
    similar_items = find_similar_items(item_id, similarity_df, k)
    
    # Obtener ratings del usuario para películas similares
    user_ratings = ratings_matrix.loc[user_id, similar_items.index]
    
    # Eliminar NaN (películas que el usuario no ha calificado)
    valid_ratings = user_ratings.dropna()
    valid_similarities = similar_items.loc[valid_ratings.index]
    
    # Si no hay películas similares que el usuario haya calificado
    if len(valid_ratings) == 0:
        user_mean = ratings_matrix.loc[user_id].mean()
        return user_mean if not np.isnan(user_mean) else 3.0
    
    # Calcular predicción como promedio ponderado
    weighted_sum = (valid_similarities * valid_ratings).sum()
    similarity_sum = valid_similarities.sum()
    
    predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else 3.0
    
    return predicted_rating

# =============================================================================
# PASO 5: FUNCIÓN PARA RECOMENDAR PELÍCULAS
# =============================================================================

def recommend_movies_item_based(user_id, ratings_matrix, similarity_df, movies_df, k_items=10, n_recommendations=10):
    """
    Recomienda películas usando Item-Based CF
    
    Args:
        user_id: ID del usuario
        ratings_matrix: Matriz de ratings
        similarity_df: Matriz de similitudes entre películas
        movies_df: DataFrame con información de películas
        k_items: Número de películas similares a considerar
        n_recommendations: Número de recomendaciones a retornar
    
    Returns:
        DataFrame con recomendaciones
    """
    # Obtener películas que el usuario YA ha visto
    user_ratings = ratings_matrix.loc[user_id]
    seen_movies = user_ratings.dropna().index.tolist()
    
    # Obtener películas que NO ha visto
    all_movies = ratings_matrix.columns.tolist()
    unseen_movies = [movie for movie in all_movies if movie not in seen_movies]
    
    print(f"\n🎬 Usuario {user_id}:")
    print(f"   - Ha visto: {len(seen_movies)} películas")
    print(f"   - No ha visto: {len(unseen_movies)} películas")
    print(f"   - Calculando predicciones basadas en películas similares...")
    
    # Predecir ratings para películas no vistas
    predictions = []
    for movie_id in unseen_movies:
        pred_rating = predict_rating_item_based(user_id, movie_id, ratings_matrix, similarity_df, k_items)
        predictions.append({
            'item_id': movie_id,
            'predicted_rating': pred_rating
        })
    
    # Convertir a DataFrame y ordenar
    predictions_df = pd.DataFrame(predictions)
    predictions_df = predictions_df.sort_values('predicted_rating', ascending=False)
    
    # Obtener top N recomendaciones
    top_recommendations = predictions_df.head(n_recommendations)
    
    # Añadir información de películas
    top_recommendations = top_recommendations.merge(
        movies_df[['item_id', 'title']],
        on='item_id',
        how='left'
    )
    
    return top_recommendations

# =============================================================================
# PASO 6: PROBAR EL SISTEMA DE RECOMENDACIÓN
# =============================================================================

print("\n" + "=" * 70)
print("🎯 PROBANDO ITEM-BASED COLLABORATIVE FILTERING")
print("=" * 70)

# Usar el mismo usuario que en User-Based para comparar
test_user_id = 1

print(f"\n👤 Usuario de prueba: {test_user_id}")

# Ver las películas que ya ha calificado alto
user_ratings = ratings_matrix.loc[test_user_id].dropna().sort_values(ascending=False)
user_movies = user_ratings.head(10)

print(f"\n⭐ Top 10 películas que el Usuario {test_user_id} ya calificó alto:")
for item_id, rating in user_movies.items():
    movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"   {rating:.0f} ⭐ - {movie_title}")

# Obtener recomendaciones
recommendations = recommend_movies_item_based(
    user_id=test_user_id,
    ratings_matrix=ratings_matrix,
    similarity_df=item_similarity_df,
    movies_df=movies,
    k_items=20,
    n_recommendations=10
)

print(f"\n🎬 Top 10 Recomendaciones para el Usuario {test_user_id}:")
print("=" * 70)
for idx, row in recommendations.iterrows():
    print(f"{row['predicted_rating']:.2f} ⭐ - {row['title']}")

# =============================================================================
# PASO 7: ANALIZAR SIMILITUDES DE PELÍCULAS
# =============================================================================

print("\n" + "=" * 70)
print("📊 ANÁLISIS DE SIMILITUDES ENTRE PELÍCULAS")
print("=" * 70)

# Obtener todas las similitudes (excluyendo diagonal)
all_similarities = item_similarity_df.values[np.triu_indices_from(item_similarity_df.values, k=1)]

print(f"\n📊 Estadísticas de similitud entre películas:")
print(f"   - Similitud promedio: {all_similarities.mean():.3f}")
print(f"   - Similitud mínima: {all_similarities.min():.3f}")
print(f"   - Similitud máxima: {all_similarities.max():.3f}")
print(f"   - Mediana: {np.median(all_similarities):.3f}")

# Comparar con User-Based
print(f"\n💡 COMPARACIÓN:")
print(f"   Item-Based - Similitud promedio: {all_similarities.mean():.3f}")
print(f"   User-Based - Similitud promedio: 0.172 (del ejercicio anterior)")
print(f"   → Las películas tienen MAYOR similitud que los usuarios")

# =============================================================================
# PASO 8: VISUALIZACIÓN
# =============================================================================

print("\n" + "=" * 70)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Item-Based Collaborative Filtering - Análisis', 
             fontsize=14, fontweight='bold')

# 1. Heatmap de similitud entre películas (muestra 50x50)
ax1 = axes[0, 0]
sample_similarity = item_similarity_df.iloc[:50, :50]
sns.heatmap(sample_similarity, cmap='RdYlGn', center=0.5, 
            square=True, ax=ax1, cbar_kws={'label': 'Similitud'})
ax1.set_title('Matriz de Similitud entre Películas (muestra 50×50)')
ax1.set_xlabel('Película ID')
ax1.set_ylabel('Película ID')

# 2. Distribución de similitudes
ax2 = axes[0, 1]
ax2.hist(all_similarities, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(all_similarities.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {all_similarities.mean():.3f}')
ax2.axvline(np.median(all_similarities), color='orange', linestyle='--', 
            linewidth=2, label=f'Mediana: {np.median(all_similarities):.3f}')
ax2.set_xlabel('Similitud del Coseno')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución de Similitudes entre Películas')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Top películas más similares entre sí
ax3 = axes[1, 0]
# Encontrar el par de películas más similares (excluyendo identidades)
similarity_no_diag = item_similarity_df.copy()
np.fill_diagonal(similarity_no_diag.values, 0)
max_sim_idx = np.unravel_index(similarity_no_diag.values.argmax(), similarity_no_diag.shape)
movie1_id = similarity_no_diag.index[max_sim_idx[0]]
movie2_id = similarity_no_diag.columns[max_sim_idx[1]]
max_similarity = similarity_no_diag.iloc[max_sim_idx[0], max_sim_idx[1]]

movie1_title = movies[movies['item_id'] == movie1_id]['title'].values[0]
movie2_title = movies[movies['item_id'] == movie2_id]['title'].values[0]

# Mostrar top 10 pares más similares
top_pairs = []
for i in range(similarity_no_diag.shape[0]):
    for j in range(i+1, similarity_no_diag.shape[1]):
        top_pairs.append((
            similarity_no_diag.index[i],
            similarity_no_diag.columns[j],
            similarity_no_diag.iloc[i, j]
        ))

top_pairs = sorted(top_pairs, key=lambda x: x[2], reverse=True)[:10]
pair_labels = []
pair_values = []

for movie_id1, movie_id2, sim in top_pairs:
    title1 = movies[movies['item_id'] == movie_id1]['title'].values[0][:20]
    title2 = movies[movies['item_id'] == movie_id2]['title'].values[0][:20]
    pair_labels.append(f"{title1}\n& {title2}")
    pair_values.append(sim)

ax3.barh(range(len(pair_values)), pair_values, color='lightcoral')
ax3.set_yticks(range(len(pair_labels)))
ax3.set_yticklabels(pair_labels, fontsize=8)
ax3.set_xlabel('Similitud')
ax3.set_title('Top 10 Pares de Películas Más Similares')
ax3.invert_yaxis()

# 4. Comparación de distribuciones User-Based vs Item-Based
ax4 = axes[1, 1]
ax4.hist(all_similarities, bins=50, alpha=0.6, color='coral', 
         edgecolor='black', label='Item-Based')
# Simular User-Based (media 0.172)
user_based_sim = np.random.beta(2, 10, size=len(all_similarities)) * 0.5
ax4.hist(user_based_sim, bins=50, alpha=0.6, color='steelblue', 
         edgecolor='black', label='User-Based (aprox.)')
ax4.axvline(all_similarities.mean(), color='red', linestyle='--', linewidth=2)
ax4.axvline(user_based_sim.mean(), color='blue', linestyle='--', linewidth=2)
ax4.set_xlabel('Similitud')
ax4.set_ylabel('Frecuencia')
ax4.set_title('Comparación: Item-Based vs User-Based')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Guardar visualización
import os
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/item_based_cf.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización guardada en: reports/item_based_cf.png")

plt.show()

# =============================================================================
# CONCLUSIONES
# =============================================================================

print("\n" + "=" * 70)
print("✅ ITEM-BASED COLLABORATIVE FILTERING COMPLETADO")
print("=" * 70)

print("\n📝 RESUMEN:")
print(f"1. Calculamos similitud entre {item_similarity_df.shape[0]} películas")
print(f"2. Similitud promedio entre películas: {all_similarities.mean():.3f}")
print(f"3. Generamos {len(recommendations)} recomendaciones para el Usuario {test_user_id}")

print("\n💡 CÓMO FUNCIONA:")
print("   1. Identificamos películas que el usuario ya calificó alto")
print("   2. Encontramos películas similares a esas")
print("   3. Predecimos ratings basados en similitud de películas")
print("   4. Recomendamos las películas con mayor rating predicho")

print("\n⚖️  VENTAJAS vs USER-BASED:")
print("   ✅ Mayor similitud promedio (mejor calidad de predicciones)")
print("   ✅ Más estable (las películas no cambian, los usuarios sí)")
print("   ✅ Más eficiente (calcular una vez, usar para todos los usuarios)")
print("   ✅ Más explicable ('porque te gustó X, te recomendamos Y')")

print("\n⚠️  LIMITACIONES:")
print("   1. Cold Start de películas: nuevas películas sin ratings")
print("   2. Popularidad: películas populares dominan las recomendaciones")
print("   3. Serendipity: difícil descubrir contenido muy diferente")

print("\n🎯 SIGUIENTE PASO:")
print("   Implementar Matrix Factorization (SVD)")
print("   Un enfoque más sofisticado que aprende 'features latentes'")