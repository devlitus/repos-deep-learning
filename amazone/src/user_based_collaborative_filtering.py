"""
User-Based Collaborative Filtering
Paso 3: Implementar sistema de recomendación basado en similitud de usuarios
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("👥 USER-BASED COLLABORATIVE FILTERING")
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

# Crear matriz user-item
ratings_matrix = ratings.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
)

print(f"\n📐 Matriz de ratings: {ratings_matrix.shape[0]} usuarios × {ratings_matrix.shape[1]} películas")

# =============================================================================
# PASO 2: CALCULAR SIMILITUD ENTRE USUARIOS
# =============================================================================

print("\n" + "=" * 70)
print("🔢 CALCULANDO SIMILITUD ENTRE USUARIOS")
print("=" * 70)

# Rellenar NaN con 0 para el cálculo de similitud
# Importante: esto es una simplificación, hay métodos más sofisticados
ratings_matrix_filled = ratings_matrix.fillna(0)

print("\n⚙️  Calculando matriz de similitud del coseno...")
print("   (Esto puede tardar unos segundos...)")

# Calcular similitud del coseno entre usuarios
# Cada fila es un usuario, cada columna es una película
user_similarity = cosine_similarity(ratings_matrix_filled)

# Convertir a DataFrame para facilitar el acceso
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

print(f"✅ Matriz de similitud creada: {user_similarity_df.shape[0]} × {user_similarity_df.shape[1]}")

# Mostrar ejemplo de similitudes
print("\n📊 Ejemplo - Similitud del Usuario 1 con otros usuarios:")
user_1_similarities = user_similarity_df.loc[1].sort_values(ascending=False).head(6)
print(user_1_similarities)
print("\nNota: El usuario 1 tiene similitud 1.0 consigo mismo (es idéntico a sí mismo)")

# =============================================================================
# PASO 3: FUNCIÓN PARA ENCONTRAR USUARIOS SIMILARES
# =============================================================================

def find_similar_users(user_id, similarity_df, k=10):
    """
    Encuentra los k usuarios más similares a un usuario dado
    
    Args:
        user_id: ID del usuario
        similarity_df: DataFrame con similitudes
        k: Número de usuarios similares a retornar
    
    Returns:
        Series con los k usuarios más similares y sus similitudes
    """
    # Obtener similitudes del usuario
    similarities = similarity_df.loc[user_id]
    
    # Ordenar de mayor a menor (excluir el mismo usuario)
    similar_users = similarities.sort_values(ascending=False)[1:k+1]
    
    return similar_users

# =============================================================================
# PASO 4: FUNCIÓN PARA PREDECIR RATING
# =============================================================================

def predict_rating(user_id, item_id, ratings_matrix, similarity_df, k=10):
    """
    Predice el rating que un usuario daría a una película
    usando el promedio ponderado de usuarios similares
    
    Args:
        user_id: ID del usuario
        item_id: ID de la película
        ratings_matrix: Matriz de ratings
        similarity_df: Matriz de similitudes
        k: Número de usuarios similares a considerar
    
    Returns:
        Rating predicho (float)
    """
    # Encontrar usuarios similares
    similar_users = find_similar_users(user_id, similarity_df, k)
    
    # Obtener ratings de usuarios similares para esta película
    similar_users_ratings = ratings_matrix.loc[similar_users.index, item_id]
    
    # Eliminar NaN (usuarios que no han calificado esta película)
    valid_ratings = similar_users_ratings.dropna()
    valid_similarities = similar_users.loc[valid_ratings.index]
    
    # Si no hay usuarios similares que hayan calificado esta película
    if len(valid_ratings) == 0:
        # Retornar el rating promedio del usuario
        user_mean = ratings_matrix.loc[user_id].mean()
        return user_mean if not np.isnan(user_mean) else 3.0
    
    # Calcular predicción como promedio ponderado
    # rating_predicho = sum(similitud × rating) / sum(similitud)
    weighted_sum = (valid_similarities * valid_ratings).sum()
    similarity_sum = valid_similarities.sum()
    
    predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else 3.0
    
    return predicted_rating

# =============================================================================
# PASO 5: FUNCIÓN PARA RECOMENDAR PELÍCULAS
# =============================================================================

def recommend_movies(user_id, ratings_matrix, similarity_df, movies_df, k_users=10, n_recommendations=10):
    """
    Recomienda las top N películas para un usuario
    
    Args:
        user_id: ID del usuario
        ratings_matrix: Matriz de ratings
        similarity_df: Matriz de similitudes
        movies_df: DataFrame con información de películas
        k_users: Número de usuarios similares a considerar
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
    print(f"   - Calculando predicciones para películas no vistas...")
    
    # Predecir ratings para películas no vistas
    predictions = []
    for movie_id in unseen_movies:
        pred_rating = predict_rating(user_id, movie_id, ratings_matrix, similarity_df, k_users)
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
print("🎯 PROBANDO EL SISTEMA DE RECOMENDACIÓN")
print("=" * 70)

# Seleccionar un usuario de ejemplo
test_user_id = 1

print(f"\n👤 Usuario de prueba: {test_user_id}")

# Ver las películas que ya ha calificado
user_ratings = ratings_matrix.loc[test_user_id].dropna().sort_values(ascending=False)
user_movies = user_ratings.head(10)

print(f"\n⭐ Top 10 películas que el Usuario {test_user_id} ya calificó alto:")
for item_id, rating in user_movies.items():
    movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"   {rating:.0f} ⭐ - {movie_title}")

# Obtener recomendaciones
recommendations = recommend_movies(
    user_id=test_user_id,
    ratings_matrix=ratings_matrix,
    similarity_df=user_similarity_df,
    movies_df=movies,
    k_users=20,
    n_recommendations=10
)

print(f"\n🎬 Top 10 Recomendaciones para el Usuario {test_user_id}:")
print("=" * 70)
for idx, row in recommendations.iterrows():
    print(f"{row['predicted_rating']:.2f} ⭐ - {row['title']}")

# =============================================================================
# PASO 7: VISUALIZAR SIMILITUDES
# =============================================================================

print("\n" + "=" * 70)
print("📊 GENERANDO VISUALIZACIÓN DE SIMILITUDES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('User-Based Collaborative Filtering - Análisis de Similitud', 
             fontsize=14, fontweight='bold')

# 1. Heatmap de similitud (muestra 30x30)
ax1 = axes[0]
sample_similarity = user_similarity_df.iloc[:30, :30]
sns.heatmap(sample_similarity, cmap='RdYlGn', center=0.5, 
            square=True, ax=ax1, cbar_kws={'label': 'Similitud'})
ax1.set_title('Matriz de Similitud entre Usuarios (muestra 30×30)')
ax1.set_xlabel('Usuario ID')
ax1.set_ylabel('Usuario ID')

# 2. Distribución de similitudes
ax2 = axes[1]
# Obtener todas las similitudes (excluyendo diagonal)
all_similarities = user_similarity_df.values[np.triu_indices_from(user_similarity_df.values, k=1)]
ax2.hist(all_similarities, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(all_similarities.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {all_similarities.mean():.3f}')
ax2.set_xlabel('Similitud del Coseno')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución de Similitudes entre Usuarios')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Guardar visualización
import os
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/user_based_cf.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización guardada en: reports/user_based_cf.png")

plt.show()

# =============================================================================
# CONCLUSIONES
# =============================================================================

print("\n" + "=" * 70)
print("✅ USER-BASED COLLABORATIVE FILTERING COMPLETADO")
print("=" * 70)

print("\n📝 RESUMEN:")
print(f"1. Calculamos similitud entre {ratings_matrix.shape[0]} usuarios")
print(f"2. Similitud promedio entre usuarios: {all_similarities.mean():.3f}")
print(f"3. Generamos {len(recommendations)} recomendaciones para el Usuario {test_user_id}")
print(f"4. Las recomendaciones se basan en usuarios con similitud > 0.5")

print("\n💡 CÓMO FUNCIONA:")
print("   1. Encontramos usuarios similares al usuario objetivo")
print("   2. Vemos qué películas les gustaron a esos usuarios similares")
print("   3. Predecimos ratings usando promedio ponderado por similitud")
print("   4. Recomendamos las películas con mayor rating predicho")

print("\n⚠️  LIMITACIONES:")
print("   1. Sparsity: Si hay pocos usuarios similares con ratings comunes, las predicciones son malas")
print("   2. Cold Start: No funciona bien con usuarios nuevos (sin ratings)")
print("   3. Escalabilidad: Calcular similitud para millones de usuarios es costoso")

print("\n🎯 SIGUIENTE PASO:")
print("   Implementar Item-Based Collaborative Filtering")
print("   (Encuentra películas similares en lugar de usuarios similares)")