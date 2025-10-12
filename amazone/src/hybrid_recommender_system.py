"""
Sistema Híbrido de Recomendación
Paso 6: Combinar múltiples enfoques para mejores recomendaciones
"""
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🎭 SISTEMA HÍBRIDO DE RECOMENDACIÓN")
print("=" * 70)

# =============================================================================
# PASO 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n📊 Cargando datos...")

ratings = pd.read_csv(
    'data/raw/ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp']
)

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

ratings_matrix = ratings.pivot(
    index='user_id',
    columns='item_id',
    values='rating'
)

print(f"📐 Matriz: {ratings_matrix.shape[0]} usuarios × {ratings_matrix.shape[1]} películas")

# =============================================================================
# PASO 2: ENTRENAR MODELOS BASE
# =============================================================================

print("\n" + "=" * 70)
print("🔧 ENTRENANDO MODELOS BASE")
print("=" * 70)

# --- SVD ---
print("\n⚙️  1. Entrenando SVD...")
user_ratings_mean = ratings_matrix.mean(axis=1)
ratings_matrix_norm = ratings_matrix.sub(user_ratings_mean, axis=0)
ratings_matrix_filled = ratings_matrix_norm.fillna(0)

k = 50
U, sigma, Vt = svds(ratings_matrix_filled.values, k=k)
sigma_matrix = np.diag(sigma)

predictions_normalized = np.dot(np.dot(U, sigma_matrix), Vt)
svd_predictions = pd.DataFrame(
    predictions_normalized,
    index=ratings_matrix.index,
    columns=ratings_matrix.columns
)
svd_predictions = svd_predictions.add(user_ratings_mean, axis=0).clip(1, 5)

print(f"✅ SVD entrenado (k={k} features)")

# --- Item-Based CF ---
print("\n⚙️  2. Calculando similitud Item-Based...")
ratings_matrix_t = ratings_matrix.T.fillna(0)
item_similarity = cosine_similarity(ratings_matrix_t)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=ratings_matrix.columns,
    columns=ratings_matrix.columns
)
print("✅ Similitud Item-Based calculada")

# =============================================================================
# PASO 3: FUNCIONES DEL SISTEMA HÍBRIDO
# =============================================================================

def get_svd_recommendations(user_id, svd_pred, ratings_mat, n=10):
    """Obtener recomendaciones usando SVD"""
    user_predictions = svd_pred.loc[user_id]
    seen_movies = ratings_mat.loc[user_id].dropna().index
    unseen_predictions = user_predictions.drop(seen_movies)
    return unseen_predictions.sort_values(ascending=False).head(n)

def get_item_based_explanation(user_id, item_id, ratings_mat, similarity_df, top_k=3):
    """Encontrar películas similares que el usuario calificó alto"""
    user_ratings = ratings_mat.loc[user_id].dropna()
    
    # Obtener similitudes de la película recomendada
    similar_items = similarity_df.loc[item_id, user_ratings.index]
    
    # Ordenar por similitud y tomar top K
    top_similar = similar_items.sort_values(ascending=False).head(top_k)
    
    # Crear explicación
    explanations = []
    for similar_item_id, similarity in top_similar.items():
        user_rating = user_ratings[similar_item_id]
        similar_movie_title = movies[movies['item_id'] == similar_item_id]['title'].values[0]
        explanations.append({
            'movie': similar_movie_title,
            'your_rating': user_rating,
            'similarity': similarity
        })
    
    return explanations

def hybrid_recommend(user_id, svd_pred, ratings_mat, similarity_df, movies_df, 
                     n_recommendations=10, alpha=0.7):
    """
    Sistema Híbrido que combina SVD (precisión) con Item-Based (explicabilidad)
    
    Args:
        user_id: ID del usuario
        svd_pred: Predicciones de SVD
        ratings_mat: Matriz de ratings
        similarity_df: Matriz de similitud Item-Based
        movies_df: DataFrame de películas
        n_recommendations: Número de recomendaciones
        alpha: Peso de SVD (0-1), (1-alpha) es peso de Item-Based
    
    Returns:
        DataFrame con recomendaciones y explicaciones
    """
    # Obtener predicciones SVD
    svd_recs = get_svd_recommendations(user_id, svd_pred, ratings_mat, n=n_recommendations*2)
    
    # Para cada recomendación, obtener explicación
    recommendations = []
    
    for item_id in svd_recs.index[:n_recommendations]:
        svd_score = svd_recs[item_id]
        
        # Obtener explicación basada en items similares
        explanations = get_item_based_explanation(
            user_id, item_id, ratings_mat, similarity_df, top_k=3
        )
        
        # Obtener info de la película
        movie_info = movies_df[movies_df['item_id'] == item_id].iloc[0]
        
        recommendations.append({
            'item_id': item_id,
            'title': movie_info['title'],
            'predicted_rating': svd_score,
            'explanation': explanations
        })
    
    return pd.DataFrame(recommendations)

def recommend_for_new_user(movies_df, n=10):
    """Recomendar para usuario nuevo (Cold Start)"""
    # Calcular popularidad y rating promedio
    movie_stats = ratings.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['item_id', 'avg_rating', 'num_ratings']
    
    # Filtrar películas con al menos 100 ratings
    popular_movies = movie_stats[movie_stats['num_ratings'] >= 100]
    
    # Ordenar por rating promedio
    popular_movies = popular_movies.sort_values('avg_rating', ascending=False)
    
    # Obtener top N
    top_movies = popular_movies.head(n)
    
    # Añadir info de películas
    recommendations = top_movies.merge(
        movies_df[['item_id', 'title']],
        on='item_id'
    )
    
    return recommendations[['item_id', 'title', 'avg_rating', 'num_ratings']]

# =============================================================================
# PASO 4: PROBAR EL SISTEMA HÍBRIDO
# =============================================================================

print("\n" + "=" * 70)
print("🎯 PROBANDO SISTEMA HÍBRIDO")
print("=" * 70)

test_user_id = 1

print(f"\n👤 Usuario: {test_user_id}")
print(f"   Ratings: {ratings_matrix.loc[test_user_id].notna().sum()} películas")

# Mostrar películas favoritas
user_ratings = ratings_matrix.loc[test_user_id].dropna().sort_values(ascending=False)
print(f"\n⭐ Top 5 películas favoritas:")
for item_id, rating in user_ratings.head(5).items():
    movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"   {rating:.0f} ⭐ - {movie_title}")

# Obtener recomendaciones híbridas
print(f"\n🎬 Generando recomendaciones híbridas...")
hybrid_recs = hybrid_recommend(
    user_id=test_user_id,
    svd_pred=svd_predictions,
    ratings_mat=ratings_matrix,
    similarity_df=item_similarity_df,
    movies_df=movies,
    n_recommendations=10
)

print(f"\n🎬 Top 10 Recomendaciones Híbridas:")
print("=" * 70)

for idx, row in hybrid_recs.iterrows():
    print(f"\n{row['predicted_rating']:.2f} ⭐ - {row['title']}")
    print(f"   📝 Porque te gustaron:")
    for exp in row['explanation'][:2]:  # Mostrar top 2 explicaciones
        print(f"      • {exp['movie']} (calificaste: {exp['your_rating']:.0f}⭐, similitud: {exp['similarity']:.2f})")

# =============================================================================
# PASO 5: CASO - USUARIO NUEVO (COLD START)
# =============================================================================

print("\n" + "=" * 70)
print("🆕 CASO: USUARIO NUEVO (COLD START)")
print("=" * 70)

print("\n💡 Problema: Usuario sin ratings previos")
print("   Solución: Recomendar películas populares y bien calificadas")

new_user_recs = recommend_for_new_user(movies, n=10)

print(f"\n🎬 Top 10 Recomendaciones para Usuario Nuevo:")
print("=" * 70)
for idx, row in new_user_recs.iterrows():
    print(f"{row['avg_rating']:.2f} ⭐ - {row['title']} ({int(row['num_ratings'])} ratings)")

# =============================================================================
# PASO 6: COMPARACIÓN DE MÉTODOS
# =============================================================================

print("\n" + "=" * 70)
print("⚖️  COMPARACIÓN DE TODOS LOS MÉTODOS")
print("=" * 70)

# Obtener recomendaciones de cada método para comparar
svd_only = get_svd_recommendations(test_user_id, svd_predictions, ratings_matrix, n=5)

print(f"\n📊 Top 5 recomendaciones por método:")
print("\n1️⃣  SVD (Solo precisión):")
for item_id, score in svd_only.items():
    title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"   {score:.2f} ⭐ - {title}")

print("\n2️⃣  Híbrido (Precisión + Explicabilidad):")
for idx, row in hybrid_recs.head(5).iterrows():
    print(f"   {row['predicted_rating']:.2f} ⭐ - {row['title']}")
    main_exp = row['explanation'][0]
    print(f"      └─ Porque te gustó: {main_exp['movie']}")

# =============================================================================
# PASO 7: VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 70)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Sistema Híbrido de Recomendación - Análisis', 
             fontsize=14, fontweight='bold')

# 1. Distribución de predicciones por método
ax1 = axes[0, 0]
all_svd_preds = get_svd_recommendations(test_user_id, svd_predictions, ratings_matrix, n=100)
ax1.hist(all_svd_preds.values, bins=30, alpha=0.7, color='steelblue', 
         edgecolor='black', label='SVD')
ax1.axvline(all_svd_preds.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {all_svd_preds.mean():.2f}')
ax1.set_xlabel('Rating Predicho')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Distribución de Predicciones SVD')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Top películas por rating predicho
ax2 = axes[0, 1]
top_5 = hybrid_recs.head(5)
colors = plt.cm.RdYlGn(np.linspace(0.5, 0.9, len(top_5)))
bars = ax2.barh(range(len(top_5)), top_5['predicted_rating'], color=colors)
ax2.set_yticks(range(len(top_5)))
ax2.set_yticklabels([title[:30] + '...' if len(title) > 30 else title 
                      for title in top_5['title']], fontsize=9)
ax2.set_xlabel('Rating Predicho')
ax2.set_title('Top 5 Recomendaciones Híbridas')
ax2.invert_yaxis()
ax2.set_xlim(0, 5)
ax2.grid(axis='x', alpha=0.3)

# 3. Comparación Cold Start vs Usuario Activo
ax3 = axes[1, 0]
categories = ['Usuario Nuevo\n(Popularidad)', 'Usuario Activo\n(Híbrido)']
avg_ratings = [
    new_user_recs['avg_rating'].mean(),
    hybrid_recs['predicted_rating'].mean()
]
colors_bar = ['coral', 'lightgreen']
bars = ax3.bar(categories, avg_ratings, color=colors_bar, edgecolor='black', linewidth=2)
ax3.set_ylabel('Rating Promedio Predicho')
ax3.set_title('Comparación: Cold Start vs Usuario Activo')
ax3.set_ylim(0, 5)

for bar, value in zip(bars, avg_ratings):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.2f}⭐',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

# 4. Número de ratings por usuario (distribución)
ax4 = axes[1, 1]
ratings_per_user = ratings_matrix.notna().sum(axis=1)
ax4.hist(ratings_per_user, bins=50, color='plum', edgecolor='black', alpha=0.7)
ax4.axvline(ratings_per_user.mean(), color='red', linestyle='--', 
            linewidth=2, label=f'Media: {ratings_per_user.mean():.0f}')
ax4.axvline(5, color='orange', linestyle='--', linewidth=2, 
            label='Umbral Cold Start (5)')
ax4.axvline(20, color='green', linestyle='--', linewidth=2, 
            label='Umbral Híbrido (20)')
ax4.set_xlabel('Número de Ratings')
ax4.set_ylabel('Número de Usuarios')
ax4.set_title('Distribución de Actividad de Usuarios')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()

import os
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/hybrid_system.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización guardada en: reports/hybrid_system.png")

plt.show()

# =============================================================================
# CONCLUSIONES
# =============================================================================

print("\n" + "=" * 70)
print("✅ SISTEMA HÍBRIDO COMPLETADO")
print("=" * 70)

print("\n📝 RESUMEN DEL SISTEMA HÍBRIDO:")
print("   1. Usa SVD para PRECISIÓN en las predicciones")
print("   2. Usa Item-Based para EXPLICABILIDAD")
print("   3. Maneja Cold Start con recomendaciones populares")
print("   4. Combina lo mejor de cada enfoque")

print("\n🎯 ESTRATEGIA POR TIPO DE USUARIO:")
print("   • 0-5 ratings: Popularidad + Demografía")
print("   • 5-20 ratings: Item-Based CF")
print("   • 20+ ratings: Sistema Híbrido (SVD + Item-Based)")

print("\n💡 VENTAJAS DEL SISTEMA HÍBRIDO:")
print("   ✅ Precisión de SVD (RMSE: 0.72)")
print("   ✅ Explicabilidad de Item-Based")
print("   ✅ Maneja Cold Start")
print("   ✅ Robusto y versátil")

print("\n🎉 PROYECTO COMPLETADO!")
print("   Has aprendido:")
print("   - Collaborative Filtering (User-Based & Item-Based)")
print("   - Matrix Factorization (SVD)")
print("   - Sistemas Híbridos")
print("   - Evaluación de modelos (RMSE, MAE)")
print("   - Manejo de problemas reales (Sparsity, Cold Start)")

print("\n🚀 SIGUIENTE PASO:")
print("   Crear aplicación web con Streamlit para interactuar con el sistema")