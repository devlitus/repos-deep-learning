"""
Matrix Factorization con SVD (Singular Value Decomposition)
Paso 5: Implementar sistema de recomendación usando descomposición matricial
"""
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("🔬 MATRIX FACTORIZATION CON SVD")
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
print(f"📊 Sparsity: {(ratings_matrix.isna().sum().sum() / (ratings_matrix.shape[0] * ratings_matrix.shape[1]) * 100):.2f}%")

# =============================================================================
# PASO 2: PREPARAR DATOS PARA SVD
# =============================================================================

print("\n" + "=" * 70)
print("🔧 PREPARANDO DATOS PARA SVD")
print("=" * 70)

print("\n💡 TEORÍA - SVD (Singular Value Decomposition):")
print("   SVD descompone la matriz R en tres matrices:")
print("   R ≈ U × Σ × V^T")
print("")
print("   Donde:")
print("   - U: Matriz de usuarios (users × k features)")
print("   - Σ: Valores singulares (importancia de cada feature)")
print("   - V^T: Matriz de películas (k features × items)")
print("   - k: Número de features latentes (dimensión reducida)")
print("")
print("   Propósito: Reducir dimensionalidad y capturar patrones ocultos")

# Rellenar NaN con la media de cada usuario
# Esto es importante: SVD necesita una matriz completa (sin NaN)
print("\n⚙️  Rellenando valores faltantes con la media de cada usuario...")

# Calcular media por usuario
user_ratings_mean = ratings_matrix.mean(axis=1)

# Crear matriz normalizada (restar media de cada usuario)
# Esto ayuda a que SVD capture preferencias relativas, no absolutas
ratings_matrix_norm = ratings_matrix.sub(user_ratings_mean, axis=0)

# Rellenar NaN con 0 (significa "sin preferencia especial" después de normalizar)
ratings_matrix_filled = ratings_matrix_norm.fillna(0)

print(f"✅ Matriz preparada: {ratings_matrix_filled.shape}")
print(f"   - Media de ratings por usuario calculada")
print(f"   - Matriz normalizada (restando media)")
print(f"   - NaN rellenados con 0")

# =============================================================================
# PASO 3: APLICAR SVD
# =============================================================================

print("\n" + "=" * 70)
print("🔬 APLICANDO SVD")
print("=" * 70)

# Número de features latentes (dimensión reducida)
k = 50  # Reducir de 1682 dimensiones a 50

print(f"\n⚙️  Descomponiendo matriz con k={k} features latentes...")
print("   (Esto puede tardar unos segundos...)")

# Aplicar SVD
# svds devuelve: U, sigma, Vt
U, sigma, Vt = svds(ratings_matrix_filled.values, k=k)

# sigma es un vector, necesitamos convertirlo en matriz diagonal
sigma_matrix = np.diag(sigma)

print(f"\n✅ SVD completado!")
print(f"   - Matriz U (usuarios): {U.shape}")
print(f"   - Matriz Σ (valores singulares): {sigma_matrix.shape}")
print(f"   - Matriz V^T (películas): {Vt.shape}")

print(f"\n📊 Valores singulares (importancia de features):")
print(f"   - Máximo: {sigma.max():.2f}")
print(f"   - Mínimo: {sigma.min():.2f}")
print(f"   - Los primeros valores son los más importantes")

# =============================================================================
# PASO 4: RECONSTRUIR MATRIZ DE PREDICCIONES
# =============================================================================

print("\n" + "=" * 70)
print("🔄 RECONSTRUYENDO MATRIZ DE PREDICCIONES")
print("=" * 70)

# Reconstruir la matriz: U × Σ × V^T
print("\n⚙️  Calculando: Predicciones = U × Σ × V^T")
predictions_normalized = np.dot(np.dot(U, sigma_matrix), Vt)

# Convertir de vuelta a DataFrame
predictions_normalized_df = pd.DataFrame(
    predictions_normalized,
    index=ratings_matrix.index,
    columns=ratings_matrix.columns
)

# Añadir de vuelta la media de cada usuario (desnormalizar)
print("⚙️  Añadiendo media de usuarios (desnormalización)...")
predictions_df = predictions_normalized_df.add(user_ratings_mean, axis=0)

# Limitar predicciones al rango [1, 5]
predictions_df = predictions_df.clip(lower=1, upper=5)

print(f"✅ Matriz de predicciones creada: {predictions_df.shape}")
print(f"   - Todas las celdas ahora tienen predicciones (incluso las que eran NaN)")

# =============================================================================
# PASO 5: EVALUAR EL MODELO
# =============================================================================

print("\n" + "=" * 70)
print("📊 EVALUANDO EL MODELO")
print("=" * 70)

# Obtener ratings reales (solo los que existen)
real_ratings = ratings_matrix.values[~np.isnan(ratings_matrix.values)]
predicted_ratings = predictions_df.values[~np.isnan(ratings_matrix.values)]

# Calcular métricas
rmse = np.sqrt(mean_squared_error(real_ratings, predicted_ratings))
mae = mean_absolute_error(real_ratings, predicted_ratings)

print(f"\n📈 Métricas del modelo:")
print(f"   - RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"   - MAE (Mean Absolute Error): {mae:.4f}")
print(f"\n💡 Interpretación:")
print(f"   - RMSE: En promedio, nos equivocamos por ±{rmse:.2f} estrellas")
print(f"   - MAE: Error absoluto promedio de {mae:.2f} estrellas")
print(f"   - Menor es mejor (ideal sería 0.0)")

# Comparar con baseline (predecir siempre la media)
baseline_predictions = np.full_like(real_ratings, real_ratings.mean())
baseline_rmse = np.sqrt(mean_squared_error(real_ratings, baseline_predictions))
baseline_mae = mean_absolute_error(real_ratings, baseline_predictions)

print(f"\n📊 Comparación con baseline (predecir siempre la media):")
print(f"   Baseline RMSE: {baseline_rmse:.4f}")
print(f"   SVD RMSE: {rmse:.4f}")
print(f"   Mejora: {((baseline_rmse - rmse) / baseline_rmse * 100):.1f}%")

# =============================================================================
# PASO 6: FUNCIÓN PARA RECOMENDAR PELÍCULAS
# =============================================================================

def recommend_movies_svd(user_id, predictions_df, ratings_matrix, movies_df, n_recommendations=10):
    """
    Recomienda películas usando las predicciones de SVD
    
    Args:
        user_id: ID del usuario
        predictions_df: DataFrame con predicciones
        ratings_matrix: Matriz original de ratings
        movies_df: DataFrame con información de películas
        n_recommendations: Número de recomendaciones
    
    Returns:
        DataFrame con recomendaciones
    """
    # Obtener predicciones para el usuario
    user_predictions = predictions_df.loc[user_id]
    
    # Obtener películas que el usuario YA ha visto
    user_ratings = ratings_matrix.loc[user_id]
    seen_movies = user_ratings.dropna().index.tolist()
    
    # Filtrar películas no vistas
    unseen_predictions = user_predictions.drop(seen_movies)
    
    # Ordenar por rating predicho
    top_predictions = unseen_predictions.sort_values(ascending=False).head(n_recommendations)
    
    # Crear DataFrame con recomendaciones
    recommendations = pd.DataFrame({
        'item_id': top_predictions.index,
        'predicted_rating': top_predictions.values
    })
    
    # Añadir información de películas
    recommendations = recommendations.merge(
        movies_df[['item_id', 'title']],
        on='item_id',
        how='left'
    )
    
    return recommendations

# =============================================================================
# PASO 7: PROBAR EL SISTEMA DE RECOMENDACIÓN
# =============================================================================

print("\n" + "=" * 70)
print("🎯 PROBANDO SVD - RECOMENDACIONES")
print("=" * 70)

# Usar el mismo usuario que en ejercicios anteriores
test_user_id = 1

print(f"\n👤 Usuario de prueba: {test_user_id}")

# Ver películas que ya calificó alto
user_ratings = ratings_matrix.loc[test_user_id].dropna().sort_values(ascending=False)
user_movies = user_ratings.head(10)

print(f"\n⭐ Top 10 películas que el Usuario {test_user_id} ya calificó alto:")
for item_id, rating in user_movies.items():
    movie_title = movies[movies['item_id'] == item_id]['title'].values[0]
    print(f"   {rating:.0f} ⭐ - {movie_title}")

# Obtener recomendaciones
print(f"\n🎬 Calculando recomendaciones con SVD...")
recommendations = recommend_movies_svd(
    user_id=test_user_id,
    predictions_df=predictions_df,
    ratings_matrix=ratings_matrix,
    movies_df=movies,
    n_recommendations=10
)

print(f"\n🎬 Top 10 Recomendaciones para el Usuario {test_user_id}:")
print("=" * 70)
for idx, row in recommendations.iterrows():
    print(f"{row['predicted_rating']:.2f} ⭐ - {row['title']}")

# =============================================================================
# PASO 8: VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 70)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Matrix Factorization con SVD - Análisis', 
             fontsize=14, fontweight='bold')

# 1. Valores singulares (importancia de features)
ax1 = axes[0, 0]
ax1.plot(range(1, k+1), sigma, 'o-', color='steelblue', linewidth=2, markersize=6)
ax1.set_xlabel('Índice del Feature Latente')
ax1.set_ylabel('Valor Singular (Importancia)')
ax1.set_title(f'Valores Singulares - Top {k} Features Latentes')
ax1.grid(alpha=0.3)
ax1.axhline(y=sigma.mean(), color='red', linestyle='--', label=f'Media: {sigma.mean():.1f}')
ax1.legend()

# 2. Distribución de errores
ax2 = axes[0, 1]
errors = predicted_ratings - real_ratings
ax2.hist(errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
ax2.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
            label=f'Media: {errors.mean():.3f}')
ax2.set_xlabel('Error de Predicción (Predicho - Real)')
ax2.set_ylabel('Frecuencia')
ax2.set_title('Distribución de Errores del Modelo SVD')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Real vs Predicho (muestra aleatoria)
ax3 = axes[1, 0]
sample_size = 1000
sample_indices = np.random.choice(len(real_ratings), sample_size, replace=False)
ax3.scatter(real_ratings[sample_indices], predicted_ratings[sample_indices], 
            alpha=0.3, s=10, color='steelblue')
ax3.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predicción perfecta')
ax3.set_xlabel('Rating Real')
ax3.set_ylabel('Rating Predicho')
ax3.set_title(f'Real vs Predicho (muestra de {sample_size} ratings)')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xlim(0.5, 5.5)
ax3.set_ylim(0.5, 5.5)

# 4. Comparación de métodos
ax4 = axes[1, 1]
methods = ['Baseline\n(Media)', 'SVD']
rmse_values = [baseline_rmse, rmse]
colors_bars = ['lightcoral', 'lightgreen']

bars = ax4.bar(methods, rmse_values, color=colors_bars, edgecolor='black', linewidth=2)
ax4.set_ylabel('RMSE')
ax4.set_title('Comparación de Métodos - Error de Predicción')
ax4.set_ylim(0, max(rmse_values) * 1.2)

# Añadir valores en las barras
for bar, value in zip(bars, rmse_values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=12)

# Añadir mejora porcentual
improvement = (baseline_rmse - rmse) / baseline_rmse * 100
ax4.text(0.5, max(rmse_values) * 1.1, 
         f'Mejora: {improvement:.1f}%',
         ha='center', fontsize=12, fontweight='bold', color='green')

plt.tight_layout()

# Guardar visualización
import os
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/svd_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualización guardada en: reports/svd_analysis.png")

plt.show()

# =============================================================================
# PASO 9: ANÁLISIS DE FEATURES LATENTES
# =============================================================================

print("\n" + "=" * 70)
print("🔬 ANÁLISIS DE FEATURES LATENTES")
print("=" * 70)

print(f"\n💡 Features latentes descubiertos:")
print(f"   SVD encontró {k} características ocultas")
print(f"   Estas NO son géneros explícitos, sino patrones de preferencias")
print(f"\n   Ejemplo hipotético de lo que podrían representar:")
print(f"   - Feature 1: 'Acción vs Drama'")
print(f"   - Feature 2: 'Mainstream vs Independiente'")
print(f"   - Feature 3: 'Clásico vs Moderno'")
print(f"   - ... (y 47 más)")
print(f"\n   El modelo aprende estos features AUTOMÁTICAMENTE de los datos")

# Mostrar el "perfil latente" del Usuario 1
user_latent_profile = U[test_user_id - 1]  # -1 porque índices empiezan en 0
print(f"\n👤 Perfil latente del Usuario {test_user_id} (primeros 10 features):")
for i, value in enumerate(user_latent_profile[:10], 1):
    print(f"   Feature {i}: {value:+.3f}")

print(f"\n💡 Interpretación:")
print(f"   - Valores positivos: preferencia hacia ese feature")
print(f"   - Valores negativos: preferencia contraria")
print(f"   - Magnitud: qué tan fuerte es la preferencia")

# =============================================================================
# CONCLUSIONES
# =============================================================================

print("\n" + "=" * 70)
print("✅ MATRIX FACTORIZATION CON SVD COMPLETADO")
print("=" * 70)

print("\n📝 RESUMEN:")
print(f"1. Descompusimos matriz de {ratings_matrix.shape[0]}×{ratings_matrix.shape[1]} en {k} features latentes")
print(f"2. RMSE del modelo: {rmse:.4f} ({improvement:.1f}% mejor que baseline)")
print(f"3. Generamos recomendaciones basadas en patrones ocultos")

print("\n💡 CÓMO FUNCIONA SVD:")
print("   1. Encuentra características latentes (ocultas) en los datos")
print("   2. Representa usuarios y películas en ese espacio latente")
print("   3. Predice ratings calculando similitud en el espacio latente")
print("   4. Captura patrones complejos que CF básico no puede")

print("\n⚖️  COMPARACIÓN DE MÉTODOS:")
print("   User-Based CF:")
print("   ✅ Fácil de entender e implementar")
print("   ❌ Baja similitud promedio (0.172)")
print("   ❌ Problemas con sparsity")
print("")
print("   Item-Based CF:")
print("   ✅ Más explicable ('porque te gustó X')")
print("   ✅ Más estable")
print("   ❌ Similitud aún baja (0.086)")
print("")
print("   SVD:")
print("   ✅ Maneja sparsity mejor")
print("   ✅ Captura patrones complejos")
print("   ✅ Mejor precisión (RMSE más bajo)")
print("   ❌ Menos interpretable (features latentes abstractas)")

print("\n🎯 SIGUIENTE PASO:")
print("   Implementar Sistema Híbrido")
print("   (Combinar múltiples enfoques para mejores recomendaciones)")