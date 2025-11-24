"""
User-Based Collaborative Filtering para Amazon Fashion Reviews
Paso 3: Implementar sistema de recomendación basado en similitud de usuarios
"""
import sys
import io
import json
from pathlib import Path

# Configurar codificación UTF-8 para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("👥 USER-BASED COLLABORATIVE FILTERING - AMAZON FASHION")
print("=" * 70)

# =============================================================================
# PASO 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n📊 Cargando datos...")

# Obtener ruta del proyecto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw'

# Cargar ratings desde JSON
json_file = DATA_DIR / 'fashion_reviews.json'

if not json_file.exists():
    print(f"❌ Error: No se encontró {json_file}")
    print("Por favor ejecuta primero: python download_fashion.py")
    sys.exit(1)

# Cargar datos JSON
reviews = []
print("📥 Cargando datos JSON...")
with open(json_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            review = json.loads(line.strip())
            reviews.append({
                'user_id': review.get('reviewerID'),
                'product_id': review.get('asin'),
                'rating': review.get('overall')
            })
        except json.JSONDecodeError:
            continue

ratings = pd.DataFrame(reviews)
print(f"✅ Datos cargados: {len(ratings):,} reviews")

# Crear matriz user-item (usuario × producto)
ratings_matrix = ratings.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    aggfunc='mean'  # Si hay duplicados, tomar promedio
)

print(f"\n📐 Matriz de ratings: {ratings_matrix.shape[0]} usuarios × {ratings_matrix.shape[1]} productos")
print(f"📊 Dispersión: {(1 - ratings_matrix.notna().sum().sum() / (ratings_matrix.shape[0] * ratings_matrix.shape[1])) * 100:.2f}%")

# =============================================================================
# PASO 2: CALCULAR SIMILITUD ENTRE USUARIOS
# =============================================================================

print("\n" + "=" * 70)
print("🔢 CALCULANDO SIMILITUD ENTRE USUARIOS")
print("=" * 70)

# Rellenar NaN con 0 para el cálculo de similitud
ratings_matrix_filled = ratings_matrix.fillna(0)

print("\n⚙️  Calculando matriz de similitud del coseno...")
print("   (Esto puede tardar un poco según el tamaño del dataset...)")

# Calcular similitud del coseno entre usuarios
user_similarity = cosine_similarity(ratings_matrix_filled)

# Convertir a DataFrame
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

print(f"✅ Matriz de similitud creada: {user_similarity_df.shape[0]} × {user_similarity_df.shape[1]}")

# Mostrar ejemplo de similitudes
if len(ratings_matrix) > 0:
    example_user = ratings_matrix.index[0]
    print(f"\n📊 Ejemplo - Similitud del Usuario {example_user} con otros:")
    user_similarities = user_similarity_df.loc[example_user].sort_values(ascending=False).head(6)
    for user, sim in user_similarities.items():
        print(f"  - Usuario {user}: {sim:.4f}")

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
    try:
        similarities = similarity_df.loc[user_id]
        similar_users = similarities.sort_values(ascending=False)[1:k+1]
        return similar_users
    except KeyError:
        return pd.Series(dtype=float)

# =============================================================================
# PASO 4: FUNCIÓN PARA PREDECIR RATING
# =============================================================================

def predict_rating(user_id, product_id, ratings_matrix, similarity_df, k=10):
    """
    Predice el rating que un usuario daría a un producto
    usando el promedio ponderado de usuarios similares

    Args:
        user_id: ID del usuario
        product_id: ID del producto
        ratings_matrix: Matriz de ratings
        similarity_df: Matriz de similitudes
        k: Número de usuarios similares a considerar

    Returns:
        Rating predicho (float)
    """
    # Encontrar usuarios similares
    similar_users = find_similar_users(user_id, similarity_df, k)

    if len(similar_users) == 0:
        return 3.0

    # Obtener ratings de usuarios similares para este producto
    try:
        similar_users_ratings = ratings_matrix.loc[similar_users.index, product_id]
    except KeyError:
        return 3.0

    # Eliminar NaN
    valid_ratings = similar_users_ratings.dropna()

    if len(valid_ratings) == 0:
        user_mean = ratings_matrix.loc[user_id].mean()
        return user_mean if not np.isnan(user_mean) else 3.0

    # Obtener similitudes correspondientes
    valid_similarities = similar_users.loc[valid_ratings.index]

    # Calcular predicción como promedio ponderado
    weighted_sum = (valid_similarities * valid_ratings).sum()
    similarity_sum = valid_similarities.sum()

    predicted_rating = weighted_sum / similarity_sum if similarity_sum > 0 else 3.0

    return predicted_rating

# =============================================================================
# PASO 5: GENERAR RECOMENDACIONES
# =============================================================================

print("\n" + "=" * 70)
print("🎯 GENERANDO RECOMENDACIONES")
print("=" * 70)

def get_recommendations(user_id, ratings_matrix, similarity_df, n_recommendations=10, k_neighbors=10):
    """
    Obtiene las top-N recomendaciones para un usuario

    Args:
        user_id: ID del usuario
        ratings_matrix: Matriz de ratings
        similarity_df: Matriz de similitudes
        n_recommendations: Número de recomendaciones
        k_neighbors: Número de usuarios similares a considerar

    Returns:
        DataFrame con productos recomendados y ratings predichos
    """
    # Productos que el usuario ya ha calificado
    rated_products = ratings_matrix.loc[user_id].dropna().index.tolist()

    # Todos los productos
    all_products = ratings_matrix.columns.tolist()

    # Productos sin calificar
    unrated_products = [p for p in all_products if p not in rated_products]

    # Predecir ratings para productos sin calificar
    predictions = []
    for product_id in unrated_products:
        predicted_rating = predict_rating(user_id, product_id, ratings_matrix, similarity_df, k_neighbors)
        predictions.append({
            'product_id': product_id,
            'predicted_rating': predicted_rating
        })

    # Convertir a DataFrame y ordenar
    recommendations_df = pd.DataFrame(predictions)
    recommendations_df = recommendations_df.sort_values('predicted_rating', ascending=False).head(n_recommendations)

    return recommendations_df

# Generar recomendaciones para algunos usuarios de prueba
print("\n📋 Recomendaciones de ejemplo:")
print("-" * 70)

sample_users = ratings_matrix.index[:min(5, len(ratings_matrix))]
recommendations_results = []

for user_id in sample_users:
    print(f"\n👤 Usuario {user_id}:")
    print("   Productos que ha calificado:", len(ratings_matrix.loc[user_id].dropna()))

    recommendations = get_recommendations(
        user_id,
        ratings_matrix,
        user_similarity_df,
        n_recommendations=5,
        k_neighbors=10
    )

    if len(recommendations) > 0:
        print("   Top 5 recomendaciones:")
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"      {idx}. Producto {row['product_id']}: {row['predicted_rating']:.2f}/5.0 ⭐")
        recommendations_results.append({
            'user_id': user_id,
            'recommendations': recommendations
        })
    else:
        print("      (No hay productos sin calificar)")

# =============================================================================
# PASO 6: EVALUAR RECOMENDACIONES
# =============================================================================

print("\n" + "=" * 70)
print("📊 MÉTRICAS DE EVALUACIÓN")
print("=" * 70)

from sklearn.metrics import mean_squared_error, mean_absolute_error

# Validación cruzada simple
print("\n⚙️  Calculando métricas de evaluación...")

actual_ratings = []
predicted_ratings = []

# Seleccionar un subset de datos para evaluación
test_sample_size = min(100, len(ratings) // 100)
test_data = ratings.sample(n=test_sample_size, random_state=42)

for _, row in test_data.iterrows():
    user_id = row['user_id']
    product_id = row['product_id']
    actual_rating = row['rating']

    # Verificar que el usuario y producto existen en matrices
    if user_id in ratings_matrix.index and product_id in ratings_matrix.columns:
        # Crear matriz temporal sin esta valoración
        temp_matrix = ratings_matrix.copy()
        temp_matrix.loc[user_id, product_id] = np.nan

        # Predecir
        predicted = predict_rating(user_id, product_id, temp_matrix, user_similarity_df, k=10)
        predicted_ratings.append(predicted)
        actual_ratings.append(actual_rating)

if len(actual_ratings) > 0:
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = mean_absolute_error(actual_ratings, predicted_ratings)

    print(f"\n✅ Métricas calculadas en {len(actual_ratings)} predicciones:")
    print(f"   - RMSE (Root Mean Square Error): {rmse:.4f}")
    print(f"   - MAE (Mean Absolute Error): {mae:.4f}")
    print(f"   - Rating promedio (datos): {np.mean(actual_ratings):.2f}")
else:
    print("❌ No se pudieron calcular métricas")

# =============================================================================
# PASO 7: VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 70)
print("📈 GENERANDO VISUALIZACIONES")
print("=" * 70)

REPORTS_DIR = PROJECT_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Gráfico 1: Distribución de similitudes
fig, ax = plt.subplots(figsize=(12, 6))
similarity_values = user_similarity_df.values.flatten()
similarity_values = similarity_values[similarity_values < 1.0]  # Excluir diagonal (similitud 1.0 consigo mismo)

ax.hist(similarity_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_title('Distribución de Similitudes entre Usuarios (Coseno)', fontsize=14, fontweight='bold')
ax.set_xlabel('Similitud del Coseno', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'user_similarity_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico guardado: user_similarity_distribution.png")
plt.close()

# Gráfico 2: Métricas de evaluación (si existen)
if len(actual_ratings) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Predicción vs Actual
    axes[0].scatter(actual_ratings, predicted_ratings, alpha=0.6, s=50, color='steelblue', edgecolors='black')
    axes[0].plot([1, 5], [1, 5], 'r--', lw=2, label='Predicción perfecta')
    axes[0].set_xlabel('Rating Real', fontsize=12)
    axes[0].set_ylabel('Rating Predicho', fontsize=12)
    axes[0].set_title('Rating Predicho vs Real', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribución de residuos
    residuals = np.array(predicted_ratings) - np.array(actual_ratings)
    axes[1].hist(residuals, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', lw=2, label='Error = 0')
    axes[1].set_xlabel('Error (Predicho - Real)', fontsize=12)
    axes[1].set_ylabel('Frecuencia', fontsize=12)
    axes[1].set_title('Distribución de Errores de Predicción', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'user_based_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico guardado: user_based_predictions.png")
    plt.close()

print("\n" + "=" * 70)
print("✅ USER-BASED COLLABORATIVE FILTERING COMPLETADO")
print("=" * 70)
