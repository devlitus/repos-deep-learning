"""
Item-Based Collaborative Filtering para Amazon Fashion Reviews
Paso 4: Implementar sistema de recomendación basado en similitud de productos
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
print("👕 ITEM-BASED COLLABORATIVE FILTERING - AMAZON FASHION")
print("=" * 70)

# =============================================================================
# PASO 1: CARGAR Y PREPARAR DATOS
# =============================================================================

print("\n📊 Cargando datos...")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw'

json_file = DATA_DIR / 'fashion_reviews.json'

if not json_file.exists():
    print(f"❌ Error: No se encontró {json_file}")
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

# Crear matriz user-item
ratings_matrix = ratings.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    aggfunc='mean'
)

print(f"\n📐 Matriz de ratings: {ratings_matrix.shape[0]} usuarios × {ratings_matrix.shape[1]} productos")

# =============================================================================
# PASO 2: CALCULAR SIMILITUD ENTRE PRODUCTOS
# =============================================================================

print("\n" + "=" * 70)
print("🔢 CALCULANDO SIMILITUD ENTRE PRODUCTOS")
print("=" * 70)

print("\n💡 DIFERENCIA CLAVE CON USER-BASED:")
print("   User-Based: Compara usuarios → ¿Qué usuarios son similares?")
print("   Item-Based: Compara productos → ¿Qué productos son similares?")

ratings_matrix_transposed = ratings_matrix.T
ratings_matrix_filled = ratings_matrix_transposed.fillna(0)

print(f"\n⚙️  Calculando matriz de similitud del coseno...")

item_similarity = cosine_similarity(ratings_matrix_filled)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=ratings_matrix_transposed.index,
    columns=ratings_matrix_transposed.index
)

print(f"✅ Matriz de similitud creada: {item_similarity_df.shape[0]} × {item_similarity_df.shape[1]}")

# =============================================================================
# PASO 3: FUNCIÓN PARA ENCONTRAR PRODUCTOS SIMILARES
# =============================================================================

def find_similar_items(item_id, similarity_df, k=10):
    """Encuentra los k productos más similares a un producto dado"""
    try:
        similarities = similarity_df.loc[item_id]
        similar_items = similarities.sort_values(ascending=False)[1:k+1]
        return similar_items
    except KeyError:
        return pd.Series(dtype=float)

# =============================================================================
# PASO 4: FUNCIÓN PARA PREDECIR RATING (ITEM-BASED)
# =============================================================================

def predict_rating_item_based(user_id, item_id, ratings_matrix, similarity_df, k=10):
    """Predice rating basado en productos similares que el usuario ya calificó"""
    similar_items = find_similar_items(item_id, similarity_df, k)

    if len(similar_items) == 0:
        return 3.0

    try:
        user_ratings_for_similar = ratings_matrix.loc[user_id, similar_items.index]
    except KeyError:
        return 3.0

    valid_ratings = user_ratings_for_similar.dropna()

    if len(valid_ratings) == 0:
        user_mean = ratings_matrix.loc[user_id].mean()
        return user_mean if not np.isnan(user_mean) else 3.0

    valid_similarities = similar_items.loc[valid_ratings.index]
    weighted_sum = (valid_similarities * valid_ratings).sum()
    similarity_sum = valid_similarities.sum()

    return weighted_sum / similarity_sum if similarity_sum > 0 else 3.0

# =============================================================================
# PASO 5: GENERAR RECOMENDACIONES
# =============================================================================

print("\n" + "=" * 70)
print("🎯 GENERANDO RECOMENDACIONES BASADAS EN PRODUCTOS SIMILARES")
print("=" * 70)

def get_recommendations_item_based(user_id, ratings_matrix, similarity_df, n_recommendations=10, k_similar=10):
    """Obtiene recomendaciones basadas en productos similares"""
    rated_items = ratings_matrix.loc[user_id].dropna().index.tolist()
    all_items = ratings_matrix.columns.tolist()
    unrated_items = [item for item in all_items if item not in rated_items]

    predictions = []
    for item_id in unrated_items:
        predicted_rating = predict_rating_item_based(user_id, item_id, ratings_matrix, similarity_df, k_similar)
        predictions.append({
            'product_id': item_id,
            'predicted_rating': predicted_rating
        })

    recommendations_df = pd.DataFrame(predictions)
    return recommendations_df.sort_values('predicted_rating', ascending=False).head(n_recommendations)

# Generar recomendaciones para usuarios de ejemplo
print("\n📋 Recomendaciones:")
sample_users = ratings_matrix.index[:min(5, len(ratings_matrix))]

for user_id in sample_users:
    print(f"\n👤 Usuario {user_id}:")
    print("   Productos calificados:", len(ratings_matrix.loc[user_id].dropna()))

    recommendations = get_recommendations_item_based(user_id, ratings_matrix, item_similarity_df, n_recommendations=5, k_similar=10)

    if len(recommendations) > 0:
        print("   Top 5 recomendaciones:")
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"      {idx}. Producto {row['product_id']}: {row['predicted_rating']:.2f}/5.0 ⭐")

# =============================================================================
# PASO 6: EVALUACIÓN
# =============================================================================

print("\n" + "=" * 70)
print("📊 MÉTRICAS DE EVALUACIÓN")
print("=" * 70)

from sklearn.metrics import mean_squared_error, mean_absolute_error

actual_ratings = []
predicted_ratings = []

test_sample_size = min(100, len(ratings) // 100)
test_data = ratings.sample(n=test_sample_size, random_state=42)

for _, row in test_data.iterrows():
    user_id = row['user_id']
    item_id = row['product_id']
    actual_rating = row['rating']

    if user_id in ratings_matrix.index and item_id in ratings_matrix.columns:
        predicted = predict_rating_item_based(user_id, item_id, ratings_matrix, item_similarity_df, k=10)
        predicted_ratings.append(predicted)
        actual_ratings.append(actual_rating)

if len(actual_ratings) > 0:
    rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
    mae = mean_absolute_error(actual_ratings, predicted_ratings)

    print(f"\n✅ Métricas ({len(actual_ratings)} predicciones):")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")

# =============================================================================
# PASO 7: VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 70)
print("📈 GENERANDO VISUALIZACIONES")
print("=" * 70)

REPORTS_DIR = PROJECT_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 6))
similarity_values = item_similarity_df.values.flatten()
similarity_values = similarity_values[similarity_values < 1.0]

ax.hist(similarity_values, bins=50, color='coral', edgecolor='black', alpha=0.7)
ax.set_title('Distribución de Similitudes entre Productos', fontsize=14, fontweight='bold')
ax.set_xlabel('Similitud del Coseno', fontsize=12)
ax.set_ylabel('Frecuencia', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'item_similarity_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico: item_similarity_distribution.png")
plt.close()

if len(actual_ratings) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(actual_ratings, predicted_ratings, alpha=0.6, s=50, color='coral')
    axes[0].plot([1, 5], [1, 5], 'r--', lw=2, label='Predicción perfecta')
    axes[0].set_xlabel('Rating Real', fontsize=12)
    axes[0].set_ylabel('Rating Predicho', fontsize=12)
    axes[0].set_title('Item-Based: Rating Predicho vs Real', fontsize=14, fontweight='bold')
    axes[0].set_xlim(0.5, 5.5)
    axes[0].set_ylim(0.5, 5.5)
    axes[0].grid(True, alpha=0.3)

    residuals = np.array(predicted_ratings) - np.array(actual_ratings)
    axes[1].hist(residuals, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1].axvline(0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Error (Predicho - Real)', fontsize=12)
    axes[1].set_ylabel('Frecuencia', fontsize=12)
    axes[1].set_title('Distribución de Errores', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'item_based_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico: item_based_predictions.png")
    plt.close()

print("\n✅ ITEM-BASED COLLABORATIVE FILTERING COMPLETADO")
