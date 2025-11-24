"""
Matrix Factorization (SVD) para Amazon Fashion Reviews
Paso 5: Descomposición SVD para predicción de ratings
"""
import sys
import io
import json
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

print("=" * 70)
print("📊 MATRIX FACTORIZATION (SVD) - AMAZON FASHION")
print("=" * 70)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / 'data' / 'raw'

json_file = DATA_DIR / 'fashion_reviews.json'

if not json_file.exists():
    print(f"❌ Error: No se encontró {json_file}")
    sys.exit(1)

# Cargar datos
print("\n📥 Cargando datos...")
reviews = []
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
print(f"✅ {len(ratings):,} reviews cargadas")

# Crear matriz de ratings
ratings_matrix = ratings.pivot_table(
    index='user_id',
    columns='product_id',
    values='rating',
    aggfunc='mean'
)

print(f"📐 Matriz: {ratings_matrix.shape[0]} usuarios × {ratings_matrix.shape[1]} productos")

# =============================================================================
# APLICAR SVD
# =============================================================================

print("\n" + "=" * 70)
print("🔢 APLICANDO SVD (SINGULAR VALUE DECOMPOSITION)")
print("=" * 70)

# Rellenar con promedio de usuario
ratings_filled = ratings_matrix.fillna(ratings_matrix.mean())

print(f"\n⚙️  Descomponiendo matriz con SVD...")

# Convertir a matriz sparse para eficiencia
ratings_sparse = csr_matrix(ratings_filled.values)

# Aplicar SVD con 50 factores latentes
U, sigma, Vt = svds(ratings_sparse, k=50)

print(f"✅ SVD completado")
print(f"   - U (usuarios): {U.shape}")
print(f"   - Sigma (valores singulares): {len(sigma)}")
print(f"   - Vt (productos): {Vt.shape}")

# Reconstruir matriz de predicciones
sigma_diag = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma_diag), Vt)
predicted_ratings_df = pd.DataFrame(
    predicted_ratings,
    index=ratings_matrix.index,
    columns=ratings_matrix.columns
)

# Clip a rango 1-5
predicted_ratings_df = predicted_ratings_df.clip(1, 5)

print(f"\n✅ Matriz de predicciones generada")

# =============================================================================
# EVALUACIÓN
# =============================================================================

print("\n" + "=" * 70)
print("📊 MÉTRICAS DE EVALUACIÓN")
print("=" * 70)

from sklearn.metrics import mean_squared_error, mean_absolute_error

actual = []
predicted = []

test_data = ratings.sample(n=min(100, len(ratings)//100), random_state=42)

for _, row in test_data.iterrows():
    user_id = row['user_id']
    product_id = row['product_id']
    
    if user_id in predicted_ratings_df.index and product_id in predicted_ratings_df.columns:
        actual.append(row['rating'])
        predicted.append(predicted_ratings_df.loc[user_id, product_id])

if len(actual) > 0:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    
    print(f"\n✅ Métricas ({len(actual)} predicciones):")
    print(f"   - RMSE: {rmse:.4f}")
    print(f"   - MAE: {mae:.4f}")

# =============================================================================
# VISUALIZACIONES
# =============================================================================

print("\n" + "=" * 70)
print("📈 GENERANDO VISUALIZACIONES")
print("=" * 70)

REPORTS_DIR = PROJECT_DIR / 'reports'
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Valores singulares
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(sorted(sigma, reverse=True), marker='o', linewidth=2, markersize=6)
ax.set_title('Valores Singulares (SVD)', fontsize=14, fontweight='bold')
ax.set_xlabel('Factor Latente', fontsize=12)
ax.set_ylabel('Valor Singular', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORTS_DIR / 'svd_singular_values.png', dpi=300, bbox_inches='tight')
print("✅ Gráfico: svd_singular_values.png")
plt.close()

# Predicción vs Real
if len(actual) > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(actual, predicted, alpha=0.6, s=50, color='green')
    ax.plot([1, 5], [1, 5], 'r--', lw=2, label='Predicción perfecta')
    ax.set_xlabel('Rating Real', fontsize=12)
    ax.set_ylabel('Rating Predicho (SVD)', fontsize=12)
    ax.set_title('SVD: Rating Predicho vs Real', fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'svd_predictions.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico: svd_predictions.png")
    plt.close()

print("\n✅ MATRIX FACTORIZATION (SVD) COMPLETADO")
