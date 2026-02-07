"""
Módulo de Matrix Factorization con SVD
Descompone la matriz usuario-producto en factores latentes para predicción
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

# Configurar UTF-8 para Windows
if sys.platform == 'win32' and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Importar configuración
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    COL_USER_ID, COL_PRODUCT_ID, COL_RATING,
    SVD_K_FACTORS, RANDOM_STATE,
    REPORTS_DIR, MODELS_DIR, PLOT_STYLE,
    REPORT_SVD_SINGULAR_VALUES, REPORT_SVD_PREDICTIONS, REPORT_SVD_LATENT_SPACE,
    SVD_MODEL_FILE
)

warnings.filterwarnings('ignore')


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(text):
    print(f"\n{'─' * 80}")
    print(f"📍 {text}")
    print("─" * 80)


def prepare_sparse_matrix(rating_matrix):
    """
    Convierte la matriz de ratings a formato sparse de scipy.

    Args:
        rating_matrix: DataFrame (usuarios x productos)

    Returns:
        scipy.sparse.csr_matrix: Matriz sparse
    """
    print_step("Preparando matriz sparse")

    rating_matrix_filled = rating_matrix.fillna(0).astype(np.float32)
    sparse_matrix = csr_matrix(rating_matrix_filled.values)

    dense_bytes = rating_matrix_filled.values.nbytes
    sparse_bytes = sparse_matrix.data.nbytes + sparse_matrix.indices.nbytes + sparse_matrix.indptr.nbytes

    print(f"  Matriz densa: {rating_matrix_filled.shape}")
    print(f"  Memoria densa: ~{dense_bytes / (1024**2):.2f} MB")
    print(f"  Memoria sparse: ~{sparse_bytes / (1024**2):.2f} MB")
    print(f"  Compresión: {((1 - sparse_bytes / dense_bytes) * 100):.1f}%")

    return sparse_matrix


def apply_svd(sparse_matrix, k=None):
    """
    Aplica descomposición SVD a la matriz sparse.

    Args:
        sparse_matrix: scipy.sparse.csr_matrix
        k: Número de factores latentes (default: config.SVD_K_FACTORS)

    Returns:
        tuple: (U, sigma, Vt)
    """
    if k is None:
        k = SVD_K_FACTORS

    # Asegurar que k no exceda las dimensiones de la matriz
    k = min(k, min(sparse_matrix.shape) - 1)

    print_step(f"Aplicando SVD con k={k} factores latentes")

    U, sigma, Vt = svds(sparse_matrix, k=k)

    print(f"  ✅ SVD completado")
    print(f"  U (factores de usuarios): {U.shape}")
    print(f"  Sigma (importancia): {sigma.shape}")
    print(f"  Vt (factores de productos): {Vt.shape}")

    # Estadísticas de valores singulares
    print(f"\n  📊 Valores Singulares:")
    print(f"    Mayor: {sigma.max():.2f}")
    print(f"    Menor: {sigma.min():.2f}")
    print(f"    Media: {sigma.mean():.2f}")

    return U, sigma, Vt


def predict_rating_svd(user_idx, product_idx, U, sigma, Vt):
    """
    Predice el rating usando SVD.

    Args:
        user_idx: Índice del usuario en la matriz
        product_idx: Índice del producto en la matriz
        U: Matriz de factores de usuarios
        sigma: Vector de valores singulares
        Vt: Matriz transpuesta de factores de productos

    Returns:
        float: Rating predicho (1-5)
    """
    try:
        user_factors = U[user_idx, :] * sigma
        product_factors = Vt[:, product_idx]
        predicted_rating = np.dot(user_factors, product_factors)
        return float(np.clip(predicted_rating, 1.0, 5.0))
    except Exception:
        return 3.0


def get_recommendations_svd(user_id, rating_matrix, U, sigma, Vt, n=5):
    """
    Genera recomendaciones personalizadas usando SVD.

    Args:
        user_id: ID del usuario
        rating_matrix: DataFrame (usuarios x productos)
        U, sigma, Vt: Factores SVD
        n: Número de recomendaciones

    Returns:
        pd.DataFrame: Top-N productos recomendados
    """
    try:
        user_idx = rating_matrix.index.get_loc(user_id)
    except KeyError:
        return pd.DataFrame(columns=['product_id', 'predicted_rating'])

    # Productos ya revisados
    already_rated = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()].index.tolist()

    predictions = {}
    for product_idx in range(rating_matrix.shape[1]):
        product_id = rating_matrix.columns[product_idx]

        if product_id in already_rated:
            continue

        pred = predict_rating_svd(user_idx, product_idx, U, sigma, Vt)
        if not np.isnan(pred) and 1.0 <= pred <= 5.0:
            predictions[product_id] = pred

    if not predictions:
        return pd.DataFrame(columns=['product_id', 'predicted_rating'])

    recommendations = pd.DataFrame(
        list(predictions.items()),
        columns=['product_id', 'predicted_rating']
    ).sort_values('predicted_rating', ascending=False)

    return recommendations.head(n)


def evaluate_svd(rating_matrix, U, sigma, Vt, n_samples=100):
    """
    Evalúa el modelo SVD.

    Args:
        rating_matrix: DataFrame (usuarios x productos)
        U, sigma, Vt: Factores SVD
        n_samples: Número de muestras

    Returns:
        dict: Métricas {RMSE, MAE, n_predictions, y_true, y_pred}
    """
    print_step("Evaluando modelo SVD")

    # Obtener pares con ratings conocidos
    known_ratings = []
    for user_id in rating_matrix.index:
        user_idx = rating_matrix.index.get_loc(user_id)
        rated = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()]
        for product_id in rated.index:
            product_idx = rating_matrix.columns.get_loc(product_id)
            known_ratings.append((user_idx, product_idx, rated[product_id]))

    np.random.seed(RANDOM_STATE)
    n_samples = min(n_samples, len(known_ratings))
    sample_indices = np.random.choice(len(known_ratings), size=n_samples, replace=False)
    samples = [known_ratings[i] for i in sample_indices]

    y_true = []
    y_pred = []

    for user_idx, product_idx, true_rating in samples:
        pred = predict_rating_svd(user_idx, product_idx, U, sigma, Vt)
        if not np.isnan(pred):
            y_true.append(true_rating)
            y_pred.append(pred)

    if len(y_true) == 0:
        print("  ⚠️  No hay suficientes datos para evaluar")
        return {'RMSE': None, 'MAE': None, 'n_predictions': 0}

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"\n  📈 RESULTADOS SVD:")
    print(f"    Predicciones realizadas: {len(y_true)}")
    print(f"    RMSE: {rmse:.4f}⭐ (error promedio ±{rmse:.2f} estrellas)")
    print(f"    MAE:  {mae:.4f}⭐")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'n_predictions': len(y_true),
        'y_true': y_true,
        'y_pred': y_pred
    }


def visualize_svd(sigma, U, Vt, rating_matrix):
    """
    Genera visualizaciones del modelo SVD.
    Guarda en reports/svd_analysis.png

    Args:
        sigma: Vector de valores singulares
        U: Matriz de factores de usuarios
        Vt: Matriz de factores de productos
        rating_matrix: DataFrame (usuarios x productos)
    """
    print_step("Generando visualizaciones SVD")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Valores singulares
    ax1 = axes[0, 0]
    ax1.plot(sigma, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Factor Latente')
    ax1.set_ylabel('Valor Singular')
    ax1.set_title('Importancia de Factores Latentes', fontweight='bold')
    ax1.grid(alpha=0.3)

    # 2. Varianza explicada acumulativa
    ax2 = axes[0, 1]
    variance_explained = (sigma ** 2) / (sigma ** 2).sum()
    cumulative_variance = np.cumsum(variance_explained)
    ax2.plot(cumulative_variance * 100, 'r-o', linewidth=2, markersize=4)
    ax2.axhline(90, color='green', linestyle='--', linewidth=2, label='90% varianza')
    ax2.axhline(95, color='orange', linestyle='--', linewidth=2, label='95% varianza')
    ax2.set_xlabel('Número de Factores')
    ax2.set_ylabel('Varianza Explicada Acumulativa (%)')
    ax2.set_title('Varianza Explicada Acumulativa', fontweight='bold')
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Usuarios en espacio latente (primeros 2 factores)
    ax3 = axes[1, 0]
    ax3.scatter(U[:, 0], U[:, 1], alpha=0.5, s=20, edgecolor='black', linewidth=0.3)
    ax3.set_xlabel(f'Factor Latente 1')
    ax3.set_ylabel(f'Factor Latente 2')
    ax3.set_title('Usuarios en Espacio Latente (2D)', fontweight='bold')
    ax3.grid(alpha=0.3)

    # 4. Productos en espacio latente (primeros 2 factores)
    ax4 = axes[1, 1]
    ax4.scatter(Vt[0, :], Vt[1, :], alpha=0.5, s=20, c='coral', edgecolor='black', linewidth=0.3)
    ax4.set_xlabel(f'Factor Latente 1')
    ax4.set_ylabel(f'Factor Latente 2')
    ax4.set_title('Productos en Espacio Latente (2D)', fontweight='bold')
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    output_path = REPORTS_DIR / 'svd_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {output_path.name}")


def run_svd_analysis(df, rating_matrix):
    """
    Ejecuta el pipeline completo de Matrix Factorization con SVD.

    Args:
        df: DataFrame con las reviews
        rating_matrix: DataFrame (usuarios x productos)

    Returns:
        dict: Resultados incluyendo U, sigma, Vt y métricas
    """
    print_header("🔢 MATRIX FACTORIZATION (SVD)")

    # 1. Preparar matriz sparse
    sparse_matrix = prepare_sparse_matrix(rating_matrix)

    # 2. Aplicar SVD
    U, sigma, Vt = apply_svd(sparse_matrix)

    # 3. Ejemplo de recomendaciones
    print_step("Ejemplo de Recomendaciones SVD")
    sample_users = rating_matrix.index[:3]
    for user_id in sample_users:
        recs = get_recommendations_svd(user_id, rating_matrix, U, sigma, Vt, n=5)
        user_ratings = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()]
        print(f"\n  👤 Usuario {user_id} ({len(user_ratings)} reviews, promedio {user_ratings.mean():.2f}⭐):")
        if len(recs) > 0:
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                stars = '⭐' * int(round(row['predicted_rating']))
                print(f"    {i}. {row['product_id']}: {row['predicted_rating']:.2f}/5.0 {stars}")
        else:
            print(f"    ⚠️  No hay recomendaciones disponibles")

    # 4. Evaluar
    eval_results = evaluate_svd(rating_matrix, U, sigma, Vt, n_samples=100)

    # 5. Visualizar
    visualize_svd(sigma, U, Vt, rating_matrix)

    # 6. Guardar modelo
    print_step("Guardando modelo SVD")
    model_data = {
        'U': U,
        'sigma': sigma,
        'Vt': Vt,
        'rating_matrix_index': rating_matrix.index.tolist(),
        'rating_matrix_columns': rating_matrix.columns.tolist(),
        'evaluation': eval_results,
        'k_factors': SVD_K_FACTORS
    }
    with open(SVD_MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  💾 Modelo guardado: {SVD_MODEL_FILE.name}")

    print(f"\n  ✅ SVD completado")

    return {
        'U': U,
        'sigma': sigma,
        'Vt': Vt,
        'evaluation': eval_results
    }


if __name__ == '__main__':
    from data_loader import load_data, prepare_data, get_user_item_matrix

    df = load_data()
    df_clean = prepare_data(df)
    rating_matrix = get_user_item_matrix(df_clean)
    results = run_svd_analysis(df_clean, rating_matrix)
