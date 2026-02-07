"""
Módulo de Item-Based Collaborative Filtering
Recomienda productos basándose en la similitud entre productos
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
    ITEM_CF_K_NEIGHBORS, RANDOM_STATE,
    REPORTS_DIR, MODELS_DIR, PLOT_STYLE,
    REPORT_ITEM_SIMILARITY_DIST, REPORT_ITEM_BASED_PREDICTIONS,
    ITEM_SIMILARITY_FILE
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


def calculate_product_similarity(rating_matrix):
    """
    Calcula la matriz de similitud coseno entre productos.
    Transpone la matriz antes de calcular (productos como filas).

    Args:
        rating_matrix: DataFrame (usuarios x productos) con ratings

    Returns:
        pd.DataFrame: Matriz de similitud producto-producto
    """
    print_step("Calculando similitud entre productos")

    # Transponer: productos como filas, usuarios como columnas
    rating_matrix_T = rating_matrix.T.fillna(0)

    # Calcular similitud coseno
    product_sim = cosine_similarity(rating_matrix_T.values)

    # Convertir a DataFrame
    product_similarity_df = pd.DataFrame(
        product_sim,
        index=rating_matrix.columns,
        columns=rating_matrix.columns
    )

    # Estadísticas
    similarities_no_diag = product_sim[np.triu_indices_from(product_sim, k=1)]
    print(f"  ✅ Matriz de similitud: {product_similarity_df.shape}")
    print(f"  Similitud media: {similarities_no_diag.mean():.4f}")
    print(f"  Similitud máxima: {similarities_no_diag.max():.4f}")

    return product_similarity_df


def predict_rating_item_based(user_id, product_id, rating_matrix, similarity_df, k=None):
    """
    Predice el rating que un usuario daría a un producto usando Item-Based CF.

    Args:
        user_id: ID del usuario
        product_id: ID del producto
        rating_matrix: DataFrame (usuarios x productos)
        similarity_df: DataFrame de similitud producto-producto
        k: Número de productos similares (default: config.ITEM_CF_K_NEIGHBORS)

    Returns:
        float: Rating predicho (1-5)
    """
    if k is None:
        k = ITEM_CF_K_NEIGHBORS

    try:
        # Obtener ratings del usuario
        user_ratings = rating_matrix.loc[user_id]

        # Productos que el usuario ha revisado
        rated_items = user_ratings[user_ratings.notna()].index.tolist()

        if len(rated_items) == 0:
            avg = rating_matrix[product_id].mean()
            return float(avg) if not pd.isna(avg) else 3.0

        # Similitud del producto target con los que el usuario revisó
        sims = similarity_df.loc[product_id, rated_items]

        # Top-K similares
        top_sims = sims.nlargest(k)

        if len(top_sims) == 0:
            avg = rating_matrix[product_id].mean()
            return float(avg) if not pd.isna(avg) else 3.0

        # Ratings del usuario en esos productos
        top_ratings = user_ratings[top_sims.index]

        # Calcular promedio ponderado
        numerator = (top_sims.values * top_ratings.values).sum()
        denominator = top_sims.values.sum()

        if denominator == 0:
            avg = rating_matrix[product_id].mean()
            return float(avg) if not pd.isna(avg) else 3.0

        prediction = float(numerator / denominator)
        return float(np.clip(prediction, 1.0, 5.0))

    except Exception:
        return 3.0


def get_recommendations_item_based(user_id, rating_matrix, similarity_df, n=5, k=None):
    """
    Genera recomendaciones personalizadas para un usuario.

    Args:
        user_id: ID del usuario
        rating_matrix: DataFrame (usuarios x productos)
        similarity_df: DataFrame de similitud producto-producto
        n: Número de recomendaciones
        k: Número de productos similares

    Returns:
        pd.DataFrame: Top-N productos recomendados con predicted_rating
    """
    if k is None:
        k = ITEM_CF_K_NEIGHBORS

    # Productos no revisados
    user_ratings = rating_matrix.loc[user_id]
    not_rated = user_ratings[user_ratings.isna()].index.tolist()

    predictions = {}
    for item in not_rated:
        pred = predict_rating_item_based(user_id, item, rating_matrix, similarity_df, k=k)
        predictions[item] = float(pred)

    if not predictions:
        return pd.DataFrame(columns=['product_id', 'predicted_rating'])

    recommendations = pd.DataFrame(
        list(predictions.items()),
        columns=['product_id', 'predicted_rating']
    ).sort_values('predicted_rating', ascending=False).head(n)

    return recommendations


def evaluate_item_based(rating_matrix, similarity_df, n_samples=100):
    """
    Evalúa el modelo Item-Based CF usando validación.

    Args:
        rating_matrix: DataFrame (usuarios x productos)
        similarity_df: DataFrame de similitud producto-producto
        n_samples: Número de muestras para evaluación

    Returns:
        dict: Métricas {RMSE, MAE, n_predictions, y_true, y_pred}
    """
    print_step("Evaluando Item-Based Collaborative Filtering")

    # Obtener pares (usuario, producto) con ratings conocidos
    known_ratings = []
    for user_id in rating_matrix.index:
        rated_products = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()].index
        for product_id in rated_products:
            known_ratings.append((user_id, product_id, rating_matrix.loc[user_id, product_id]))

    # Seleccionar muestras aleatorias
    np.random.seed(RANDOM_STATE)
    n_samples = min(n_samples, len(known_ratings))
    sample_indices = np.random.choice(len(known_ratings), size=n_samples, replace=False)
    samples = [known_ratings[i] for i in sample_indices]

    y_true = []
    y_pred = []

    for user_id, product_id, true_rating in samples:
        # Temporalmente ocultar el rating
        original_value = rating_matrix.loc[user_id, product_id]
        rating_matrix.loc[user_id, product_id] = np.nan

        # Predecir
        pred = predict_rating_item_based(user_id, product_id, rating_matrix, similarity_df)

        # Restaurar
        rating_matrix.loc[user_id, product_id] = original_value

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

    print(f"\n  📈 RESULTADOS Item-Based CF:")
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


def visualize_item_based(results):
    """
    Genera visualizaciones del modelo Item-Based CF.
    Guarda en reports/item_based_cf.png

    Args:
        results: dict con y_true, y_pred del evaluate_item_based
    """
    print_step("Generando visualizaciones Item-Based CF")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    if results.get('y_true') is None or len(results['y_true']) == 0:
        print("  ⚠️  Sin datos para visualizar")
        return

    y_true = results['y_true']
    y_pred = results['y_pred']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: Predicción vs Real
    ax1 = axes[0]
    ax1.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolor='black')
    ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predicción Perfecta')
    ax1.set_xlabel('Rating Real')
    ax1.set_ylabel('Rating Predicho')
    ax1.set_title('Item-Based CF: Predicción vs Realidad', fontweight='bold')
    ax1.set_xlim(0.5, 5.5)
    ax1.set_ylim(0.5, 5.5)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Histograma de errores
    ax2 = axes[1]
    errors = y_pred - y_true
    ax2.hist(errors, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax2.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                label=f'Error medio: {errors.mean():.3f}')
    ax2.set_xlabel('Error (Predicción - Real)')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Distribución de Errores', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(REPORT_ITEM_BASED_PREDICTIONS, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {REPORT_ITEM_BASED_PREDICTIONS.name}")


def run_item_based_cf(df, rating_matrix):
    """
    Ejecuta el pipeline completo de Item-Based Collaborative Filtering.

    Args:
        df: DataFrame con las reviews
        rating_matrix: DataFrame (usuarios x productos)

    Returns:
        dict: Resultados incluyendo similarity_df y métricas
    """
    print_header("👕 ITEM-BASED COLLABORATIVE FILTERING")

    # 1. Calcular similitud
    product_similarity_df = calculate_product_similarity(rating_matrix)

    # 2. Ejemplo de recomendaciones
    print_step("Ejemplo de Recomendaciones")
    sample_users = rating_matrix.index[:3]
    for user_id in sample_users:
        recs = get_recommendations_item_based(user_id, rating_matrix, product_similarity_df, n=5)
        user_ratings = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()]
        print(f"\n  👤 Usuario {user_id} ({len(user_ratings)} reviews, promedio {user_ratings.mean():.2f}⭐):")
        if len(recs) > 0:
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                stars = '⭐' * int(round(row['predicted_rating']))
                print(f"    {i}. {row['product_id']}: {row['predicted_rating']:.2f}/5.0 {stars}")
        else:
            print(f"    ⚠️  No hay recomendaciones disponibles")

    # 3. Evaluar
    eval_results = evaluate_item_based(rating_matrix, product_similarity_df, n_samples=100)

    # 4. Visualizar
    visualize_item_based(eval_results)

    # 5. Guardar modelo
    print_step("Guardando modelo Item-Based CF")
    model_data = {
        'similarity_matrix': product_similarity_df,
        'rating_matrix_index': rating_matrix.index.tolist(),
        'rating_matrix_columns': rating_matrix.columns.tolist(),
        'evaluation': eval_results,
        'k_neighbors': ITEM_CF_K_NEIGHBORS
    }
    with open(ITEM_SIMILARITY_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  💾 Modelo guardado: {ITEM_SIMILARITY_FILE.name}")

    print(f"\n  ✅ Item-Based CF completado")

    return {
        'similarity_df': product_similarity_df,
        'evaluation': eval_results
    }


if __name__ == '__main__':
    from data_loader import load_data, prepare_data, get_user_item_matrix

    df = load_data()
    df_clean = prepare_data(df)
    rating_matrix = get_user_item_matrix(df_clean)
    results = run_item_based_cf(df_clean, rating_matrix)
