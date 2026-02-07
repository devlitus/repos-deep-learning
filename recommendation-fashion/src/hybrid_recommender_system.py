"""
Módulo de Sistema Híbrido de Recomendación
Combina User-Based CF, Item-Based CF y SVD para máxima precisión
"""

import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
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
    HYBRID_WEIGHT_USER_CF, HYBRID_WEIGHT_ITEM_CF, HYBRID_WEIGHT_SVD,
    RANDOM_STATE, REPORTS_DIR, MODELS_DIR, PLOT_STYLE,
    HYBRID_MODEL_FILE
)

# Importar funciones de predicción de los módulos base
from user_based_collaborative_filtering import predict_rating_user_based
from item_based_collaborative_filtering import predict_rating_item_based
from matrix_factorization_svd import predict_rating_svd

warnings.filterwarnings('ignore')

# Pesos por defecto
HYBRID_WEIGHTS = {
    'user_based': HYBRID_WEIGHT_USER_CF,
    'item_based': HYBRID_WEIGHT_ITEM_CF,
    'svd': HYBRID_WEIGHT_SVD
}


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(text):
    print(f"\n{'─' * 80}")
    print(f"📍 {text}")
    print("─" * 80)


def predict_rating_hybrid(user_id, product_id, rating_matrix,
                           user_sim, product_sim, U, sigma, Vt,
                           weights=None):
    """
    Predice rating usando sistema híbrido (promedio ponderado).

    Args:
        user_id: ID del usuario
        product_id: ID del producto
        rating_matrix: DataFrame (usuarios x productos)
        user_sim: DataFrame de similitud usuario-usuario
        product_sim: DataFrame de similitud producto-producto
        U, sigma, Vt: Factores SVD
        weights: dict con pesos para cada algoritmo

    Returns:
        float: Rating predicho (1-5)
    """
    if weights is None:
        weights = HYBRID_WEIGHTS.copy()

    try:
        # Predicción User-Based CF
        pred_user_based = predict_rating_user_based(
            user_id, product_id, rating_matrix, user_sim
        )

        # Predicción Item-Based CF
        pred_item_based = predict_rating_item_based(
            user_id, product_id, rating_matrix, product_sim
        )

        # Predicción SVD
        try:
            user_idx = rating_matrix.index.get_loc(user_id)
            product_idx = rating_matrix.columns.get_loc(product_id)
            pred_svd = predict_rating_svd(user_idx, product_idx, U, sigma, Vt)
        except Exception:
            pred_svd = 3.0

        # Promedio ponderado
        weighted_sum = (
            weights['user_based'] * pred_user_based +
            weights['item_based'] * pred_item_based +
            weights['svd'] * pred_svd
        )
        weight_total = sum(weights.values())
        final_prediction = weighted_sum / weight_total

        return float(np.clip(final_prediction, 1.0, 5.0))

    except Exception:
        return 3.0


def get_recommendations_hybrid(user_id, rating_matrix, user_sim, product_sim,
                                U, sigma, Vt, n=5, weights=None):
    """
    Genera recomendaciones usando sistema híbrido.

    Args:
        user_id: ID del usuario
        rating_matrix: DataFrame (usuarios x productos)
        user_sim: Similitud usuario-usuario
        product_sim: Similitud producto-producto
        U, sigma, Vt: Factores SVD
        n: Número de recomendaciones
        weights: Pesos del sistema híbrido

    Returns:
        pd.DataFrame: Top-N productos recomendados
    """
    if weights is None:
        weights = HYBRID_WEIGHTS.copy()

    # Productos ya revisados
    already_rated = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()].index.tolist()

    predictions = {}
    for product_id in rating_matrix.columns:
        if product_id in already_rated:
            continue

        pred = predict_rating_hybrid(
            user_id, product_id, rating_matrix,
            user_sim, product_sim, U, sigma, Vt, weights=weights
        )

        if not np.isnan(pred) and 1.0 <= pred <= 5.0:
            predictions[product_id] = pred

    if not predictions:
        return pd.DataFrame(columns=['product_id', 'predicted_rating'])

    recommendations = pd.DataFrame(
        list(predictions.items()),
        columns=['product_id', 'predicted_rating']
    ).sort_values('predicted_rating', ascending=False)

    return recommendations.head(n)


def optimize_weights(rating_matrix, user_sim, product_sim, U, sigma, Vt, n_samples=50):
    """
    Encuentra los mejores pesos para el sistema híbrido.

    Args:
        rating_matrix: DataFrame (usuarios x productos)
        user_sim, product_sim: Matrices de similitud
        U, sigma, Vt: Factores SVD
        n_samples: Muestras para evaluación

    Returns:
        dict: Mejores pesos encontrados
    """
    print_step("Optimizando pesos del sistema híbrido")

    # Obtener pares conocidos para evaluar
    known_ratings = []
    for user_id in rating_matrix.index:
        rated = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()]
        for product_id in rated.index:
            known_ratings.append((user_id, product_id, rated[product_id]))

    np.random.seed(RANDOM_STATE)
    n_samples = min(n_samples, len(known_ratings))
    sample_indices = np.random.choice(len(known_ratings), size=n_samples, replace=False)
    samples = [known_ratings[i] for i in sample_indices]

    # Precalcular predicciones individuales
    preds_ub = []
    preds_ib = []
    preds_svd = []
    y_true = []

    for user_id, product_id, true_rating in samples:
        # Ocultar rating
        original = rating_matrix.loc[user_id, product_id]
        rating_matrix.loc[user_id, product_id] = np.nan

        pred_ub = predict_rating_user_based(user_id, product_id, rating_matrix, user_sim)
        pred_ib = predict_rating_item_based(user_id, product_id, rating_matrix, product_sim)

        try:
            user_idx = rating_matrix.index.get_loc(user_id)
            product_idx = rating_matrix.columns.get_loc(product_id)
            pred_sv = predict_rating_svd(user_idx, product_idx, U, sigma, Vt)
        except Exception:
            pred_sv = 3.0

        # Restaurar
        rating_matrix.loc[user_id, product_id] = original

        if not any(np.isnan(x) for x in [pred_ub, pred_ib, pred_sv]):
            preds_ub.append(pred_ub)
            preds_ib.append(pred_ib)
            preds_svd.append(pred_sv)
            y_true.append(true_rating)

    if len(y_true) == 0:
        print("  ⚠️  Sin datos para optimizar, usando pesos por defecto")
        return HYBRID_WEIGHTS.copy()

    preds_ub = np.array(preds_ub)
    preds_ib = np.array(preds_ib)
    preds_svd = np.array(preds_svd)
    y_true = np.array(y_true)

    # Probar combinaciones
    weight_combinations = [
        {'user_based': 0.33, 'item_based': 0.33, 'svd': 0.34, 'name': 'Parejo (1/3-1/3-1/3)'},
        {'user_based': 0.5, 'item_based': 0.25, 'svd': 0.25, 'name': 'Enfasis User-Based'},
        {'user_based': 0.25, 'item_based': 0.5, 'svd': 0.25, 'name': 'Enfasis Item-Based'},
        {'user_based': 0.2, 'item_based': 0.2, 'svd': 0.6, 'name': 'Enfasis SVD'},
        {'user_based': 0.3, 'item_based': 0.3, 'svd': 0.4, 'name': 'Balanceado (30-30-40)'},
        {'user_based': 0.25, 'item_based': 0.25, 'svd': 0.5, 'name': 'SVD-Centric (25-25-50)'},
    ]

    best_rmse = float('inf')
    best_config = None
    results = []

    for config in weight_combinations:
        w_ub = config['user_based']
        w_ib = config['item_based']
        w_svd = config['svd']

        y_pred = np.clip(w_ub * preds_ub + w_ib * preds_ib + w_svd * preds_svd, 1.0, 5.0)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results.append({
            'name': config['name'],
            'RMSE': rmse,
            'MAE': mae,
            'weights': {'user_based': w_ub, 'item_based': w_ib, 'svd': w_svd}
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_config = {'user_based': w_ub, 'item_based': w_ib, 'svd': w_svd}

    # Mostrar resultados
    print(f"\n  📊 Resultados de optimización ({len(y_true)} muestras):\n")
    for r in results:
        marker = '🏆' if r['RMSE'] == best_rmse else '  '
        print(f"  {marker} {r['name']:<30s} RMSE: {r['RMSE']:.4f}  MAE: {r['MAE']:.4f}")

    print(f"\n  🏆 Mejor configuración: RMSE = {best_rmse:.4f}")

    return best_config


def evaluate_hybrid(rating_matrix, user_sim, product_sim, U, sigma, Vt,
                     weights=None, n_samples=100):
    """
    Evalúa el sistema híbrido.

    Args:
        rating_matrix: DataFrame
        user_sim, product_sim: Matrices de similitud
        U, sigma, Vt: Factores SVD
        weights: Pesos del híbrido
        n_samples: Número de muestras

    Returns:
        dict: Métricas {RMSE, MAE, n_predictions, y_true, y_pred}
    """
    print_step("Evaluando Sistema Híbrido")

    if weights is None:
        weights = HYBRID_WEIGHTS.copy()

    known_ratings = []
    for user_id in rating_matrix.index:
        rated = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()]
        for product_id in rated.index:
            known_ratings.append((user_id, product_id, rated[product_id]))

    np.random.seed(RANDOM_STATE)
    n_samples = min(n_samples, len(known_ratings))
    sample_indices = np.random.choice(len(known_ratings), size=n_samples, replace=False)
    samples = [known_ratings[i] for i in sample_indices]

    y_true = []
    y_pred = []

    for user_id, product_id, true_rating in samples:
        original = rating_matrix.loc[user_id, product_id]
        rating_matrix.loc[user_id, product_id] = np.nan

        pred = predict_rating_hybrid(
            user_id, product_id, rating_matrix,
            user_sim, product_sim, U, sigma, Vt, weights=weights
        )

        rating_matrix.loc[user_id, product_id] = original

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

    print(f"\n  📈 RESULTADOS Sistema Híbrido:")
    print(f"    Pesos: UB={weights['user_based']}, IB={weights['item_based']}, SVD={weights['svd']}")
    print(f"    Predicciones: {len(y_true)}")
    print(f"    RMSE: {rmse:.4f}⭐")
    print(f"    MAE:  {mae:.4f}⭐")

    return {
        'RMSE': rmse,
        'MAE': mae,
        'n_predictions': len(y_true),
        'y_true': y_true,
        'y_pred': y_pred
    }


def compare_all_methods(results_dict):
    """
    Compara todos los métodos de recomendación.

    Args:
        results_dict: dict con claves como nombres de algoritmos
                      y valores con 'RMSE' y 'MAE'

    Returns:
        pd.DataFrame: Tabla comparativa
    """
    print_step("Comparación de Todos los Métodos")

    rows = []
    for name, metrics in results_dict.items():
        if metrics.get('RMSE') is not None:
            rows.append({
                'Algoritmo': name,
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'Predicciones': metrics.get('n_predictions', 0)
            })

    if not rows:
        print("  ⚠️  Sin métricas para comparar")
        return pd.DataFrame()

    comparison_df = pd.DataFrame(rows).sort_values('RMSE')

    print(f"\n  ┌{'─'*30}┬{'─'*10}┬{'─'*10}┬{'─'*14}┐")
    print(f"  │ {'Algoritmo':<28s} │ {'RMSE':>8s} │ {'MAE':>8s} │ {'Predicciones':>12s} │")
    print(f"  ├{'─'*30}┼{'─'*10}┼{'─'*10}┼{'─'*14}┤")
    for _, row in comparison_df.iterrows():
        print(f"  │ {row['Algoritmo']:<28s} │ {row['RMSE']:>8.4f} │ {row['MAE']:>8.4f} │ {row['Predicciones']:>12} │")
    print(f"  └{'─'*30}┴{'─'*10}┴{'─'*10}┴{'─'*14}┘")

    best = comparison_df.iloc[0]
    print(f"\n  🏆 Mejor algoritmo: {best['Algoritmo']} (RMSE: {best['RMSE']:.4f})")

    return comparison_df


def visualize_hybrid(results, comparison):
    """
    Genera visualizaciones del sistema híbrido.
    Guarda en reports/hybrid_system.png

    Args:
        results: dict con y_true, y_pred
        comparison: DataFrame comparativo
    """
    print_step("Generando visualizaciones del sistema híbrido")

    try:
        plt.style.use(PLOT_STYLE)
    except:
        pass

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Comparación de RMSE
    ax1 = axes[0, 0]
    if len(comparison) > 0:
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
        bars = ax1.bar(comparison['Algoritmo'], comparison['RMSE'],
                       color=colors[:len(comparison)], edgecolor='black')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Comparación de RMSE', fontweight='bold')
        ax1.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, comparison['RMSE']):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax1.tick_params(axis='x', rotation=15)

    # 2. Comparación de MAE
    ax2 = axes[0, 1]
    if len(comparison) > 0:
        bars = ax2.bar(comparison['Algoritmo'], comparison['MAE'],
                       color=colors[:len(comparison)], edgecolor='black')
        ax2.set_ylabel('MAE')
        ax2.set_title('Comparación de MAE', fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        for bar, val in zip(bars, comparison['MAE']):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax2.tick_params(axis='x', rotation=15)

    # 3. Predicción Hybrid vs Real
    ax3 = axes[1, 0]
    if results.get('y_true') is not None and len(results['y_true']) > 0:
        ax3.scatter(results['y_true'], results['y_pred'], alpha=0.6, s=50,
                   edgecolor='black', color='gold')
        ax3.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfecto')
        ax3.set_xlabel('Rating Real')
        ax3.set_ylabel('Rating Predicho (Hybrid)')
        ax3.set_title('Predicciones Hybrid vs Reales', fontweight='bold')
        ax3.set_xlim(0.5, 5.5)
        ax3.set_ylim(0.5, 5.5)
        ax3.legend()
        ax3.grid(alpha=0.3)

    # 4. Distribución de errores
    ax4 = axes[1, 1]
    if results.get('y_true') is not None and len(results['y_true']) > 0:
        errors = results['y_pred'] - results['y_true']
        ax4.hist(errors, bins=15, color='gold', edgecolor='black', alpha=0.7)
        ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Error=0')
        ax4.axvline(errors.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Media: {errors.mean():.3f}')
        ax4.set_xlabel('Error (Predicción - Real)')
        ax4.set_ylabel('Frecuencia')
        ax4.set_title('Distribución de Errores (Hybrid)', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = REPORTS_DIR / 'hybrid_system.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Gráfico guardado: {output_path.name}")


def run_hybrid_system(df, rating_matrix, user_sim, product_sim, U, sigma, Vt):
    """
    Ejecuta el pipeline completo del sistema híbrido.

    Args:
        df: DataFrame con las reviews
        rating_matrix: DataFrame (usuarios x productos)
        user_sim: Similitud usuario-usuario
        product_sim: Similitud producto-producto
        U, sigma, Vt: Factores SVD

    Returns:
        dict: Resultados del sistema híbrido
    """
    print_header("🎯 SISTEMA HÍBRIDO DE RECOMENDACIÓN")

    # 1. Optimizar pesos
    best_weights = optimize_weights(
        rating_matrix, user_sim, product_sim, U, sigma, Vt, n_samples=50
    )

    # 2. Ejemplo de recomendaciones
    print_step("Ejemplo de Recomendaciones Híbridas")
    sample_users = rating_matrix.index[:3]
    for user_id in sample_users:
        recs = get_recommendations_hybrid(
            user_id, rating_matrix, user_sim, product_sim,
            U, sigma, Vt, n=5, weights=best_weights
        )
        user_ratings = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()]
        print(f"\n  👤 Usuario {user_id} ({len(user_ratings)} reviews, promedio {user_ratings.mean():.2f}⭐):")
        if len(recs) > 0:
            for i, (_, row) in enumerate(recs.iterrows(), 1):
                stars = '⭐' * int(round(row['predicted_rating']))
                print(f"    {i}. {row['product_id']}: {row['predicted_rating']:.2f}/5.0 {stars}")
        else:
            print(f"    ⚠️  No hay recomendaciones disponibles")

    # 3. Evaluar
    eval_results = evaluate_hybrid(
        rating_matrix, user_sim, product_sim, U, sigma, Vt,
        weights=best_weights, n_samples=100
    )

    # 4. Guardar modelo
    print_step("Guardando modelo Híbrido")
    model_data = {
        'weights': best_weights,
        'evaluation': eval_results,
        'user_sim_shape': user_sim.shape,
        'product_sim_shape': product_sim.shape,
        'U_shape': U.shape,
        'sigma_shape': sigma.shape,
        'Vt_shape': Vt.shape,
        'rating_matrix_index': rating_matrix.index.tolist(),
        'rating_matrix_columns': rating_matrix.columns.tolist()
    }
    with open(HYBRID_MODEL_FILE, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"  💾 Modelo guardado: {HYBRID_MODEL_FILE.name}")

    print(f"\n  ✅ Sistema Híbrido completado")

    return {
        'weights': best_weights,
        'evaluation': eval_results
    }


if __name__ == '__main__':
    from data_loader import load_data, prepare_data, get_user_item_matrix
    from user_based_collaborative_filtering import calculate_user_similarity
    from item_based_collaborative_filtering import calculate_product_similarity
    from matrix_factorization_svd import prepare_sparse_matrix, apply_svd

    df = load_data()
    df_clean = prepare_data(df)
    rating_matrix = get_user_item_matrix(df_clean)

    user_sim = calculate_user_similarity(rating_matrix)
    product_sim = calculate_product_similarity(rating_matrix)
    sparse_matrix = prepare_sparse_matrix(rating_matrix)
    U, sigma, Vt = apply_svd(sparse_matrix)

    results = run_hybrid_system(df_clean, rating_matrix, user_sim, product_sim, U, sigma, Vt)
