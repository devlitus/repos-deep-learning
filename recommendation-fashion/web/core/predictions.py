"""
Funciones de predicción para Collaborative Filtering
Incluye: User-Based CF, Item-Based CF, SVD, Hybrid
"""
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Asegurar que config está en el path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    HYBRID_WEIGHT_USER_CF,
    HYBRID_WEIGHT_ITEM_CF,
    HYBRID_WEIGHT_SVD
)


def predict_user_based(user_id, product_id, rating_matrix, user_sim_df, k=10):
    try:
        similar = user_sim_df[user_id].sort_values(ascending=False)[1:k+1]
        who_rated = similar[rating_matrix.loc[similar.index, product_id].notna()]
        if len(who_rated) == 0:
            avg = rating_matrix[product_id].mean()
            return float(avg) if not pd.isna(avg) else 3.0
        ratings = rating_matrix.loc[who_rated.index, product_id]
        sims = who_rated.values
        denom = np.sum(sims)
        if denom == 0:
            return 3.0
        return float(np.clip(np.sum(sims * ratings) / denom, 1.0, 5.0))
    except:
        return 3.0


def predict_item_based(user_id, product_id, rating_matrix, item_sim_df, k=10):
    try:
        user_ratings = rating_matrix.loc[user_id]
        rated = user_ratings[user_ratings.notna()].index.tolist()
        if not rated:
            avg = rating_matrix[product_id].mean()
            return float(avg) if not pd.isna(avg) else 3.0
        sims = item_sim_df.loc[product_id, rated].nlargest(k)
        if len(sims) == 0 or sims.sum() == 0:
            avg = rating_matrix[product_id].mean()
            return float(avg) if not pd.isna(avg) else 3.0
        r = user_ratings[sims.index]
        return float(np.clip((sims.values * r.values).sum() / sims.values.sum(), 1.0, 5.0))
    except:
        return 3.0


def predict_svd(user_id, product_id, rating_matrix, U, sigma, Vt):
    try:
        user_idx = rating_matrix.index.get_loc(user_id)
        product_idx = rating_matrix.columns.get_loc(product_id)
        pred = np.dot(U[user_idx, :] * sigma, Vt[:, product_idx])
        return float(np.clip(pred, 1.0, 5.0))
    except:
        return 3.0


def predict_hybrid(user_id, product_id, rating_matrix, user_sim_df, item_sim_df,
                   U, sigma, Vt, weights=None):
    if weights is None:
        weights = {'user_based': HYBRID_WEIGHT_USER_CF,
                   'item_based': HYBRID_WEIGHT_ITEM_CF,
                   'svd': HYBRID_WEIGHT_SVD}

    p_ub = predict_user_based(user_id, product_id, rating_matrix, user_sim_df)
    p_ib = predict_item_based(user_id, product_id, rating_matrix, item_sim_df)
    p_svd = predict_svd(user_id, product_id, rating_matrix, U, sigma, Vt)

    total = sum(weights.values())
    pred = (weights['user_based'] * p_ub + weights['item_based'] * p_ib + weights['svd'] * p_svd) / total
    return float(np.clip(pred, 1.0, 5.0))


def get_recommendations(user_id, rating_matrix, user_sim_df, item_sim_df,
                        U, sigma, Vt, algorithm, n=10):
    """Generar recomendaciones usando el algoritmo seleccionado"""
    already_rated = rating_matrix.loc[user_id][rating_matrix.loc[user_id].notna()].index.tolist()

    predictions = {}
    for product_id in rating_matrix.columns:
        if product_id in already_rated:
            continue
        if algorithm == 'User-Based CF':
            pred = predict_user_based(user_id, product_id, rating_matrix, user_sim_df)
        elif algorithm == 'Item-Based CF':
            pred = predict_item_based(user_id, product_id, rating_matrix, item_sim_df)
        elif algorithm == 'SVD':
            pred = predict_svd(user_id, product_id, rating_matrix, U, sigma, Vt)
        else:  # Hybrid
            pred = predict_hybrid(user_id, product_id, rating_matrix, user_sim_df,
                                  item_sim_df, U, sigma, Vt)
        predictions[product_id] = pred

    if not predictions:
        return pd.DataFrame(columns=['product_id', 'predicted_rating'])

    recs = pd.DataFrame(
        list(predictions.items()),
        columns=['product_id', 'predicted_rating']
    ).sort_values('predicted_rating', ascending=False).head(n)

    return recs
