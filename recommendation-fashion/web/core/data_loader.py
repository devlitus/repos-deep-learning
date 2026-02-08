"""
Funciones de carga y preparación de datos
Todas las funciones usan @st.cache_data para optimizar rendimiento
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import sys

# Asegurar que config está en el path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import (
    DATASET_FILE, COL_USER_ID, COL_PRODUCT_ID, COL_RATING,
    MIN_USER_RATINGS, MIN_PRODUCT_RATINGS, SVD_K_FACTORS
)


@st.cache_data
def load_and_prepare_data():
    """Cargar y preprocesar datos de Fashion Reviews"""
    if not DATASET_FILE.exists():
        st.error(f"Dataset no encontrado: {DATASET_FILE}")
        st.info("Ejecuta primero: `python generate_large_dataset.py`")
        st.stop()

    reviews = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                reviews.append(json.loads(line))
            except:
                continue

    if len(reviews) == 0:
        st.error("El dataset está vacío.")
        st.info("Ejecuta: `python generate_large_dataset.py`")
        st.stop()

    df = pd.DataFrame(reviews)
    df = df.drop_duplicates(subset=[COL_USER_ID, COL_PRODUCT_ID], keep='first')

    # Ajustar filtros dinámicamente si el dataset es pequeño
    min_user = MIN_USER_RATINGS
    min_product = MIN_PRODUCT_RATINGS
    if len(df) < 1000:
        min_user = 1
        min_product = 1

    user_counts = df[COL_USER_ID].value_counts()
    valid_users = user_counts[user_counts >= min_user].index
    df = df[df[COL_USER_ID].isin(valid_users)]

    product_counts = df[COL_PRODUCT_ID].value_counts()
    valid_products = product_counts[product_counts >= min_product].index
    df = df[df[COL_PRODUCT_ID].isin(valid_products)]

    if len(df) == 0:
        st.error(
            f"⚠️ El dataset tiene muy pocos datos ({len(reviews)} reviews). "
            f"Después de filtrar usuarios/productos con pocas reviews, no queda nada."
        )
        st.info(
            "Regenera el dataset ejecutando en tu terminal:\n\n"
            "```\ncd recommendation-fashion\npython generate_large_dataset.py --size 100000\n```"
        )
        st.stop()

    rating_matrix = df.pivot_table(
        index=COL_USER_ID,
        columns=COL_PRODUCT_ID,
        values=COL_RATING,
        aggfunc='mean'
    )

    return df, rating_matrix


@st.cache_data
def compute_user_similarity(_rating_matrix):
    """Calcular similitud usuario-usuario"""
    filled = _rating_matrix.fillna(0)
    sim = cosine_similarity(filled)
    return pd.DataFrame(sim, index=_rating_matrix.index, columns=_rating_matrix.index)


@st.cache_data
def compute_item_similarity(_rating_matrix):
    """Calcular similitud producto-producto"""
    filled_t = _rating_matrix.T.fillna(0)
    sim = cosine_similarity(filled_t.values)
    return pd.DataFrame(sim, index=_rating_matrix.columns, columns=_rating_matrix.columns)


@st.cache_data
def compute_svd(_rating_matrix, k=50):
    """Calcular SVD"""
    filled = _rating_matrix.fillna(0).astype(np.float32)
    sparse = csr_matrix(filled.values)
    k = min(k, min(sparse.shape) - 1)
    U, sigma, Vt = svds(sparse, k=k)
    return U, sigma, Vt


@st.cache_data
def prepare_ncf_data(_df):
    """Preparar datos para Neural Collaborative Filtering con Train/Val/Test split"""
    from config import COL_USER_ID, COL_PRODUCT_ID, COL_RATING

    ncf_df = _df[[COL_USER_ID, COL_PRODUCT_ID, COL_RATING]].copy()
    ncf_df.columns = ['user_id', 'product_id', 'rating']

    # 1. Split PRIMERO (70% train, 15% val, 15% test)
    temp_df, test_df = train_test_split(ncf_df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(temp_df, test_size=0.176, random_state=42)

    # 2. Crear mappings SOLO desde train
    train_user_ids = train_df['user_id'].unique()
    train_product_ids = train_df['product_id'].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(train_user_ids)}
    product_to_idx = {pid: idx for idx, pid in enumerate(train_product_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}

    # 3. Aplicar mappings a train
    train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
    train_df['product_idx'] = train_df['product_id'].map(product_to_idx)

    # 4. Aplicar mappings a val/test y filtrar usuarios/productos no vistos en train
    val_df['user_idx'] = val_df['user_id'].map(user_to_idx)
    val_df['product_idx'] = val_df['product_id'].map(product_to_idx)
    val_before = len(val_df)
    val_df = val_df.dropna(subset=['user_idx', 'product_idx'])
    val_df['user_idx'] = val_df['user_idx'].astype(int)
    val_df['product_idx'] = val_df['product_idx'].astype(int)

    test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
    test_df['product_idx'] = test_df['product_id'].map(product_to_idx)
    test_before = len(test_df)
    test_df = test_df.dropna(subset=['user_idx', 'product_idx'])
    test_df['user_idx'] = test_df['user_idx'].astype(int)
    test_df['product_idx'] = test_df['product_idx'].astype(int)

    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'n_users': len(train_user_ids),
        'n_products': len(train_product_ids),
        'user_to_idx': user_to_idx,
        'product_to_idx': product_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_product': idx_to_product,
        'total_interactions': len(ncf_df),
        'val_filtered': val_before - len(val_df),
        'test_filtered': test_before - len(test_df)
    }
