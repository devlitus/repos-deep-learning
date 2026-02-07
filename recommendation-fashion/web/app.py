"""
Aplicación Web - Sistema de Recomendación de Moda
Interfaz interactiva con Streamlit para Amazon Fashion Reviews
Incluye Laboratorio de Entrenamiento NCF con ajuste de hiperparámetros
"""
import sys
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en el path
WEB_DIR = Path(__file__).parent
PROJECT_DIR = WEB_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / 'src'))

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import plotly.express as px
import plotly.graph_objects as go
import json
import time

from config import (
    DATASET_FILE, COL_USER_ID, COL_PRODUCT_ID, COL_RATING,
    COL_REVIEW_TEXT, COL_SUMMARY,
    MIN_USER_RATINGS, MIN_PRODUCT_RATINGS,
    SVD_K_FACTORS, STREAMLIT_PAGE_CONFIG, MODELS_DIR,
    HYBRID_WEIGHT_USER_CF, HYBRID_WEIGHT_ITEM_CF, HYBRID_WEIGHT_SVD
)

# Verificar si PyTorch está disponible
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    pass

# Configuración de la página
st.set_page_config(**STREAMLIT_PAGE_CONFIG)

# Título principal
st.title("👕 Sistema de Recomendación de Moda")
st.markdown("**Amazon Fashion Reviews - Sistema Híbrido (User-CF + Item-CF + SVD + NCF)**")

# Información de modelos
with st.expander("ℹ️ Información Técnica - Modelos Utilizados", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔧 Modelos Tradicionales (calculados on-the-fly):**
        - **User-Based CF** → Matriz de similitud en memoria
        - **Item-Based CF** → Matriz de similitud en memoria
        - **SVD (Matrix Factorization)** → Descomposición en valores singulares
        - **Sistema Híbrido** → Combinación ponderada de los 3 anteriores

        > 📝 Estos modelos se calculan al cargar la aplicación desde el dataset raw JSON.
        > No utilizan archivos `.pkl` pre-entrenados.
        """)

    with col2:
        st.markdown("""
        **🧠 Deep Learning (PyTorch):**
        - **NCF (Neural Collaborative Filtering)** → Red neuronal entrenada on-demand
        - Arquitectura: Embeddings + MLP (Multi-Layer Perceptron)
        - Entrenamiento: En el "Laboratorio NCF" con hiperparámetros ajustables

        > 📝 El modelo NCF se entrena desde cero en cada sesión.
        > Opcionalmente puede guardar/cargar pesos en formato `.pth` (PyTorch).
        """)

    st.info("""
    💡 **¿Por qué no usar archivos `.pkl`?**
    Esta app prioriza transparencia educativa calculando modelos en tiempo real desde datos raw.
    Los notebooks sí guardan modelos en `.pkl` (SVD, similitudes) y `.pth` (NCF) para reutilización.
    """)

st.divider()

# =============================================================================
# INICIALIZAR SESSION STATE
# =============================================================================

if 'experiments' not in st.session_state:
    st.session_state.experiments = []
if 'current_history' not in st.session_state:
    st.session_state.current_history = None
if 'training_done' not in st.session_state:
    st.session_state.training_done = False

# =============================================================================
# CLASES Y FUNCIONES DE NCF (PYTORCH)
# =============================================================================

if PYTORCH_AVAILABLE:
    class NCFDataset(Dataset):
        """Dataset personalizado para PyTorch"""
        def __init__(self, user_indices, product_indices, ratings):
            self.users = torch.LongTensor(user_indices)
            self.products = torch.LongTensor(product_indices)
            self.ratings = torch.FloatTensor(ratings)

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
            return self.users[idx], self.products[idx], self.ratings[idx]

    class NeuralCollaborativeFiltering(nn.Module):
        """Red Neuronal para Sistema de Recomendación con mejoras"""
        def __init__(self, n_users, n_products, embedding_dim=64,
                     hidden_layers=None, dropout=0.2, use_batch_norm=True):
            super(NeuralCollaborativeFiltering, self).__init__()
            if hidden_layers is None:
                hidden_layers = [128, 64, 32]

            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.product_embedding = nn.Embedding(n_products, embedding_dim)

            input_dim = embedding_dim * 2
            layers = []
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)

            self._init_weights()

        def _init_weights(self):
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.product_embedding.weight, std=0.01)
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, user_indices, product_indices):
            user_emb = self.user_embedding(user_indices)
            product_emb = self.product_embedding(product_indices)
            x = torch.cat([user_emb, product_emb], dim=-1)
            output = self.mlp(x)
            output = torch.sigmoid(output) * 4 + 1  # Rango [1, 5]
            return output.squeeze()

    class EarlyStopping:
        """Early Stopping para prevenir overfitting"""
        def __init__(self, patience=3, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            self.best_epoch = 0

        def __call__(self, val_loss, epoch):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_epoch = epoch
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0

# =============================================================================
# CARGA DE DATOS Y MODELOS (con cache)
# =============================================================================

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
    ncf_df = _df[[COL_USER_ID, COL_PRODUCT_ID, COL_RATING]].copy()
    ncf_df.columns = ['user_id', 'product_id', 'rating']

    user_ids = ncf_df['user_id'].unique()
    product_ids = ncf_df['product_id'].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
    product_to_idx = {pid: idx for idx, pid in enumerate(product_ids)}
    idx_to_user = {idx: uid for uid, idx in user_to_idx.items()}
    idx_to_product = {idx: pid for pid, idx in product_to_idx.items()}

    ncf_df['user_idx'] = ncf_df['user_id'].map(user_to_idx)
    ncf_df['product_idx'] = ncf_df['product_id'].map(product_to_idx)

    # ✨ MEJORADO: Split en Train (70%) / Validation (15%) / Test (15%)
    # Primer split: separar test set (15%)
    temp_df, test_df = train_test_split(ncf_df, test_size=0.15, random_state=42)

    # Segundo split: dividir temp en train y validation
    # 0.15 / 0.85 ≈ 0.176 para obtener 15% del total
    train_df, val_df = train_test_split(temp_df, test_size=0.176, random_state=42)

    return {
        'train_df': train_df,
        'val_df': val_df,  # ✅ Nuevo: conjunto de validación separado
        'test_df': test_df,
        'n_users': len(user_ids),
        'n_products': len(product_ids),
        'user_to_idx': user_to_idx,
        'product_to_idx': product_to_idx,
        'idx_to_user': idx_to_user,
        'idx_to_product': idx_to_product,
        'total_interactions': len(ncf_df)
    }


# Funciones de predicción (collaborative filtering)
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


# Cargar datos
with st.spinner('Cargando datos y entrenando modelos...'):
    df, rating_matrix = load_and_prepare_data()
    user_sim_df = compute_user_similarity(rating_matrix)
    item_sim_df = compute_item_similarity(rating_matrix)
    U, sigma, Vt = compute_svd(rating_matrix, k=SVD_K_FACTORS)

st.success(f'✅ Modelos cargados: {rating_matrix.shape[0]} usuarios, {rating_matrix.shape[1]} productos')
st.caption("📊 User-CF, Item-CF y SVD calculados en memoria (sin usar archivos .pkl)")

# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

st.sidebar.header("⚙️ Configuración")

modes = ["👤 Usuario Existente", "🆕 Usuario Nuevo", "📊 Estadísticas del Sistema"]
if PYTORCH_AVAILABLE:
    modes.append("🧠 Laboratorio NCF")

mode = st.sidebar.radio("Selecciona modo:", modes)

st.sidebar.divider()

# =============================================================================
# MODO 1: USUARIO EXISTENTE
# =============================================================================

if mode == "👤 Usuario Existente":
    st.header("👤 Recomendaciones para Usuario Existente")

    # Selector de usuario
    user_id = st.selectbox(
        "Selecciona un usuario:",
        options=sorted(rating_matrix.index.tolist()),
        index=0
    )

    num_ratings = rating_matrix.loc[user_id].notna().sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Usuario ID", user_id)
    with col2:
        st.metric("Productos calificados", num_ratings)
    with col3:
        avg_rating = rating_matrix.loc[user_id].dropna().mean()
        st.metric("Rating promedio", f"{avg_rating:.2f}")

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["👕 Recomendaciones", "⭐ Historial", "📈 Análisis"])

    with tab1:
        st.subheader("👕 Recomendaciones Personalizadas")

        col_algo, col_num = st.columns(2)
        with col_algo:
            algorithm = st.selectbox(
                "Algoritmo:",
                ["Hybrid (Recomendado)", "User-Based CF", "Item-Based CF", "SVD"]
            )
        with col_num:
            num_recs = st.slider("Recomendaciones:", 5, 20, 10)

        if st.button("Generar Recomendaciones", type="primary"):
            with st.spinner("Generando recomendaciones..."):
                recs = get_recommendations(
                    user_id, rating_matrix, user_sim_df, item_sim_df,
                    U, sigma, Vt, algorithm, n=num_recs
                )

                if len(recs) > 0:
                    for idx, (_, row) in enumerate(recs.iterrows(), 1):
                        col_r1, col_r2 = st.columns([3, 1])
                        with col_r1:
                            st.markdown(f"**{idx}. Producto {row['product_id']}**")
                        with col_r2:
                            st.metric("Rating predicho", f"{row['predicted_rating']:.2f} ⭐")
                        st.divider()

                    fig = px.bar(
                        recs,
                        x='predicted_rating',
                        y='product_id',
                        orientation='h',
                        title=f'Top {len(recs)} Recomendaciones ({algorithm})',
                        labels={'predicted_rating': 'Rating Predicho', 'product_id': 'Producto'},
                        color='predicted_rating',
                        color_continuous_scale='RdYlGn',
                        range_color=[1, 5]
                    )
                    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No hay productos nuevos para recomendar a este usuario.")

    with tab2:
        st.subheader("⭐ Historial de Reviews")

        user_ratings = rating_matrix.loc[user_id].dropna().sort_values(ascending=False)

        if len(user_ratings) > 0:
            history_df = pd.DataFrame({
                'Producto': user_ratings.index,
                'Rating': user_ratings.values
            })

            fig = px.bar(
                history_df,
                x='Rating',
                y='Producto',
                orientation='h',
                title='Productos Calificados por el Usuario',
                color='Rating',
                color_continuous_scale='RdYlGn',
                range_color=[1, 5]
            )
            fig.update_layout(height=max(300, len(history_df) * 30),
                            yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("Este usuario no tiene reviews.")

    with tab3:
        st.subheader("📈 Análisis del Perfil")

        user_ratings_series = rating_matrix.loc[user_id].dropna()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rating Promedio", f"{user_ratings_series.mean():.2f}")
        with col2:
            st.metric("Productos Revisados", len(user_ratings_series))
        with col3:
            coverage = (len(user_ratings_series) / len(rating_matrix.columns)) * 100
            st.metric("Cobertura", f"{coverage:.2f}%")

        if len(user_ratings_series) > 0:
            rating_counts = user_ratings_series.value_counts().sort_index()

            fig = px.bar(
                x=rating_counts.index.astype(str),
                y=rating_counts.values,
                title='Distribución de tus Ratings',
                labels={'x': 'Rating', 'y': 'Cantidad'},
                color=rating_counts.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MODO 2: USUARIO NUEVO (COLD START)
# =============================================================================

elif mode == "🆕 Usuario Nuevo":
    st.header("🆕 Recomendaciones para Usuario Nuevo")

    st.info(
        "👋 Como no tenemos historial de este usuario, recomendamos los "
        "productos más populares y mejor calificados."
    )

    col1, col2 = st.columns(2)
    with col1:
        num_recs = st.slider("Cantidad de recomendaciones:", 5, 20, 10)
    with col2:
        min_reviews = st.slider("Mínimo de reviews (confiabilidad):", 3, 20, 5)

    if st.button("Ver Recomendaciones", type="primary"):
        product_stats = df.groupby(COL_PRODUCT_ID).agg({
            COL_RATING: ['mean', 'count']
        }).reset_index()
        product_stats.columns = ['product_id', 'avg_rating', 'num_ratings']

        popular = product_stats[product_stats['num_ratings'] >= min_reviews]
        popular = popular.sort_values('avg_rating', ascending=False).head(num_recs)

        st.subheader("👕 Productos Populares y Mejor Calificados")

        for idx, row in popular.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**Producto {row['product_id']}**")
            with col2:
                st.metric("Rating", f"{row['avg_rating']:.2f} ⭐")
            with col3:
                st.metric("Reviews", int(row['num_ratings']))
            st.divider()

        fig = px.scatter(
            popular,
            x='num_ratings',
            y='avg_rating',
            text='product_id',
            title='Popularidad vs Calidad',
            labels={'num_ratings': 'Número de Reviews', 'avg_rating': 'Rating Promedio'},
            size='avg_rating',
            color='avg_rating',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5]
        )
        fig.update_traces(textposition='top center', textfont_size=8)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏆 Top Rated (mejores ratings)")
        top_rated = product_stats[product_stats['num_ratings'] >= min_reviews]
        top_rated = top_rated.sort_values(['avg_rating', 'num_ratings'], ascending=[False, False]).head(num_recs)

        st.dataframe(
            top_rated.rename(columns={
                'product_id': 'Producto',
                'avg_rating': 'Rating Promedio',
                'num_ratings': 'Total Reviews'
            }),
            use_container_width=True,
            hide_index=True
        )

# =============================================================================
# MODO 3: ESTADÍSTICAS DEL SISTEMA
# =============================================================================

elif mode == "📊 Estadísticas del Sistema":
    st.header("📊 Estadísticas del Sistema")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Usuarios", rating_matrix.shape[0])
    with col2:
        st.metric("👕 Productos", rating_matrix.shape[1])
    with col3:
        st.metric("⭐ Reviews", len(df))
    with col4:
        sparsity = (rating_matrix.isna().sum().sum() /
                   (rating_matrix.shape[0] * rating_matrix.shape[1]) * 100)
        st.metric("🕳️ Sparsity", f"{sparsity:.1f}%")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribuciones", "👕 Top Productos", "👥 Usuarios Activos", "🔧 Modelos Técnicos"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Distribución de Ratings")
            rating_dist = df[COL_RATING].value_counts().sort_index()
            fig = px.bar(
                x=rating_dist.index.astype(str),
                y=rating_dist.values,
                labels={'x': 'Rating', 'y': 'Frecuencia'},
                color=rating_dist.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Reviews por Usuario")
            ratings_per_user = rating_matrix.notna().sum(axis=1)
            fig = px.histogram(
                x=ratings_per_user,
                nbins=30,
                labels={'x': 'Número de Reviews', 'y': 'Número de Usuarios'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🏆 Productos Más Calificados")

        product_stats = df.groupby(COL_PRODUCT_ID).agg({
            COL_RATING: ['mean', 'count']
        }).reset_index()
        product_stats.columns = ['product_id', 'avg_rating', 'num_ratings']
        top_products = product_stats.nlargest(20, 'num_ratings')

        fig = px.bar(
            top_products,
            x='num_ratings',
            y='product_id',
            orientation='h',
            title='Top 20 Productos Más Calificados',
            labels={'num_ratings': 'Número de Reviews', 'product_id': 'Producto'},
            color='avg_rating',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5]
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("👥 Usuarios Más Activos")

        ratings_per_user_df = rating_matrix.notna().sum(axis=1).reset_index()
        ratings_per_user_df.columns = ['user_id', 'num_ratings']
        top_users = ratings_per_user_df.nlargest(20, 'num_ratings')

        fig = px.bar(
            top_users,
            x='user_id',
            y='num_ratings',
            title='Top 20 Usuarios Más Activos',
            labels={'user_id': 'Usuario ID', 'num_ratings': 'Número de Reviews'},
            color='num_ratings',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔧 Información Técnica de Modelos")

        st.markdown("""
        ### 📊 Estado de Modelos en Esta Sesión
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🧮 User-CF", "Calculado", help="Matriz de similitud usuario-usuario calculada en memoria")
            st.metric("🧮 Item-CF", "Calculado", help="Matriz de similitud producto-producto calculada en memoria")

        with col2:
            st.metric("🔢 SVD", f"{SVD_K_FACTORS} factores", help=f"Matrix Factorization con {SVD_K_FACTORS} componentes latentes")
            st.metric("🎭 Híbrido", f"{int(HYBRID_WEIGHT_USER_CF*100)}-{int(HYBRID_WEIGHT_ITEM_CF*100)}-{int(HYBRID_WEIGHT_SVD*100)}",
                     help="Combinación ponderada: User-CF, Item-CF, SVD")

        with col3:
            ncf_status = "✅ Disponible" if PYTORCH_AVAILABLE else "❌ PyTorch no instalado"
            st.metric("🧠 NCF (DL)", ncf_status, help="Neural Collaborative Filtering con PyTorch")
            experiments_count = len(st.session_state.experiments) if 'experiments' in st.session_state else 0
            st.metric("🔬 Experimentos", experiments_count, help="Entrenamientos realizados en Laboratorio NCF")

        st.divider()

        st.markdown("""
        ### 💾 Archivos de Persistencia

        Esta aplicación **NO carga modelos pre-entrenados** de disco. Todo se calcula en tiempo real:
        """)

        # Tabla de formato de persistencia
        persistence_data = {
            'Modelo': [
                'User-Based CF',
                'Item-Based CF',
                'SVD Matrix Factorization',
                'Sistema Híbrido',
                'NCF (Deep Learning)'
            ],
            'En Web': [
                '🔵 In-Memory (calculado al inicio)',
                '🔵 In-Memory (calculado al inicio)',
                '🔵 In-Memory (calculado al inicio)',
                '🔵 In-Memory (combinación dinámica)',
                '🟡 Entrena on-demand en Lab'
            ],
            'En Notebooks': [
                '💾 user_similarity.pkl',
                '💾 item_similarity.pkl',
                '💾 svd_model.pkl',
                '💾 hybrid_model.pkl',
                '💾 ncf_*.pth (PyTorch)'
            ],
            'Formato': [
                'pickle',
                'pickle',
                'pickle',
                'pickle',
                'PyTorch'
            ]
        }

        df_persistence = pd.DataFrame(persistence_data)
        st.dataframe(df_persistence, use_container_width=True, hide_index=True)

        st.info("""
        💡 **¿Por qué calculado en memoria?**
        Para fines educativos, esta web muestra el proceso completo desde datos raw.
        Los notebooks sí guardan y cargan modelos para acelerar experimentación.
        """)

        st.divider()

        # Información de archivos disponibles
        st.markdown("### 📁 Archivos `.pkl` Generados por Notebooks")

        import os
        models_dir = MODELS_DIR
        pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')] if models_dir.exists() else []
        pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')] if models_dir.exists() else []

        if pkl_files:
            st.write("**Modelos Pickle encontrados:**")
            for pkl in pkl_files:
                file_path = models_dir / pkl
                if file_path.exists():
                    size_kb = file_path.stat().st_size / 1024
                    st.text(f"  📦 {pkl} ({size_kb:.1f} KB)")
            st.caption("Estos archivos fueron generados por los notebooks de exploración pero NO se usan en esta web.")
        else:
            st.warning("No se encontraron archivos .pkl en models/")

        if pth_files:
            st.write("**Modelos PyTorch encontrados:**")
            for pth in pth_files:
                file_path = models_dir / pth
                if file_path.exists():
                    size_kb = file_path.stat().st_size / 1024
                    st.text(f"  🔥 {pth} ({size_kb:.1f} KB)")
            st.caption("Estos archivos son pesos de redes neuronales NCF entrenadas en notebooks.")
        else:
            st.info("No se encontraron archivos .pth. Entrena un modelo NCF en el Laboratorio para generarlos.")

# =============================================================================
# MODO 4: LABORATORIO NCF
# =============================================================================

elif mode == "🧠 Laboratorio NCF":
    st.header("🧠 Laboratorio de Entrenamiento NCF")
    st.markdown(
        "Experimenta con diferentes **hiperparámetros** de la red neuronal "
        "y observa cómo afectan al rendimiento del modelo en tiempo real."
    )

    if not PYTORCH_AVAILABLE:
        st.error("PyTorch no está instalado. Ejecuta: `pip install torch`")
        st.stop()

    # Preparar datos NCF (cacheado)
    ncf_data = prepare_ncf_data(df)

    st.divider()

    # -----------------------------------------------------------------
    # SECCIÓN: HIPERPARÁMETROS
    # -----------------------------------------------------------------
    st.subheader("⚙️ Configuración de Hiperparámetros")
    st.caption(
        "Ajusta estos controles para cambiar cómo aprende la red neuronal. "
        "Pasa el ratón sobre el **?** de cada control para ver una explicación."
    )

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        embedding_dim = st.select_slider(
            "📐 Tamaño de Representación",
            options=[16, 32, 64, 128, 256],
            value=64,
            help="¿Con cuántos números describimos a cada usuario y producto? "
                 "Más números = más detalle, pero necesita más datos para aprender bien. "
                 "Piensa en ello como la resolución de una foto: más píxeles = más detalle."
        )

        architecture = st.selectbox(
            "🏗️ Tamaño del Cerebro",
            options=[
                "Pequeña [64, 32]",
                "Mediana [128, 64, 32]",
                "Grande [256, 128, 64]",
                "Muy Grande [512, 256, 128, 64]",
                "Personalizada"
            ],
            index=1,
            help="El 'cerebro' de la red tiene varias capas de neuronas. "
                 "Más capas y más neuronas = puede aprender patrones más complejos, "
                 "pero también tarda más y puede memorizar en vez de aprender."
        )

    with col_b:
        dropout = st.slider(
            "🎲 Olvido Aleatorio (Dropout)",
            min_value=0.0,
            max_value=0.7,
            value=0.2,
            step=0.05,
            help="En cada ronda, este porcentaje de neuronas se 'apaga' al azar. "
                 "Esto obliga al modelo a no depender demasiado de neuronas específicas "
                 "y aprende patrones más generales. Como estudiar tapando partes del libro."
        )

        use_batch_norm = st.toggle(
            "📊 Normalización (Batch Norm)",
            value=True,
            help="Mantiene los números dentro de la red en un rango controlado. "
                 "Esto hace que el aprendizaje sea más estable y rápido. "
                 "Casi siempre es buena idea tenerlo activado."
        )

    with col_c:
        learning_rate = st.select_slider(
            "🚶 Velocidad de Aprendizaje",
            options=[0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.005, 0.01],
            value=0.001,
            help="¿Qué tan grandes son los 'pasos' que da el modelo para aprender? "
                 "Muy rápido (0.01) = puede pasarse de largo y no encontrar la solución. "
                 "Muy lento (0.0001) = aprende bien pero tarda mucho. "
                 "0.001 suele ser un buen punto de partida."
        )

        optimizer_choice = st.selectbox(
            "🔧 Estrategia de Aprendizaje",
            options=["AdamW (Recomendado)", "Adam", "SGD + Momentum"],
            index=0,
            help="El algoritmo que decide cómo ajustar la red en cada paso. "
                 "AdamW: inteligente y con protección contra memorización. "
                 "Adam: versión clásica sin protección extra. "
                 "SGD: el más simple, pero puede funcionar bien con paciencia."
        )

    # Capas personalizadas
    arch_map = {
        "Pequeña [64, 32]": [64, 32],
        "Mediana [128, 64, 32]": [128, 64, 32],
        "Grande [256, 128, 64]": [256, 128, 64],
        "Muy Grande [512, 256, 128, 64]": [512, 256, 128, 64],
    }

    if architecture == "Personalizada":
        custom_layers_str = st.text_input(
            "Capas personalizadas (separadas por coma):",
            value="128, 64, 32",
            help="Ejemplo: 256, 128, 64"
        )
        try:
            hidden_layers = [int(x.strip()) for x in custom_layers_str.split(',')]
        except ValueError:
            st.warning("Formato inválido. Usando [128, 64, 32]")
            hidden_layers = [128, 64, 32]
    else:
        hidden_layers = arch_map[architecture]

    st.divider()

    # Segunda fila de hiperparámetros
    col_d, col_e, col_f, col_g = st.columns(4)

    with col_d:
        weight_decay = st.select_slider(
            "⚖️ Penalización por Complejidad",
            options=[0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            value=1e-5,
            format_func=lambda x: f"{x:.0e}" if x > 0 else "0 (sin penalización)",
            help="Penaliza al modelo si sus pesos internos crecen demasiado. "
                 "Esto evita que el modelo se vuelva demasiado 'seguro' de sí mismo. "
                 "Valores bajos (1e-5) son un buen comienzo."
        )

    with col_e:
        batch_size = st.select_slider(
            "📦 Datos por Paso",
            options=[64, 128, 256, 512, 1024],
            value=256,
            help="¿Cuántos ejemplos ve el modelo antes de actualizar lo aprendido? "
                 "Más ejemplos = aprendizaje más estable pero más lento. "
                 "Menos ejemplos = más rápido pero más ruidoso."
        )

    with col_f:
        epochs = st.slider(
            "🔄 Rondas Máximas",
            min_value=3,
            max_value=50,
            value=15,
            help="¿Cuántas veces puede el modelo repasar todos los datos? "
                 "Más rondas = más oportunidades de aprender, pero puede memorizar. "
                 "El sistema puede parar antes si detecta que ya no mejora."
        )

    with col_g:
        patience = st.slider(
            "⏳ Paciencia",
            min_value=2,
            max_value=10,
            value=3,
            help="¿Cuántas rondas sin mejorar esperamos antes de parar? "
                 "Poca paciencia (2) = para rápido, puede perder mejoras lentas. "
                 "Mucha paciencia (8+) = da más tiempo, pero puede memorizar."
        )

    st.sidebar.markdown("### 🔧 Ajustes Avanzados")

    gradient_clip = st.sidebar.slider(
        "✂️ Límite de Gradiente",
        min_value=0.5,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Limita los 'saltos' que da el modelo al aprender. "
             "Evita que el entrenamiento se desestabilice por correcciones demasiado grandes."
    )

    use_scheduler = st.sidebar.toggle(
        "📉 Reducción Automática de Velocidad",
        value=True,
        help="Si el modelo se estanca, reduce automáticamente la velocidad de aprendizaje "
             "para hacer ajustes más finos. Como cuando afinas un instrumento: primero giras "
             "la clavija rápido, luego ajustas despacio."
    )

    # -----------------------------------------------------------------
    # RESUMEN DE CONFIGURACIÓN
    # -----------------------------------------------------------------
    total_emb_params = (ncf_data['n_users'] + ncf_data['n_products']) * embedding_dim
    mlp_params = 0
    prev_dim = embedding_dim * 2
    for h in hidden_layers:
        mlp_params += prev_dim * h + h  # weights + biases
        if use_batch_norm:
            mlp_params += h * 2  # gamma + beta
        prev_dim = h
    mlp_params += prev_dim * 1 + 1  # output layer
    total_params = total_emb_params + mlp_params

    with st.expander("📋 Resumen de Configuración", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Arquitectura:**
            - Embeddings: {embedding_dim} dims
            - Capas: {hidden_layers}
            - Dropout: {dropout}
            - Batch Norm: {'✅' if use_batch_norm else '❌'}
            - Total parámetros: **{total_params:,}**
            """)
        with col2:
            st.markdown(f"""
            **Entrenamiento:**
            - LR: {learning_rate}
            - Optimizador: {optimizer_choice.split(' ')[0]}
            - Weight Decay: {weight_decay:.0e}
            - Batch Size: {batch_size}
            - Epochs: {epochs} (patience: {patience})
            - Gradient Clip: {gradient_clip}
            - LR Scheduler: {'✅' if use_scheduler else '❌'}
            """)

        st.markdown(f"""
        **Datos:**
        - Usuarios: {ncf_data['n_users']:,} | Productos: {ncf_data['n_products']:,}
        - Train: {len(ncf_data['train_df']):,} ({len(ncf_data['train_df'])/ncf_data['total_interactions']*100:.1f}%)
        - Validation: {len(ncf_data['val_df']):,} ({len(ncf_data['val_df'])/ncf_data['total_interactions']*100:.1f}%)
        - Test: {len(ncf_data['test_df']):,} ({len(ncf_data['test_df'])/ncf_data['total_interactions']*100:.1f}%)
        """)

        with st.expander("ℹ️ ¿Para qué sirve cada conjunto de datos?"):
            st.markdown("""
            **🎓 Metodología Train/Validation/Test:**

            - **Train (70%)**: El modelo aprende SOLO de estos datos. Ajusta sus pesos aquí.
            - **Validation (15%)**: Durante el entrenamiento, evaluamos aquí para:
                - Ajustar hiperparámetros (learning rate, momentum, etc.)
                - Decidir cuándo parar (early stopping)
                - Activar el LR Scheduler
            - **Test (15%)**: Se usa UNA SOLA VEZ al FINAL para medir el rendimiento real.

            **⚠️ Regla de oro**: El test set NUNCA debe influir en decisiones de entrenamiento.
            Si el modelo "viera" el test durante el entrenamiento, estaríamos **haciendo trampa** y
            reportaríamos un error artificialmente bajo.

            **✅ En esta implementación:**
            - Durante el entrenamiento solo verás Train RMSE y Validation RMSE
            - El Test RMSE se calcula UNA vez al terminar el entrenamiento
            - Así garantizamos una evaluación justa y sin sesgos
            """)

    st.divider()

    # -----------------------------------------------------------------
    # BOTÓN DE ENTRENAMIENTO
    # -----------------------------------------------------------------
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ✨ MEJORADO: Crear datasets para Train, Validation y Test
        train_dataset = NCFDataset(
            ncf_data['train_df']['user_idx'].values,
            ncf_data['train_df']['product_idx'].values,
            ncf_data['train_df']['rating'].values
        )
        val_dataset = NCFDataset(
            ncf_data['val_df']['user_idx'].values,
            ncf_data['val_df']['product_idx'].values,
            ncf_data['val_df']['rating'].values
        )
        test_dataset = NCFDataset(
            ncf_data['test_df']['user_idx'].values,
            ncf_data['test_df']['product_idx'].values,
            ncf_data['test_df']['rating'].values
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Crear modelo
        model = NeuralCollaborativeFiltering(
            n_users=ncf_data['n_users'],
            n_products=ncf_data['n_products'],
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        ).to(device)

        # Optimizador
        if "AdamW" in optimizer_choice:
            opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif "Adam" in optimizer_choice:
            opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

        criterion = nn.MSELoss()

        # Scheduler
        scheduler = None
        if use_scheduler:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, min_lr=1e-6)

        early_stopping = EarlyStopping(patience=patience)

        # Historial
        history = {
            'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
            'learning_rates': [], 'epoch_times': []
        }

        # UI de progreso
        st.markdown("### 🏋️ Entrenamiento en Progreso...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        chart_placeholder = st.empty()

        start_time = time.time()
        stopped_early = False

        for epoch in range(epochs):
            epoch_start = time.time()

            # === ENTRENAMIENTO ===
            model.train()
            train_loss = 0.0
            for users, products, ratings in train_loader:
                users, products, ratings = users.to(device), products.to(device), ratings.to(device)
                predictions = model(users, products)
                loss = criterion(predictions, ratings)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                opt.step()
                train_loss += loss.item() * len(users)

            train_loss /= len(train_dataset)
            train_rmse = np.sqrt(train_loss)

            # === VALIDACIÓN ===
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for users, products, ratings in val_loader:
                    users, products, ratings = users.to(device), products.to(device), ratings.to(device)
                    predictions = model(users, products)
                    loss = criterion(predictions, ratings)
                    val_loss += loss.item() * len(users)

            val_loss /= len(val_dataset)
            val_rmse = np.sqrt(val_loss)

            current_lr = opt.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start

            if scheduler:
                scheduler.step(val_rmse)

            history['train_rmse'].append(train_rmse)
            history['val_rmse'].append(val_rmse)
            history['learning_rates'].append(current_lr)
            history['epoch_times'].append(epoch_time)

            # Actualizar UI
            progress_bar.progress((epoch + 1) / epochs)
            status_text.markdown(
                f"**Epoch {epoch+1}/{epochs}** — "
                f"Train RMSE: `{train_rmse:.4f}` | "
                f"Val RMSE: `{val_rmse:.4f}` | "
                f"LR: `{current_lr:.6f}` | "
                f"Tiempo: `{epoch_time:.1f}s`"
            )

            # Actualizar métricas
            with metrics_cols[0]:
                st.metric("Train RMSE", f"{train_rmse:.4f}",
                         delta=f"{train_rmse - history['train_rmse'][-2]:.4f}" if epoch > 0 else None)
            with metrics_cols[1]:
                st.metric("Val RMSE", f"{val_rmse:.4f}",
                         delta=f"{val_rmse - history['val_rmse'][-2]:.4f}" if epoch > 0 else None,
                         delta_color="inverse")
            with metrics_cols[2]:
                st.metric("Mejor Val RMSE", f"{min(history['val_rmse']):.4f}")
            with metrics_cols[3]:
                gap = val_rmse - train_rmse
                if gap < 0.10:
                    health_label, health_icon = "Saludable", "🟢"
                elif gap < 0.20:
                    health_label, health_icon = "Atención", "🟡"
                else:
                    health_label, health_icon = "Memorizando", "🔴"
                st.metric(f"{health_icon} Salud", health_label)

            # Gráfico en tiempo real
            with chart_placeholder.container():
                epochs_range = list(range(1, len(history['train_rmse']) + 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs_range, y=history['train_rmse'],
                    mode='lines+markers', name='Train RMSE',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=epochs_range, y=history['val_rmse'],
                    mode='lines+markers', name='Validation RMSE',
                    line=dict(color='#F18F01', width=2)
                ))
                fig.update_layout(
                    title='Curvas de Entrenamiento (RMSE)',
                    xaxis_title='Época',
                    yaxis_title='RMSE',
                    height=350,
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Early stopping
            early_stopping(val_rmse, epoch + 1)
            if early_stopping.early_stop:
                stopped_early = True
                break

        # === EVALUACIÓN FINAL EN TEST SET ===
        # El test set SOLO se usa AQUÍ, después del entrenamiento completo
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for users, products, ratings in test_loader:
                users, products, ratings = users.to(device), products.to(device), ratings.to(device)
                predictions = model(users, products)
                loss = criterion(predictions, ratings)
                test_loss += loss.item() * len(users)

        test_loss /= len(test_dataset)
        final_test_rmse = np.sqrt(test_loss)
        history['test_rmse'].append(final_test_rmse)  # Solo un valor al final

        total_time = time.time() - start_time
        best_val_rmse = min(history['val_rmse'])
        best_epoch = np.argmin(history['val_rmse']) + 1
        total_epochs_run = len(history['val_rmse'])

        progress_bar.progress(1.0)

        if stopped_early:
            status_text.markdown(
                f"🛑 **Early Stopping** en epoch {total_epochs_run} — "
                f"Mejor Val RMSE: **{best_val_rmse:.4f}** (epoch {best_epoch}) — "
                f"**Test Final RMSE: {final_test_rmse:.4f}** — "
                f"Tiempo total: **{total_time:.1f}s**"
            )
        else:
            status_text.markdown(
                f"✅ **Entrenamiento completado** ({total_epochs_run} epochs) — "
                f"Mejor Val RMSE: **{best_val_rmse:.4f}** (epoch {best_epoch}) — "
                f"**Test Final RMSE: {final_test_rmse:.4f}** — "
                f"Tiempo total: **{total_time:.1f}s**"
            )

        # Guardar en session state
        experiment = {
            'id': len(st.session_state.experiments) + 1,
            'embedding_dim': embedding_dim,
            'hidden_layers': str(hidden_layers),
            'dropout': dropout,
            'batch_norm': use_batch_norm,
            'learning_rate': learning_rate,
            'optimizer': optimizer_choice.split(' ')[0],
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'max_epochs': epochs,
            'patience': patience,
            'gradient_clip': gradient_clip,
            'lr_scheduler': use_scheduler,
            'total_params': total_params,
            'best_val_rmse': best_val_rmse,
            'final_test_rmse': final_test_rmse,
            'best_epoch': best_epoch,
            'total_epochs': total_epochs_run,
            'stopped_early': stopped_early,
            'train_time': round(total_time, 1),
            'history': history
        }
        st.session_state.experiments.append(experiment)
        st.session_state.current_history = history
        st.session_state.training_done = True

        # ✅ Opción para guardar modelo en formato .pth
        st.success("🎉 Entrenamiento completado exitosamente")

        col_save1, col_save2 = st.columns([2, 1])
        with col_save1:
            model_name = st.text_input(
                "💾 Nombre del modelo (opcional)",
                value=f"ncf_exp{experiment['id']}_test{final_test_rmse:.4f}",
                help="Si deseas guardar este modelo para usarlo después"
            )
        with col_save2:
            st.write("")  # Espaciador
            st.write("")  # Espaciador
            if st.button("💾 Guardar Modelo (.pth)", type="secondary"):
                # Guardar modelo completo
                model_path = MODELS_DIR / f"{model_name}.pth"
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'n_users': ncf_data['n_users'],
                    'n_products': ncf_data['n_products'],
                    'embedding_dim': embedding_dim,
                    'hidden_layers': hidden_layers,
                    'dropout': dropout,
                    'use_batch_norm': use_batch_norm,
                    'user_to_idx': ncf_data['user_to_idx'],
                    'product_to_idx': ncf_data['product_to_idx'],
                    'idx_to_user': ncf_data['idx_to_user'],
                    'idx_to_product': ncf_data['idx_to_product'],
                    'best_val_rmse': best_val_rmse,
                    'final_test_rmse': final_test_rmse,
                    'best_epoch': best_epoch,
                    'total_params': total_params,
                    'history': history,
                    'hyperparameters': {
                        'learning_rate': learning_rate,
                        'weight_decay': weight_decay,
                        'batch_size': batch_size,
                        'optimizer': optimizer_choice,
                        'gradient_clip': gradient_clip,
                        'lr_scheduler': use_scheduler
                    }
                }
                torch.save(checkpoint, str(model_path))
                st.success(f"✅ Modelo guardado en: `models/{model_name}.pth` ({model_path.stat().st_size / 1024:.1f} KB)")
                st.info("💡 Este archivo .pth contiene todos los pesos del modelo y puede ser cargado en notebooks o futuras sesiones.")

        st.divider()

        # Obtener predicciones completas para visualización
        model.eval()
        all_preds, all_actuals = [], []
        with torch.no_grad():
            for users, products, ratings in test_loader:
                users, products, ratings = users.to(device), products.to(device), ratings.to(device)
                preds = model(users, products)
                all_preds.extend(preds.cpu().numpy())
                all_actuals.extend(ratings.cpu().numpy())
        all_preds = np.array(all_preds)
        all_actuals = np.array(all_actuals)

        st.divider()

        # =============================================================
        # RESULTADOS DETALLADOS
        # =============================================================
        st.subheader("📊 Resultados Detallados")

        tab_curves, tab_preds, tab_compare = st.tabs([
            "📈 Curvas de Entrenamiento",
            "🎯 Predicciones",
            "🔬 Comparación de Experimentos"
        ])

        with tab_curves:
            # Explicación general para principiantes
            st.info(
                "📖 **¿Cómo leer estos gráficos?** "
                "La línea **azul** muestra qué tan bien el modelo acierta con datos que ya conoce. "
                "La línea **roja** muestra qué tan bien acierta con datos **nuevos** (lo que importa). "
                "Lo ideal es que **ambas líneas bajen juntas**. Si la azul baja mucho y la roja sube, "
                "el modelo está memorizando en vez de aprender."
            )

            col1, col2 = st.columns(2)

            with col1:
                # Gráfico de RMSE completo
                epochs_range = list(range(1, total_epochs_run + 1))
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs_range, y=history['train_rmse'],
                    mode='lines+markers',
                    name='Datos conocidos (Train)',
                    line=dict(color='#2E86AB', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=epochs_range, y=history['test_rmse'],
                    mode='lines+markers',
                    name='Datos nuevos (Test)',
                    line=dict(color='#D62828', width=2)
                ))
                fig.add_vline(x=best_epoch, line_dash="dash",
                             line_color="green",
                             annotation_text=f"Mejor punto: {best_rmse:.4f}")
                fig.update_layout(
                    title='📉 ¿Cuánto se equivoca el modelo? (menor = mejor)',
                    xaxis_title='Ronda de aprendizaje',
                    yaxis_title='Error promedio (RMSE)',
                    height=400, template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Salud del modelo (reemplazo del gap técnico)
                gap_values = np.array(history['test_rmse']) - np.array(history['train_rmse'])
                final_gap = gap_values[-1]

                fig = go.Figure()

                # Zonas de colores de fondo (de abajo a arriba)
                fig.add_hrect(y0=0, y1=0.10, fillcolor="#06A77D", opacity=0.12,
                             line_width=0, annotation_text="Aprendiendo bien",
                             annotation_position="top left",
                             annotation_font_color="#06A77D")
                fig.add_hrect(y0=0.10, y1=0.20, fillcolor="#F18F01", opacity=0.12,
                             line_width=0, annotation_text="Empezando a memorizar",
                             annotation_position="top left",
                             annotation_font_color="#F18F01")
                fig.add_hrect(y0=0.20, y1=max(0.40, max(gap_values) * 1.2),
                             fillcolor="#D62828", opacity=0.12,
                             line_width=0, annotation_text="Memorizando datos",
                             annotation_position="top left",
                             annotation_font_color="#D62828")

                # Colorear la línea según la zona
                line_colors = []
                for g in gap_values:
                    if g < 0.10:
                        line_colors.append('#06A77D')
                    elif g < 0.20:
                        line_colors.append('#F18F01')
                    else:
                        line_colors.append('#D62828')

                # Línea principal
                fig.add_trace(go.Scatter(
                    x=epochs_range, y=gap_values,
                    mode='lines+markers',
                    name='Salud del modelo',
                    line=dict(color=line_colors[-1], width=3),
                    marker=dict(color=line_colors, size=8)
                ))

                fig.update_layout(
                    title='🩺 ¿Está aprendiendo o memorizando?',
                    xaxis_title='Época',
                    yaxis_title='Diferencia de error',
                    yaxis_range=[0, max(0.40, max(gap_values) * 1.2)],
                    height=400, template='plotly_white',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

                # Interpretación automática en lenguaje simple
                if final_gap < 0.10:
                    st.success(
                        "🟢 **El modelo está aprendiendo correctamente.** "
                        "Lo que aprende con los datos de entrenamiento también "
                        "funciona con datos que nunca ha visto."
                    )
                elif final_gap < 0.20:
                    st.warning(
                        "🟡 **El modelo empieza a memorizar** en vez de aprender patrones. "
                        "Prueba a subir el Dropout, activar Batch Normalization, "
                        "o reducir el tamaño de la red."
                    )
                else:
                    st.error(
                        "🔴 **El modelo está memorizando los datos** en lugar de aprender. "
                        "Se sabe las respuestas de entrenamiento de memoria, pero falla "
                        "con datos nuevos. Sube el Dropout, reduce Epochs o usa una red más pequeña."
                    )

            # Learning rate evolution
            if use_scheduler:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=epochs_range, y=history['learning_rates'],
                    mode='lines+markers', name='Velocidad de aprendizaje',
                    line=dict(color='#06A77D', width=2),
                    fill='tozeroy', fillcolor='rgba(6, 167, 125, 0.1)'
                ))
                fig.update_layout(
                    title='🎚️ Velocidad de Aprendizaje (se reduce cuando el modelo se estanca)',
                    xaxis_title='Ronda de aprendizaje',
                    yaxis_title='Velocidad',
                    yaxis_type='log',
                    height=300, template='plotly_white',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption(
                    "Cuando el modelo deja de mejorar, el sistema reduce "
                    "automáticamente la velocidad de aprendizaje para hacer ajustes más finos."
                )

        with tab_preds:
            st.info(
                "📖 **¿Cómo leer estos gráficos?** "
                "A la izquierda: cada punto es una predicción. Si el modelo fuera perfecto, "
                "todos los puntos estarían sobre la línea roja diagonal. "
                "A la derecha: muestra cuánto se equivoca y con qué frecuencia. "
                "Lo ideal es una campana centrada en 0 (sin error)."
            )

            col1, col2 = st.columns(2)

            errors = all_actuals - all_preds

            with col1:
                # Scatter: actual vs predicted
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=all_actuals, y=all_preds,
                    mode='markers',
                    marker=dict(size=4, opacity=0.3, color='#2E86AB'),
                    name='Predicciones'
                ))
                fig.add_trace(go.Scatter(
                    x=[1, 5], y=[1, 5],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    name='Línea de acierto perfecto'
                ))
                fig.update_layout(
                    title='🎯 ¿Qué tan cerca están las predicciones?',
                    xaxis_title='Rating que el usuario dio realmente',
                    yaxis_title='Rating que el modelo predijo',
                    height=450, template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Distribución de errores con zonas
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=errors, nbinsx=40,
                    marker_color='#A23B72',
                    name='Predicciones'
                ))
                fig.add_vline(x=0, line_dash="dash", line_color="green",
                             annotation_text="Sin error",
                             annotation_position="top")
                fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="green", opacity=0.08,
                             line_width=0, annotation_text="Muy buenas",
                             annotation_position="top left",
                             annotation_font_color="green")
                fig.update_layout(
                    title='📊 ¿Cuánto se equivoca el modelo?',
                    xaxis_title='Error (negativo = predijo de más, positivo = predijo de menos)',
                    yaxis_title='Cantidad de predicciones',
                    height=450, template='plotly_white',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

            # Métricas resumidas con explicación
            mae = np.mean(np.abs(errors))
            rmse_val = np.sqrt(np.mean(errors**2))
            pct_within_half = (np.abs(errors) < 0.5).mean() * 100
            pct_within_one = (np.abs(errors) < 1.0).mean() * 100

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Error promedio", f"{mae:.2f} ⭐",
                         help="En promedio, el modelo se equivoca por esta cantidad de estrellas")
            with col2:
                st.metric("RMSE", f"{rmse_val:.4f}",
                         help="Métrica técnica de error. Menor = mejor")
            with col3:
                st.metric("Aciertos cercanos", f"{pct_within_half:.0f}%",
                         help="Predicciones con menos de media estrella de error")
            with col4:
                st.metric("Aciertos razonables", f"{pct_within_one:.0f}%",
                         help="Predicciones con menos de 1 estrella de error")

            # Interpretación
            if mae < 0.5:
                st.success(
                    f"🎉 **Excelente precisión.** El modelo se equivoca en promedio "
                    f"por solo **{mae:.2f} estrellas** y el **{pct_within_half:.0f}%** "
                    f"de las predicciones están a menos de media estrella del valor real."
                )
            elif mae < 0.8:
                st.info(
                    f"👍 **Buena precisión.** Error promedio de **{mae:.2f} estrellas**. "
                    f"El **{pct_within_one:.0f}%** de predicciones están a menos "
                    f"de 1 estrella del valor real."
                )
            else:
                st.warning(
                    f"⚠️ **Precisión mejorable.** Error promedio de **{mae:.2f} estrellas**. "
                    f"Prueba otros hiperparámetros para reducir el error."
                )

            # Muestra de predicciones
            st.markdown("**🎯 Ejemplos de Predicciones:**")
            sample_indices = np.random.choice(len(all_actuals), min(10, len(all_actuals)), replace=False)
            sample_data = []
            for i in sample_indices:
                error_abs = abs(all_actuals[i] - all_preds[i])
                if error_abs < 0.5:
                    accuracy = "🟢 Excelente"
                elif error_abs < 1.0:
                    accuracy = "🟡 Buena"
                else:
                    accuracy = "🔴 Lejos"
                sample_data.append({
                    'Rating Real': f"{all_actuals[i]:.1f} ⭐",
                    'Rating Predicho': f"{all_preds[i]:.2f} ⭐",
                    'Diferencia': f"{error_abs:.2f}",
                    'Precisión': accuracy
                })
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)

        with tab_compare:
            if len(st.session_state.experiments) > 0:
                st.markdown("**📊 Historial de Experimentos:**")
                st.caption(
                    "Cada fila es un entrenamiento que hiciste con diferentes configuraciones. "
                    "Compara el \"Error\" (RMSE) para ver cuál funciona mejor."
                )

                # Tabla comparativa simplificada
                exp_table = []
                for exp in st.session_state.experiments:
                    # Determinar salud
                    gap = exp['history']['test_rmse'][-1] - exp['history']['train_rmse'][-1]
                    if gap < 0.10:
                        salud = "🟢 Saludable"
                    elif gap < 0.20:
                        salud = "🟡 Atención"
                    else:
                        salud = "🔴 Memorizando"

                    exp_table.append({
                        '#': exp['id'],
                        'Red': f"Emb {exp['embedding_dim']} → {exp['hidden_layers']}",
                        'Dropout': exp['dropout'],
                        'Batch Norm': '✅' if exp['batch_norm'] else '❌',
                        'Velocidad (LR)': exp['learning_rate'],
                        'Optimizador': exp['optimizer'],
                        'Rondas': f"{exp['total_epochs']}/{exp['max_epochs']}",
                        'Paró antes': '🛑 Sí' if exp['stopped_early'] else '✅ No',
                        'Error Test (RMSE)': f"{exp['final_test_rmse']:.4f}",
                        'Salud': salud,
                        'Tiempo': f"{exp['train_time']}s"
                    })

                exp_df = pd.DataFrame(exp_table)
                st.dataframe(exp_df, use_container_width=True, hide_index=True)

                # Gráfico de comparación de curvas
                if len(st.session_state.experiments) > 1:
                    st.markdown("**📈 ¿Cuál aprende mejor?**")
                    st.caption("Compara las curvas de validación de cada experimento. El número en la leyenda muestra el RMSE final en el test set.")
                    fig = go.Figure()

                    colors = px.colors.qualitative.Set2
                    for i, exp in enumerate(st.session_state.experiments):
                        color = colors[i % len(colors)]
                        exp_epochs = list(range(1, len(exp['history']['val_rmse']) + 1))
                        fig.add_trace(go.Scatter(
                            x=exp_epochs,
                            y=exp['history']['val_rmse'],
                            mode='lines+markers',
                            name=f"#{exp['id']} — Test Final: {exp['final_test_rmse']:.4f}",
                            line=dict(color=color, width=2)
                        ))

                    fig.update_layout(
                        title='Curvas de Validación por Experimento',
                        xaxis_title='Ronda de aprendizaje (Epoch)',
                        yaxis_title='Validation RMSE',
                        height=450,
                        template='plotly_white',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Ranking
                st.markdown("**🏆 Ranking (por Test RMSE):**")
                ranking = sorted(st.session_state.experiments,
                               key=lambda x: x['final_test_rmse'])
                for i, exp in enumerate(ranking):
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                    st.markdown(
                        f"{medal} **Experimento #{exp['id']}** — "
                        f"Test RMSE: **{exp['final_test_rmse']:.4f}** — "
                        f"Red: Emb {exp['embedding_dim']} → {exp['hidden_layers']}, "
                        f"Dropout: {exp['dropout']}, LR: {exp['learning_rate']}"
                    )

                if st.button("🗑️ Limpiar Historial de Experimentos"):
                    st.session_state.experiments = []
                    st.rerun()
            else:
                st.info(
                    "Aún no hay experimentos. Configura los hiperparámetros arriba "
                    "y pulsa **Entrenar Modelo** para ver resultados aquí. "
                    "Puedes entrenar varias veces con diferentes configuraciones "
                    "y este panel te mostrará cuál funciona mejor."
                )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
with st.expander("📦 Detalles Técnicos - Persistencia de Modelos"):
    st.markdown("""
    ### Formatos de Almacenamiento Utilizados

    Esta aplicación web utiliza diferentes estrategias de persistencia:

    | Modelo | Formato | Ubicación | Uso en Web |
    |--------|---------|-----------|------------|
    | **User-Based CF** | In-Memory | Calculado on-the-fly | ✅ Calculado al inicio |
    | **Item-Based CF** | In-Memory | Calculado on-the-fly | ✅ Calculado al inicio |
    | **SVD** | In-Memory | Calculado on-the-fly | ✅ Calculado al inicio |
    | **Sistema Híbrido** | In-Memory | Combinación de los 3 | ✅ Calculado on-the-fly |
    | **NCF (Deep Learning)** | `.pth` (PyTorch) | `models/ncf_*.pth` | ❌ Entrena desde cero* |

    <small>* El Laboratorio NCF puede guardar modelos en `.pth` pero no los carga automáticamente</small>

    ---

    ### Archivos `.pkl` Disponibles (generados por notebooks)

    Los siguientes archivos existen en `models/` pero **NO se usan en la web**:
    - `user_similarity.pkl` → Matriz de similitud user-based (de notebook 02)
    - `item_similarity.pkl` → Matriz de similitud item-based (de notebook 03)
    - `svd_model.pkl` → Factorización SVD (de notebook 04)
    - `hybrid_model.pkl` → Sistema híbrido pre-calculado (de notebook 05)

    **¿Por qué no cargarlos?** Para fines educativos, esta app muestra el proceso completo
    de cálculo desde datos raw. Los notebooks usan `.pkl` para persistencia entre sesiones.

    ---

    ### Modelos PyTorch `.pth` Disponibles

    - `ncf_model.pth` → Modelo básico NCF (de notebook 06)
    - `ncf_improved_model.pth` → NCF con BatchNorm + L2 (de notebook 01_tutorial)
    - `ncf_tutorial_model.pth` → Modelo del tutorial educativo

    Estos archivos contienen:
    - `model_state_dict` → Pesos y biases de la red
    - `n_users`, `n_products` → Dimensiones
    - `embedding_dim`, `hidden_layers` → Arquitectura
    - `user_to_idx`, `product_to_idx` → Mapeos ID→índice
    - `history` → Métricas de entrenamiento
    """)

st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎓 <strong>Sistema de Recomendación de Moda</strong></p>
        <p>Proyecto educativo usando Amazon Fashion Reviews</p>
        <p>Tecnologías: User-CF + Item-CF + SVD + NCF (PyTorch) + Streamlit</p>
        <p style='font-size: 0.85em; color: #666;'>Modelos calculados in-memory | Notebooks usan .pkl/.pth para persistencia</p>
    </div>
    """,
    unsafe_allow_html=True
)
