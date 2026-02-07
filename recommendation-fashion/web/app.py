"""
Aplicación Web - Sistema de Recomendación de Moda
Interfaz interactiva con Streamlit para Amazon Fashion Reviews
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import plotly.express as px
import plotly.graph_objects as go

from config import (
    DATASET_FILE, COL_USER_ID, COL_PRODUCT_ID, COL_RATING,
    COL_REVIEW_TEXT, COL_SUMMARY,
    MIN_USER_RATINGS, MIN_PRODUCT_RATINGS,
    SVD_K_FACTORS, STREAMLIT_PAGE_CONFIG,
    HYBRID_WEIGHT_USER_CF, HYBRID_WEIGHT_ITEM_CF, HYBRID_WEIGHT_SVD
)

# Configuración de la página
st.set_page_config(**STREAMLIT_PAGE_CONFIG)

# Título principal
st.title("👕 Sistema de Recomendación de Moda")
st.markdown("**Amazon Fashion Reviews - Sistema Híbrido (User-CF + Item-CF + SVD)**")
st.divider()

# =============================================================================
# CARGA DE DATOS Y MODELOS (con cache)
# =============================================================================

@st.cache_data
def load_and_prepare_data():
    """Cargar y preprocesar datos de Fashion Reviews"""
    import json

    if not DATASET_FILE.exists():
        st.error(f"Dataset no encontrado: {DATASET_FILE}")
        st.info("Ejecuta primero: `python download_fashion.py`")
        st.stop()

    reviews = []
    with open(DATASET_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                reviews.append(json.loads(line))
            except:
                continue

    df = pd.DataFrame(reviews)

    # Eliminar duplicados
    df = df.drop_duplicates(subset=[COL_USER_ID, COL_PRODUCT_ID], keep='first')

    # Filtrar usuarios y productos con pocas reviews
    user_counts = df[COL_USER_ID].value_counts()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    df = df[df[COL_USER_ID].isin(valid_users)]

    product_counts = df[COL_PRODUCT_ID].value_counts()
    valid_products = product_counts[product_counts >= MIN_PRODUCT_RATINGS].index
    df = df[df[COL_PRODUCT_ID].isin(valid_products)]

    # Crear matriz de ratings
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


# Funciones de predicción
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

# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

st.sidebar.header("⚙️ Configuración")

mode = st.sidebar.radio(
    "Selecciona modo:",
    ["👤 Usuario Existente", "🆕 Usuario Nuevo", "📊 Estadísticas del Sistema"]
)

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

                    # Gráfico de recomendaciones
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

        # Distribución de ratings del usuario
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
        # Productos populares
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

        # Gráfico
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

        # Top rated
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

else:
    st.header("📊 Estadísticas del Sistema")

    # Métricas generales
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

    tab1, tab2, tab3 = st.tabs(["📈 Distribuciones", "👕 Top Productos", "👥 Usuarios Activos"])

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

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎓 <strong>Sistema de Recomendación de Moda</strong></p>
        <p>Proyecto educativo usando Amazon Fashion Reviews</p>
        <p>Tecnologías: User-CF + Item-CF + SVD + Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
