"""
Aplicación Web de Sistema de Recomendación de Películas
Interfaz interactiva con Streamlit
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="🎬 Sistema de Recomendación de Películas",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🎬 Sistema de Recomendación de Películas")
st.markdown("**Basado en MovieLens 100K - Sistema Híbrido (SVD + Item-Based CF)**")
st.divider()

# =============================================================================
# CARGA DE DATOS Y MODELOS (con cache)
# =============================================================================

@st.cache_data
def load_data():
    """Cargar datos de MovieLens"""
    ratings = pd.read_csv(
        'data/raw/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    
    movies = pd.read_csv(
        'data/raw/ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
               'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
               'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
               'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    )
    
    ratings_matrix = ratings.pivot(
        index='user_id',
        columns='item_id',
        values='rating'
    )
    
    return ratings, movies, ratings_matrix

@st.cache_data
def train_svd(_ratings_matrix, k=50):
    """Entrenar modelo SVD"""
    user_ratings_mean = _ratings_matrix.mean(axis=1)
    ratings_matrix_norm = _ratings_matrix.sub(user_ratings_mean, axis=0)
    ratings_matrix_filled = ratings_matrix_norm.fillna(0)
    
    U, sigma, Vt = svds(ratings_matrix_filled.values, k=k)
    sigma_matrix = np.diag(sigma)
    
    predictions_normalized = np.dot(np.dot(U, sigma_matrix), Vt)
    svd_predictions = pd.DataFrame(
        predictions_normalized,
        index=_ratings_matrix.index,
        columns=_ratings_matrix.columns
    )
    svd_predictions = svd_predictions.add(user_ratings_mean, axis=0).clip(1, 5)
    
    return svd_predictions

@st.cache_data
def calculate_item_similarity(_ratings_matrix):
    """Calcular similitud Item-Based"""
    ratings_matrix_t = _ratings_matrix.T.fillna(0)
    item_similarity = cosine_similarity(ratings_matrix_t)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=_ratings_matrix.columns,
        columns=_ratings_matrix.columns
    )
    return item_similarity_df

# Cargar datos
with st.spinner('Cargando datos y entrenando modelos...'):
    ratings, movies, ratings_matrix = load_data()
    svd_predictions = train_svd(ratings_matrix)
    item_similarity_df = calculate_item_similarity(ratings_matrix)

st.success('✅ Modelos cargados y entrenados')

# =============================================================================
# FUNCIONES DEL SISTEMA
# =============================================================================

def get_user_top_movies(user_id, ratings_matrix, movies_df, n=10):
    """Obtener películas favoritas del usuario"""
    user_ratings = ratings_matrix.loc[user_id].dropna().sort_values(ascending=False)
    top_movies = user_ratings.head(n)
    
    results = []
    for item_id, rating in top_movies.items():
        movie_info = movies_df[movies_df['item_id'] == item_id].iloc[0]
        results.append({
            'title': movie_info['title'],
            'rating': rating,
            'item_id': item_id
        })
    
    return pd.DataFrame(results)

def get_hybrid_recommendations(user_id, svd_pred, ratings_mat, similarity_df, movies_df, n=10):
    """Obtener recomendaciones híbridas con explicaciones"""
    user_predictions = svd_pred.loc[user_id]
    seen_movies = ratings_mat.loc[user_id].dropna().index
    unseen_predictions = user_predictions.drop(seen_movies)
    top_predictions = unseen_predictions.sort_values(ascending=False).head(n)
    
    recommendations = []
    for item_id in top_predictions.index:
        svd_score = top_predictions[item_id]
        
        # Obtener explicación
        user_ratings = ratings_mat.loc[user_id].dropna()
        similar_items = similarity_df.loc[item_id, user_ratings.index]
        top_similar = similar_items.sort_values(ascending=False).head(3)
        
        explanations = []
        for similar_item_id, similarity in top_similar.items():
            user_rating = user_ratings[similar_item_id]
            similar_movie_title = movies_df[movies_df['item_id'] == similar_item_id]['title'].values[0]
            explanations.append({
                'movie': similar_movie_title,
                'your_rating': user_rating,
                'similarity': similarity
            })
        
        movie_info = movies_df[movies_df['item_id'] == item_id].iloc[0]
        
        recommendations.append({
            'item_id': item_id,
            'title': movie_info['title'],
            'predicted_rating': svd_score,
            'explanation': explanations
        })
    
    return pd.DataFrame(recommendations)

def get_popular_movies(ratings_df, movies_df, min_ratings=100, n=10):
    """Obtener películas populares para usuarios nuevos"""
    movie_stats = ratings_df.groupby('item_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()
    movie_stats.columns = ['item_id', 'avg_rating', 'num_ratings']
    
    popular_movies = movie_stats[movie_stats['num_ratings'] >= min_ratings]
    popular_movies = popular_movies.sort_values('avg_rating', ascending=False).head(n)
    
    recommendations = popular_movies.merge(
        movies_df[['item_id', 'title']],
        on='item_id'
    )
    
    return recommendations[['title', 'avg_rating', 'num_ratings']]

# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

# Sidebar
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
        options=sorted(ratings_matrix.index.tolist()),
        index=0
    )
    
    num_ratings = ratings_matrix.loc[user_id].notna().sum()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.metric("Usuario ID", f"#{user_id}")
    with col2:
        st.metric("Películas calificadas", num_ratings)
    
    st.divider()
    
    # Tabs para organizar información
    tab1, tab2, tab3 = st.tabs(["🎬 Recomendaciones", "⭐ Favoritas", "📈 Análisis"])
    
    with tab1:
        st.subheader("🎬 Top 10 Recomendaciones Personalizadas")
        
        num_recs = st.slider("Número de recomendaciones:", 5, 20, 10)
        
        if st.button("Generar Recomendaciones", type="primary"):
            with st.spinner("Generando recomendaciones..."):
                recs = get_hybrid_recommendations(
                    user_id, svd_predictions, ratings_matrix, 
                    item_similarity_df, movies, n=num_recs
                )
                
                for idx, row in recs.iterrows():
                    with st.container():
                        st.markdown(f"### {idx+1}. {row['title']}")
                        
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.metric("Rating Predicho", f"{row['predicted_rating']:.2f} ⭐")
                        
                        with col2:
                            st.markdown("**📝 Porque te gustaron:**")
                            for exp in row['explanation'][:2]:
                                st.markdown(
                                    f"• *{exp['movie']}* "
                                    f"(calificaste: {exp['your_rating']:.0f}⭐, "
                                    f"similitud: {exp['similarity']:.2f})"
                                )
                        
                        st.divider()
    
    with tab2:
        st.subheader("⭐ Tus Películas Favoritas")
        
        top_movies = get_user_top_movies(user_id, ratings_matrix, movies, n=15)
        
        # Crear gráfico de barras
        fig = px.bar(
            top_movies,
            x='rating',
            y='title',
            orientation='h',
            title='Top 15 Películas Mejor Calificadas',
            labels={'rating': 'Rating', 'title': 'Película'},
            color='rating',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5]
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla
        st.dataframe(
            top_movies[['title', 'rating']],
            use_container_width=True,
            hide_index=True
        )
    
    with tab3:
        st.subheader("📈 Análisis del Perfil")
        
        user_ratings_series = ratings_matrix.loc[user_id].dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rating Promedio", f"{user_ratings_series.mean():.2f}")
        with col2:
            st.metric("Películas Vistas", len(user_ratings_series))
        with col3:
            coverage = (len(user_ratings_series) / len(ratings_matrix.columns)) * 100
            st.metric("Cobertura", f"{coverage:.1f}%")
        
        # Distribución de ratings
        rating_counts = user_ratings_series.value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title='Distribución de tus Ratings',
            labels={'x': 'Rating', 'y': 'Cantidad'},
            color=rating_counts.index,
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MODO 2: USUARIO NUEVO (COLD START)
# =============================================================================

elif mode == "🆕 Usuario Nuevo":
    st.header("🆕 Recomendaciones para Usuario Nuevo")
    
    st.info(
        "👋 ¡Bienvenido! Como no tenemos información sobre tus gustos, "
        "te recomendamos las películas más populares y mejor calificadas."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_recs = st.slider("Número de recomendaciones:", 5, 20, 10)
    with col2:
        min_ratings = st.slider("Mínimo de ratings (confiabilidad):", 50, 300, 100)
    
    if st.button("Ver Recomendaciones", type="primary"):
        popular = get_popular_movies(ratings, movies, min_ratings=min_ratings, n=num_recs)
        
        st.subheader("🎬 Películas Más Populares y Mejor Calificadas")
        
        for idx, row in popular.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{idx+1}. {row['title']}**")
            with col2:
                st.metric("Rating", f"{row['avg_rating']:.2f} ⭐")
            with col3:
                st.metric("Ratings", f"{int(row['num_ratings'])}")
            
            st.divider()
        
        # Gráfico
        fig = px.scatter(
            popular,
            x='num_ratings',
            y='avg_rating',
            text='title',
            title='Popularidad vs Calidad',
            labels={'num_ratings': 'Número de Ratings', 'avg_rating': 'Rating Promedio'},
            size='avg_rating',
            color='avg_rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_traces(textposition='top center', textfont_size=8)
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MODO 3: ESTADÍSTICAS DEL SISTEMA
# =============================================================================

else:  # Estadísticas
    st.header("📊 Estadísticas del Sistema")
    
    # Métricas generales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Usuarios", ratings_matrix.shape[0])
    with col2:
        st.metric("🎬 Películas", ratings_matrix.shape[1])
    with col3:
        st.metric("⭐ Ratings", len(ratings))
    with col4:
        sparsity = (ratings_matrix.isna().sum().sum() / 
                   (ratings_matrix.shape[0] * ratings_matrix.shape[1]) * 100)
        st.metric("🕳️ Sparsity", f"{sparsity:.1f}%")
    
    st.divider()
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3 = st.tabs(["📈 Distribuciones", "🎬 Top Películas", "👥 Usuarios Activos"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribución de ratings
            st.subheader("Distribución de Ratings")
            rating_dist = ratings['rating'].value_counts().sort_index()
            fig = px.bar(
                x=rating_dist.index,
                y=rating_dist.values,
                labels={'x': 'Rating', 'y': 'Frecuencia'},
                color=rating_dist.index,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ratings por usuario
            st.subheader("Ratings por Usuario")
            ratings_per_user = ratings_matrix.notna().sum(axis=1)
            fig = px.histogram(
                x=ratings_per_user,
                nbins=50,
                labels={'x': 'Número de Ratings', 'y': 'Número de Usuarios'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🏆 Películas Más Calificadas")
        
        movie_stats = ratings.groupby('item_id').agg({
            'rating': ['mean', 'count']
        }).reset_index()
        movie_stats.columns = ['item_id', 'avg_rating', 'num_ratings']
        movie_stats = movie_stats.merge(movies[['item_id', 'title']], on='item_id')
        
        top_movies = movie_stats.nlargest(20, 'num_ratings')
        
        fig = px.bar(
            top_movies,
            x='num_ratings',
            y='title',
            orientation='h',
            title='Top 20 Películas Más Calificadas',
            labels={'num_ratings': 'Número de Ratings', 'title': 'Película'},
            color='avg_rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("👥 Usuarios Más Activos")
        
        ratings_per_user_df = ratings_matrix.notna().sum(axis=1).reset_index()
        ratings_per_user_df.columns = ['user_id', 'num_ratings']
        top_users = ratings_per_user_df.nlargest(20, 'num_ratings')
        
        fig = px.bar(
            top_users,
            x='user_id',
            y='num_ratings',
            title='Top 20 Usuarios Más Activos',
            labels={'user_id': 'Usuario ID', 'num_ratings': 'Número de Ratings'},
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
        <p>🎓 <strong>Sistema de Recomendación de Películas</strong></p>
        <p>Proyecto educativo usando MovieLens 100K</p>
        <p>Tecnologías: SVD + Item-Based CF + Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)