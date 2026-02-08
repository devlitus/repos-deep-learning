"""
Modo 3: Estadísticas del Sistema
Análisis global del dataset y modelos
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Asegurar que config y ncf_models están en el path
PROJECT_DIR = Path(__file__).parent.parent.parent
WEB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(WEB_DIR))

from config import (
    COL_PRODUCT_ID, COL_RATING,
    SVD_K_FACTORS, MODELS_DIR,
    HYBRID_WEIGHT_USER_CF, HYBRID_WEIGHT_ITEM_CF, HYBRID_WEIGHT_SVD
)
from core.ncf_models import PYTORCH_AVAILABLE


def render_statistics(df, rating_matrix):
    """Renderizar página de estadísticas del sistema"""
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

        from pathlib import Path
        models_dir = MODELS_DIR
        pkl_files = [f for f in models_dir.iterdir() if f.suffix == '.pkl'] if models_dir.exists() else []
        pth_files = [f for f in models_dir.iterdir() if f.suffix == '.pth'] if models_dir.exists() else []

        if pkl_files:
            st.write("**Modelos Pickle encontrados:**")
            for pkl in pkl_files:
                if pkl.exists():
                    size_kb = pkl.stat().st_size / 1024
                    st.text(f"  📦 {pkl.name} ({size_kb:.1f} KB)")
            st.caption("Estos archivos fueron generados por los notebooks de exploración pero NO se usan en esta web.")
        else:
            st.warning("No se encontraron archivos .pkl en models/")

        if pth_files:
            st.write("**Modelos PyTorch encontrados:**")
            for pth in pth_files:
                if pth.exists():
                    size_kb = pth.stat().st_size / 1024
                    st.text(f"  🔥 {pth.name} ({size_kb:.1f} KB)")
            st.caption("Estos archivos son pesos de redes neuronales NCF entrenadas en notebooks.")
        else:
            st.info("No se encontraron archivos .pth. Entrena un modelo NCF en el Laboratorio para generarlos.")
