"""
Modo 2: Usuario Nuevo (Cold Start)
Recomendaciones basadas en popularidad para usuarios sin historial
"""
import streamlit as st
import plotly.express as px
from pathlib import Path
import sys

# Asegurar que config está en el path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import COL_PRODUCT_ID, COL_RATING


def render_new_user(df):
    """Renderizar página de usuario nuevo"""
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
