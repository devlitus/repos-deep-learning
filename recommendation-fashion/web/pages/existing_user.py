"""
Modo 1: Usuario Existente
Recomendaciones personalizadas para usuarios con historial
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Asegurar que imports funcionen
WEB_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(WEB_DIR))

from core.predictions import get_recommendations


def render_existing_user(df, rating_matrix, user_sim_df, item_sim_df, U, sigma, Vt):
    """Renderizar página de usuario existente"""
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
