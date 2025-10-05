# app.py
import streamlit as st
import pandas as pd
import pickle
import os
from config import KAGGLE_FEATURES, MODELS_DIR

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Precios de Casas",
    page_icon="🏠",
    layout="wide"
)

# Cargar modelo
@st.cache_resource
def load_model():
    model_path = os.path.join(MODELS_DIR, 'modelo_kaggle_rf.pkl')
    with open(model_path, 'rb') as f:
        modelo = pickle.load(f)
    return modelo

modelo = load_model()

# Título
st.title("🏠 Predictor de Precios de Casas")
st.markdown("### Predice el precio de una casa basándose en sus características")
st.markdown("---")

# Descripción
st.sidebar.header("📋 Información del Modelo")
st.sidebar.info("""
**Modelo:** Random Forest  
**Precisión (R²):** 88.82%  
**Error promedio:** $18,714  
**Dataset:** Kaggle House Prices
""")

# Formulario de entrada
st.header("Ingresa las características de la casa:")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Características Principales")
    
    overall_qual = st.slider(
        "Calidad General (1-10)",
        min_value=1,
        max_value=10,
        value=7,
        help="Calidad general de materiales y acabados"
    )
    
    gr_liv_area = st.number_input(
        "Área Habitable (pies cuadrados)",
        min_value=500,
        max_value=5000,
        value=1500,
        step=50,
        help="Área habitable sobre el suelo"
    )
    
    total_bsmt_sf = st.number_input(
        "Área del Sótano (pies cuadrados)",
        min_value=0,
        max_value=3000,
        value=1000,
        step=50,
        help="Área total del sótano"
    )
    
    garage_cars = st.selectbox(
        "Capacidad del Garaje (# autos)",
        options=[0, 1, 2, 3, 4],
        index=2,
        help="Capacidad del garaje en número de autos"
    )
    
    garage_area = st.number_input(
        "Área del Garaje (pies cuadrados)",
        min_value=0,
        max_value=1500,
        value=500,
        step=50
    )

with col2:
    st.subheader("Características Adicionales")
    
    year_built = st.slider(
        "Año de Construcción",
        min_value=1900,
        max_value=2024,
        value=2000
    )
    
    year_remod_add = st.slider(
        "Año de Remodelación",
        min_value=1900,
        max_value=2024,
        value=2000,
        help="Año de remodelación o adición"
    )
    
    full_bath = st.selectbox(
        "Baños Completos",
        options=[0, 1, 2, 3, 4],
        index=2
    )
    
    tot_rms_abv_grd = st.slider(
        "Total de Habitaciones (sobre el suelo)",
        min_value=2,
        max_value=15,
        value=6,
        help="No incluye baños"
    )
    
    fireplaces = st.selectbox(
        "Número de Chimeneas",
        options=[0, 1, 2, 3],
        index=0
    )

st.markdown("---")

# Botón de predicción
if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
    # Crear DataFrame con los datos ingresados
    input_data = pd.DataFrame({
        'OverallQual': [overall_qual],
        'GrLivArea': [gr_liv_area],
        'GarageCars': [garage_cars],
        'GarageArea': [garage_area],
        'TotalBsmtSF': [total_bsmt_sf],
        'FullBath': [full_bath],
        'TotRmsAbvGrd': [tot_rms_abv_grd],
        'YearBuilt': [year_built],
        'YearRemodAdd': [year_remod_add],
        'Fireplaces': [fireplaces]
    })
    
    # Hacer predicción
    prediccion = modelo.predict(input_data)[0]
    
    # Mostrar resultado
    st.success("✅ Predicción completada")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💰 Precio Estimado",
            value=f"${prediccion:,.0f}"
        )
    
    with col2:
        st.metric(
            label="📊 Rango Estimado (±10%)",
            value=f"${prediccion*0.9:,.0f} - ${prediccion*1.1:,.0f}"
        )
    
    with col3:
        st.metric(
            label="🎯 Confianza del Modelo",
            value="88.82%"
        )
    
    # Mostrar datos ingresados
    with st.expander("📋 Ver datos ingresados"):
        st.dataframe(input_data.T, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desarrollado con ❤️ usando Streamlit y Random Forest</p>
    <p>Modelo entrenado con 1,460 casas del dataset de Kaggle</p>
</div>
""", unsafe_allow_html=True)