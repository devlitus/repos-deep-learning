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

from config import STREAMLIT_PAGE_CONFIG, SVD_K_FACTORS

# Importar módulos refactorizados
from components.header import render_header
from components.footer import render_footer
from core.data_loader import (
    load_and_prepare_data,
    compute_user_similarity,
    compute_item_similarity,
    compute_svd
)
from core.ncf_models import PYTORCH_AVAILABLE
from pages.existing_user import render_existing_user
from pages.new_user import render_new_user
from pages.statistics import render_statistics
from pages.ncf_lab import render_ncf_lab

# Configuración de la página
st.set_page_config(**STREAMLIT_PAGE_CONFIG)

# Renderizar encabezado
render_header()

# Cargar datos y modelos
with st.spinner('Cargando datos y entrenando modelos...'):
    df, rating_matrix = load_and_prepare_data()
    user_sim_df = compute_user_similarity(rating_matrix)
    item_sim_df = compute_item_similarity(rating_matrix)
    U, sigma, Vt = compute_svd(rating_matrix, k=SVD_K_FACTORS)

st.success(f'✅ Modelos cargados: {rating_matrix.shape[0]} usuarios, {rating_matrix.shape[1]} productos')
st.caption("📊 User-CF, Item-CF y SVD calculados en memoria (sin usar archivos .pkl)")

# Sidebar - Selección de modo
st.sidebar.header("⚙️ Configuración")

modes = ["👤 Usuario Existente", "🆕 Usuario Nuevo", "📊 Estadísticas del Sistema"]
if PYTORCH_AVAILABLE:
    modes.append("🧠 Laboratorio NCF")

mode = st.sidebar.radio("Selecciona modo:", modes)

st.sidebar.divider()

# Routing por modo
if mode == "👤 Usuario Existente":
    render_existing_user(df, rating_matrix, user_sim_df, item_sim_df, U, sigma, Vt)

elif mode == "🆕 Usuario Nuevo":
    render_new_user(df)

elif mode == "📊 Estadísticas del Sistema":
    render_statistics(df, rating_matrix)

elif mode == "🧠 Laboratorio NCF":
    render_ncf_lab(df)

# Renderizar footer
render_footer()
