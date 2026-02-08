"""
Laboratorio NCF - Orquestador principal
Integra hyperparameters, training y results
"""
import streamlit as st
from pathlib import Path
import sys

# Asegurar imports
WEB_DIR = Path(__file__).parent.parent.parent
PROJECT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from core.ncf_models import PYTORCH_AVAILABLE
from core.data_loader import prepare_ncf_data
from .hyperparameters import render_hyperparameters
from .training import run_training
from .results import render_results


def render_ncf_lab(df):
    """Renderizar el Laboratorio NCF completo"""
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

    # 1. Configurar hiperparámetros
    hyperparams = render_hyperparameters(ncf_data)

    st.divider()

    # 2. Entrenar modelo
    if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
        experiment, all_preds, all_actuals = run_training(ncf_data, hyperparams)

        st.divider()

        # 3. Mostrar resultados
        render_results(experiment, all_preds, all_actuals)
