"""Utilidades para carga de datos y modelos"""
import sys
from pathlib import Path
import streamlit as st
import joblib

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import MODELS_DIR
from src.data.load import load_data


@st.cache_resource
def load_model(model_name="random_forest"):
    """Carga el modelo entrenado (con cache para no recargar)"""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    """Carga el dataset (con cache)"""
    return load_data()
