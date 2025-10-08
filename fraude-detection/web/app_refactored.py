"""
Aplicación Web de Detección de Fraude con Streamlit - Versión Refactorizada
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st

# Importar estilos
from styles import CUSTOM_CSS, SIDEBAR_CSS

# Importar páginas
from pages import (
    show_dashboard,
    show_prediction,
    show_analytics,
    show_eda,
    show_about
)

# Configuración de la página
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Aplicar estilos CSS personalizados
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def main():
    """Función principal de la aplicación"""

    # Header con gradiente
    st.markdown("""
        <div class="main-header">
            <h1>🛡️ FRAUD DETECTION SYSTEM</h1>
            <p>Advanced Machine Learning for Real-Time Transaction Analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar con estilo
    with st.sidebar:
        # Inyectar CSS adicional para forzar color blanco
        st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

        st.markdown("## 🎯 Navigation")
        st.markdown("---")

        page = st.radio(
            "Select a page:",
            ["🏠 Dashboard", "🔍 Live Prediction", "📊 Model Analytics", "📈 Data Explorer", "ℹ️ About"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        # Selector de modelo
        model_choice = st.selectbox(
            "Model",
            ["random_forest", "logistic_regression"],
            format_func=lambda x: "Random Forest ⭐" if x == "random_forest" else "Logistic Regression"
        )

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Transactions", "284,807")
        st.metric("Fraud Rate", "0.173%")
        st.metric("Model Accuracy", "98.01%")

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #888; font-size: 0.8rem;'>
                Made with ❤️ using Streamlit<br>
                © 2025 Fraud Detection AI
            </div>
        """, unsafe_allow_html=True)

    # Renderizar página seleccionada
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 Live Prediction":
        show_prediction(model_choice)
    elif page == "📊 Model Analytics":
        show_analytics()
    elif page == "📈 Data Explorer":
        show_eda()
    elif page == "ℹ️ About":
        show_about()


if __name__ == "__main__":
    main()
