"""
Componente de encabezado de la aplicación
Incluye título, información técnica y session state initialization
"""
import streamlit as st


def render_header():
    """Renderizar encabezado de la aplicación"""
    # Título principal
    st.title("👕 Sistema de Recomendación de Moda")
    st.markdown("**Amazon Fashion Reviews - Sistema Híbrido (User-CF + Item-CF + SVD + NCF)**")

    # Información de modelos
    with st.expander("ℹ️ Información Técnica - Modelos Utilizados", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **🔧 Modelos Tradicionales (calculados on-the-fly):**
            - **User-Based CF** → Matriz de similitud en memoria
            - **Item-Based CF** → Matriz de similitud en memoria
            - **SVD (Matrix Factorization)** → Descomposición en valores singulares
            - **Sistema Híbrido** → Combinación ponderada de los 3 anteriores

            > 📝 Estos modelos se calculan al cargar la aplicación desde el dataset raw JSON.
            > No utilizan archivos `.pkl` pre-entrenados.
            """)

        with col2:
            st.markdown("""
            **🧠 Deep Learning (PyTorch):**
            - **NCF (Neural Collaborative Filtering)** → Red neuronal entrenada on-demand
            - Arquitectura: Embeddings + MLP (Multi-Layer Perceptron)
            - Entrenamiento: En el "Laboratorio NCF" con hiperparámetros ajustables

            > 📝 El modelo NCF se entrena desde cero en cada sesión.
            > Opcionalmente puede guardar/cargar pesos en formato `.pth` (PyTorch).
            """)

        st.info("""
        💡 **¿Por qué no usar archivos `.pkl`?**
        Esta app prioriza transparencia educativa calculando modelos en tiempo real desde datos raw.
        Los notebooks sí guardan modelos en `.pkl` (SVD, similitudes) y `.pth` (NCF) para reutilización.
        """)

    st.divider()

    # Inicializar session state
    if 'experiments' not in st.session_state:
        st.session_state.experiments = []
    if 'current_history' not in st.session_state:
        st.session_state.current_history = None
    if 'training_done' not in st.session_state:
        st.session_state.training_done = False
