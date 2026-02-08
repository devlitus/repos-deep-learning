"""
Componente de pie de página de la aplicación
Incluye detalles técnicos de persistencia y créditos
"""
import streamlit as st
from pathlib import Path
import sys

# Asegurar que config está en el path
PROJECT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from config import MODELS_DIR


def render_footer():
    """Renderizar pie de página de la aplicación"""
    st.divider()

    with st.expander("📦 Detalles Técnicos - Persistencia de Modelos"):
        st.markdown("""
        ### Formatos de Almacenamiento Utilizados

        Esta aplicación web utiliza diferentes estrategias de persistencia:

        | Modelo | Formato | Ubicación | Uso en Web |
        |--------|---------|-----------|------------|
        | **User-Based CF** | In-Memory | Calculado on-the-fly | ✅ Calculado al inicio |
        | **Item-Based CF** | In-Memory | Calculado on-the-fly | ✅ Calculado al inicio |
        | **SVD** | In-Memory | Calculado on-the-fly | ✅ Calculado al inicio |
        | **Sistema Híbrido** | In-Memory | Combinación de los 3 | ✅ Calculado on-the-fly |
        | **NCF (Deep Learning)** | `.pth` (PyTorch) | `models/ncf_*.pth` | ❌ Entrena desde cero* |

        <small>* El Laboratorio NCF puede guardar modelos en `.pth` pero no los carga automáticamente</small>

        ---

        ### Archivos `.pkl` Disponibles (generados por notebooks)

        Los siguientes archivos existen en `models/` pero **NO se usan en la web**:
        - `user_similarity.pkl` → Matriz de similitud user-based (de notebook 02)
        - `item_similarity.pkl` → Matriz de similitud item-based (de notebook 03)
        - `svd_model.pkl` → Factorización SVD (de notebook 04)
        - `hybrid_model.pkl` → Sistema híbrido pre-calculado (de notebook 05)

        **¿Por qué no cargarlos?** Para fines educativos, esta app muestra el proceso completo
        de cálculo desde datos raw. Los notebooks usan `.pkl` para persistencia entre sesiones.

        ---

        ### Modelos PyTorch `.pth` Disponibles

        - `ncf_model.pth` → Modelo básico NCF (de notebook 06)
        - `ncf_improved_model.pth` → NCF con BatchNorm + L2 (de notebook 01_tutorial)
        - `ncf_tutorial_model.pth` → Modelo del tutorial educativo

        Estos archivos contienen:
        - `model_state_dict` → Pesos y biases de la red
        - `n_users`, `n_products` → Dimensiones
        - `embedding_dim`, `hidden_layers` → Arquitectura
        - `user_to_idx`, `product_to_idx` → Mapeos ID→índice
        - `history` → Métricas de entrenamiento
        """)

    st.divider()
    st.markdown(
        """
        <div style='text-align: center'>
            <p>🎓 <strong>Sistema de Recomendación de Moda</strong></p>
            <p>Proyecto educativo usando Amazon Fashion Reviews</p>
            <p>Tecnologías: User-CF + Item-CF + SVD + NCF (PyTorch) + Streamlit</p>
            <p style='font-size: 0.85em; color: #666;'>Modelos calculados in-memory | Notebooks usan .pkl/.pth para persistencia</p>
        </div>
        """,
        unsafe_allow_html=True
    )
