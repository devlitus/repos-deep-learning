"""
Sentiment Analysis Web App
===========================

Aplicación Streamlit para análisis de sentimientos interactivo usando LSTM.
Basada en el modelo entrenado con el dataset IMDB.

Ejecución:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
import joblib

# Importar módulos del proyecto
import config
from src import text_preprocessing

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Sentiment Analysis - LSTM",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# ESTILOS CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .positive-sentiment {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #155724;
    }
    .negative-sentiment {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #721c24;
    }
    .neutral-sentiment {
        background-color: #fff8e1;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #663c00;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

@st.cache_resource
def load_model_and_tokenizer():
    """
    Carga el modelo LSTM y el tokenizer.
    Usa caché para evitar recargar en cada interacción.
    """
    try:
        # Cargar modelo LSTM
        model_path = config.MODEL_LSTM
        if not model_path.exists():
            st.error(f"❌ Modelo no encontrado: {model_path}")
            st.info("💡 Ejecuta `python main.py` en la raíz del proyecto para entrenar el modelo")
            return None, None

        model = keras.models.load_model(model_path)

        # Cargar tokenizer
        tokenizer_path = config.TOKENIZER_FILE
        if not tokenizer_path.exists():
            st.error(f"❌ Tokenizer no encontrado: {tokenizer_path}")
            return None, None

        tokenizer = joblib.load(tokenizer_path)

        return model, tokenizer

    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {str(e)}")
        return None, None


def predict_sentiment(text: str, model, tokenizer) -> tuple:
    """
    Predice el sentimiento de un texto.

    Returns:
        tuple: (sentimiento, confianza, probabilidad_positiva)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # 1. Preprocesar texto (limpiar)
    from src.text_preprocessing import clean_text
    cleaned_text = clean_text(
        text,
        remove_html=True,
        remove_url=True,
        remove_punct=config.REMOVE_PUNCTUATION,
        remove_num=config.REMOVE_NUMBERS,
        lowercase=True
    )

    # 2. Convertir a secuencia numérica
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # 3. Padding
    padded = pad_sequences(
        sequence,
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding='pre',
        truncating='post'
    )

    # 4. Predecir (retorna probabilidad entre 0 y 1)
    prob_positive = float(model.predict(padded, verbose=0)[0][0])

    # 5. Clasificar (> 0.5 = positivo)
    prediction = 1 if prob_positive > 0.5 else 0
    sentiment = "Positive" if prediction == 1 else "Negative"

    # 6. Calcular confianza (qué tan lejos está de 0.5)
    confidence = abs(prob_positive - 0.5) * 2

    return sentiment, confidence, prob_positive


def get_sentiment_emoji(sentiment: str, confidence: float) -> str:
    """Retorna emoji basado en sentimiento y confianza."""
    if confidence < 0.6:  # Neutral/incierto
        return "😐"
    elif sentiment == "Positive":
        if confidence > 0.9:
            return "😍"
        elif confidence > 0.75:
            return "😊"
        else:
            return "🙂"
    else:  # Negative
        if confidence > 0.9:
            return "😡"
        elif confidence > 0.75:
            return "😞"
        else:
            return "😕"

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🎬 Sentiment Analysis</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Analiza el sentimiento de reseñas de películas usando Deep Learning (LSTM)</p>',
        unsafe_allow_html=True
    )

    # Sidebar
    st.sidebar.title("⚙️ Configuración")
    st.sidebar.markdown("---")

    # Información del modelo
    st.sidebar.subheader("📊 Modelo")
    st.sidebar.info("""
    **Arquitectura:** LSTM (64 units)
    **Dataset:** IMDB (50,000 reviews)
    **Accuracy:** ~87%
    **Vocabulario:** 10,000 palabras
    """)

    # Cargar modelo
    with st.spinner("🔄 Cargando modelo..."):
        model, tokenizer = load_model_and_tokenizer()

    if model is None or tokenizer is None:
        st.stop()

    st.sidebar.success("✅ Modelo cargado")

    # Ejemplos predefinidos
    st.sidebar.markdown("---")
    st.sidebar.subheader("💡 Ejemplos")

    examples = {
        "Positiva 1": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Positiva 2": "I loved every minute of it. A masterpiece of cinema with brilliant direction and wonderful performances.",
        "Negativa 1": "Terrible movie. Waste of time and money. The acting was awful and the plot made no sense.",
        "Negativa 2": "I couldn't even finish watching it. Boring, predictable, and poorly executed.",
        "Mixta": "The movie had some good moments but overall it was disappointing. Great visuals but weak story."
    }

    selected_example = st.sidebar.selectbox("Selecciona un ejemplo:", [""] + list(examples.keys()))

    # ========================================================================
    # ÁREA PRINCIPAL - INPUT
    # ========================================================================

    st.subheader("📝 Escribe tu reseña")

    # Usar ejemplo si está seleccionado
    default_text = examples[selected_example] if selected_example else ""

    user_input = st.text_area(
        "Reseña de película:",
        value=default_text,
        height=150,
        placeholder="Escribe aquí tu reseña de película en inglés...",
        help="El modelo está entrenado con reseñas en inglés del dataset IMDB"
    )

    # Botones
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        analyze_button = st.button("🔍 Analizar", type="primary", width="stretch")

    with col2:
        clear_button = st.button("🗑️ Limpiar", width="stretch")

    if clear_button:
        st.rerun()

    # ========================================================================
    # PREDICCIÓN Y RESULTADOS
    # ========================================================================

    if analyze_button and user_input.strip():

        with st.spinner("🤔 Analizando sentimiento..."):
            sentiment, confidence, prob_positive = predict_sentiment(user_input, model, tokenizer)

        # Determinar clase CSS
        if confidence < 0.6:
            sentiment_class = "neutral-sentiment"
            sentiment_label = "Neutral/Incierto"
        elif sentiment == "Positive":
            sentiment_class = "positive-sentiment"
            sentiment_label = "Positivo"
        else:
            sentiment_class = "negative-sentiment"
            sentiment_label = "Negativo"

        emoji = get_sentiment_emoji(sentiment, confidence)

        # ====================================================================
        # RESULTADO PRINCIPAL
        # ====================================================================

        st.markdown("---")
        st.subheader("📊 Resultado del Análisis")

        # Tarjeta de resultado
        st.markdown(f"""
        <div class="{sentiment_class}">
            <h2 style="margin: 0;">{emoji} Sentimiento: {sentiment_label}</h2>
            <h3 style="margin-top: 0.5rem;">Confianza: {confidence:.1%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # ====================================================================
        # MÉTRICAS DETALLADAS
        # ====================================================================

        st.markdown("### 📈 Métricas Detalladas")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Probabilidad Positiva",
                value=f"{prob_positive:.1%}",
                delta=f"{prob_positive - 0.5:.1%}" if prob_positive > 0.5 else f"{prob_positive - 0.5:.1%}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Probabilidad Negativa",
                value=f"{1 - prob_positive:.1%}",
                delta=f"{(1 - prob_positive) - 0.5:.1%}" if prob_positive < 0.5 else f"{(1 - prob_positive) - 0.5:.1%}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Nivel de Confianza",
                value=f"{confidence:.1%}"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ====================================================================
        # BARRA DE PROBABILIDAD
        # ====================================================================

        st.markdown("### 🎯 Distribución de Probabilidades")

        # Crear DataFrame para la barra
        prob_data = pd.DataFrame({
            'Sentimiento': ['Negativo', 'Positivo'],
            'Probabilidad': [1 - prob_positive, prob_positive]
        })

        st.bar_chart(prob_data.set_index('Sentimiento'))

        # ====================================================================
        # ANÁLISIS DE TEXTO
        # ====================================================================

        st.markdown("### 🔍 Análisis del Texto")

        # Preprocesar para mostrar
        preprocessed = text_preprocessing.preprocess_text(user_input)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Texto Original:**")
            st.text_area("Texto Original", value=user_input, height=100, disabled=True, key="original", label_visibility="collapsed")

        with col2:
            st.markdown("**Texto Preprocesado:**")
            st.text_area("Texto Preprocesado", value=preprocessed, height=100, disabled=True, key="preprocessed", label_visibility="collapsed")

        # Estadísticas del texto
        words_original = len(user_input.split())
        words_preprocessed = len(preprocessed.split())

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Palabras Originales", words_original)
        with col2:
            st.metric("Palabras Preprocesadas", words_preprocessed)
        with col3:
            st.metric("Reducción", f"{(1 - words_preprocessed/words_original)*100:.0f}%")
        with col4:
            st.metric("Caracteres", len(user_input))

    elif analyze_button:
        st.warning("⚠️ Por favor escribe una reseña antes de analizar")

    # ========================================================================
    # FOOTER
    # ========================================================================

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>Sentiment Analysis Web App</strong></p>
        <p>Modelo LSTM entrenado con 50,000 reseñas de IMDB</p>
        <p>Desarrollado con ❤️ usando TensorFlow y Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
