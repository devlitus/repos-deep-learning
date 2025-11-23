"""
Helpers - Funciones auxiliares para la web app
===============================================

Utilidades reutilizables para la aplicación Streamlit.
"""

import streamlit as st
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np


def format_confidence(confidence: float) -> str:
    """
    Formatea la confianza como porcentaje con color.

    Args:
        confidence: Valor entre 0 y 1

    Returns:
        str: Confianza formateada con HTML/CSS
    """
    percentage = confidence * 100

    if percentage >= 90:
        color = "#28a745"  # Verde oscuro
        icon = "🟢"
    elif percentage >= 75:
        color = "#5cb85c"  # Verde claro
        icon = "🟢"
    elif percentage >= 60:
        color = "#f0ad4e"  # Naranja
        icon = "🟡"
    else:
        color = "#d9534f"  # Rojo
        icon = "🔴"

    return f'{icon} <span style="color: {color}; font-weight: bold;">{percentage:.1f}%</span>'


def get_sentiment_color(sentiment: str) -> str:
    """
    Retorna el color asociado a un sentimiento.

    Args:
        sentiment: "Positive" o "Negative"

    Returns:
        str: Código de color hexadecimal
    """
    colors = {
        "Positive": "#28a745",
        "Negative": "#dc3545",
        "Neutral": "#ffc107"
    }
    return colors.get(sentiment, "#6c757d")


def create_confidence_gauge(confidence: float, sentiment: str) -> str:
    """
    Crea un medidor visual de confianza en HTML/CSS.

    Args:
        confidence: Valor entre 0 y 1
        sentiment: "Positive" o "Negative"

    Returns:
        str: HTML del gauge
    """
    percentage = confidence * 100
    color = get_sentiment_color(sentiment)

    html = f"""
    <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; overflow: hidden;">
        <div style="width: {percentage}%; background-color: {color}; height: 30px;
                    display: flex; align-items: center; justify-content: center; color: white;
                    font-weight: bold; transition: width 0.3s ease;">
            {percentage:.1f}%
        </div>
    </div>
    """
    return html


def format_text_stats(text: str, preprocessed: str) -> pd.DataFrame:
    """
    Crea un DataFrame con estadísticas del texto.

    Args:
        text: Texto original
        preprocessed: Texto preprocesado

    Returns:
        DataFrame con estadísticas
    """
    stats = {
        'Métrica': [
            'Caracteres (original)',
            'Palabras (original)',
            'Caracteres (preprocesado)',
            'Palabras (preprocesado)',
            'Reducción de palabras',
            'Palabras promedio por oración'
        ],
        'Valor': [
            len(text),
            len(text.split()),
            len(preprocessed),
            len(preprocessed.split()),
            f"{(1 - len(preprocessed.split())/len(text.split()))*100:.1f}%",
            f"{len(text.split()) / max(text.count('.'), 1):.1f}"
        ]
    }
    return pd.DataFrame(stats)


def highlight_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica formato condicional a un DataFrame de predicciones.

    Args:
        df: DataFrame con columnas de predicción

    Returns:
        DataFrame con estilos aplicados
    """
    def color_sentiment(val):
        if val == "Positive":
            return 'background-color: #d4edda'
        elif val == "Negative":
            return 'background-color: #f8d7da'
        return ''

    return df.style.applymap(color_sentiment, subset=['Predicted Sentiment'])


def validate_input(text: str, min_words: int = 3, max_words: int = 1000) -> Tuple[bool, str]:
    """
    Valida el input del usuario.

    Args:
        text: Texto a validar
        min_words: Mínimo de palabras requeridas
        max_words: Máximo de palabras permitidas

    Returns:
        tuple: (es_valido, mensaje_error)
    """
    if not text or not text.strip():
        return False, "⚠️ Por favor escribe una reseña"

    words = text.split()
    num_words = len(words)

    if num_words < min_words:
        return False, f"⚠️ La reseña es muy corta. Mínimo {min_words} palabras (tienes {num_words})"

    if num_words > max_words:
        return False, f"⚠️ La reseña es muy larga. Máximo {max_words} palabras (tienes {num_words})"

    return True, ""


def create_comparison_table(predictions: List[Dict]) -> pd.DataFrame:
    """
    Crea una tabla comparativa de múltiples predicciones.

    Args:
        predictions: Lista de dicts con predicciones

    Returns:
        DataFrame formateado
    """
    df = pd.DataFrame(predictions)

    # Ordenar por confianza
    if 'confidence' in df.columns:
        df = df.sort_values('confidence', ascending=False)

    return df


def format_model_metrics(metrics: Dict) -> str:
    """
    Formatea las métricas del modelo en HTML.

    Args:
        metrics: Dict con métricas (accuracy, precision, etc.)

    Returns:
        str: HTML formateado
    """
    html = "<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>"

    for metric, value in metrics.items():
        html += f"""
        <div style='background: #f8f9fa; padding: 1rem; border-radius: 5px; text-align: center;'>
            <h3 style='margin: 0; color: #1f77b4;'>{value:.2%}</h3>
            <p style='margin: 0.5rem 0 0 0; color: #666;'>{metric.title()}</p>
        </div>
        """

    html += "</div>"
    return html


def safe_load_model(model_path):
    """
    Carga un modelo de forma segura con manejo de errores.

    Args:
        model_path: Ruta al modelo

    Returns:
        Modelo cargado o None si falla
    """
    try:
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar modelo: {str(e)}")
        return None


def display_success_message(message: str, icon: str = "✅"):
    """
    Muestra un mensaje de éxito estilizado.

    Args:
        message: Mensaje a mostrar
        icon: Emoji o icono
    """
    st.markdown(f"""
    <div style="background-color: #d4edda; border-left: 5px solid #28a745;
                padding: 1rem; border-radius: 5px; margin: 1rem 0;">
        <p style="margin: 0; color: #155724;"><strong>{icon} {message}</strong></p>
    </div>
    """, unsafe_allow_html=True)


def display_error_message(message: str, icon: str = "❌"):
    """
    Muestra un mensaje de error estilizado.

    Args:
        message: Mensaje a mostrar
        icon: Emoji o icono
    """
    st.markdown(f"""
    <div style="background-color: #f8d7da; border-left: 5px solid #dc3545;
                padding: 1rem; border-radius: 5px; margin: 1rem 0;">
        <p style="margin: 0; color: #721c24;"><strong>{icon} {message}</strong></p>
    </div>
    """, unsafe_allow_html=True)


def display_info_message(message: str, icon: str = "ℹ️"):
    """
    Muestra un mensaje informativo estilizado.

    Args:
        message: Mensaje a mostrar
        icon: Emoji o icono
    """
    st.markdown(f"""
    <div style="background-color: #d1ecf1; border-left: 5px solid #0c5460;
                padding: 1rem; border-radius: 5px; margin: 1rem 0;">
        <p style="margin: 0; color: #0c5460;"><strong>{icon} {message}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# CONSTANTES
# ============================================================================

SENTIMENT_EMOJIS = {
    "Positive": {
        "very_confident": "😍",
        "confident": "😊",
        "moderate": "🙂"
    },
    "Negative": {
        "very_confident": "😡",
        "confident": "😞",
        "moderate": "😕"
    },
    "Neutral": "😐"
}

EXAMPLE_REVIEWS = {
    "Positive - Excelente": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. A masterpiece!",
    "Positive - Buena": "I really enjoyed this film. Great cinematography and solid performances from the cast.",
    "Negative - Terrible": "Terrible movie. Complete waste of time and money. The acting was awful and the plot made no sense whatsoever.",
    "Negative - Mala": "I didn't like this movie. It was boring and predictable. Would not recommend.",
    "Mixta": "The movie had some good moments but overall it was disappointing. Great visuals but weak story.",
    "Neutral": "It was okay. Not great, not terrible. Just an average movie that I probably won't remember."
}
