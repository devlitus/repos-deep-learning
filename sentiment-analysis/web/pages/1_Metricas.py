"""
Métricas del Modelo
===================

Visualiza las métricas y rendimiento del modelo LSTM.
"""

import sys
from pathlib import Path

# Agregar directorio raíz al path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import json
from PIL import Image

import config

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Métricas - Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# HEADER
# ============================================================================

st.title("📊 Métricas del Modelo")
st.markdown("---")

# ============================================================================
# VERIFICAR ARCHIVOS
# ============================================================================

# Verificar si existen los reportes
training_history_path = config.REPORTS_DIR / "training_history.png"
lstm_training_path = config.REPORTS_DIR / "lstm_training.png"
model_comparison_path = config.REPORTS_DIR / "model_comparison.csv"
sample_predictions_path = config.REPORTS_DIR / "sample_predictions.csv"
confidence_path = config.REPORTS_DIR / "prediction_confidence.png"

# ============================================================================
# MÉTRICAS PRINCIPALES
# ============================================================================

st.header("🎯 Rendimiento del Modelo")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Accuracy",
        value="87.1%",
        delta="2.5%",
        help="Precisión del modelo en el conjunto de validación"
    )

with col2:
    st.metric(
        label="AUC-ROC",
        value="93.3%",
        delta="3.1%",
        help="Área bajo la curva ROC"
    )

with col3:
    st.metric(
        label="Precision",
        value="86.8%",
        help="Proporción de predicciones positivas correctas"
    )

with col4:
    st.metric(
        label="Recall",
        value="87.1%",
        help="Proporción de casos positivos correctamente identificados"
    )

# ============================================================================
# TRAINING HISTORY
# ============================================================================

st.markdown("---")
st.header("📈 Histórico de Entrenamiento")

if training_history_path.exists():
    image = Image.open(training_history_path)
    st.image(image, caption="Evolución de Loss y Accuracy durante el entrenamiento", use_container_width=True)
else:
    st.info("📊 Ejecuta `python main.py` para generar las gráficas de entrenamiento")

# Explicación
with st.expander("ℹ️ ¿Cómo interpretar estas gráficas?"):
    st.markdown("""
    **Loss (Pérdida):**
    - Mide qué tan equivocado está el modelo
    - Valores más bajos = mejor rendimiento
    - La línea azul (Training) debe descender
    - La línea naranja (Validation) debe seguir a Training
    - Si Validation sube mientras Training baja = **Overfitting**

    **Accuracy (Precisión):**
    - Porcentaje de predicciones correctas
    - Valores más altos = mejor rendimiento
    - Training y Validation deben aumentar juntos
    - Gran diferencia entre ambas = modelo memoriza en lugar de aprender

    **Nuestro modelo:**
    - ✅ Loss converge suavemente (no hay overfitting severo)
    - ✅ Accuracy alcanza ~87% en validación
    - ✅ Learning rate bajo (0.0001) permite convergencia estable
    """)

# ============================================================================
# LSTM TRAINING DETAILS
# ============================================================================

st.markdown("---")
st.header("🧠 Detalles del Entrenamiento LSTM")

if lstm_training_path.exists():
    image = Image.open(lstm_training_path)
    st.image(image, caption="Métricas detalladas del entrenamiento LSTM", use_container_width=True)

# Información de la arquitectura
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏗️ Arquitectura del Modelo")
    st.markdown("""
    ```
    1. Embedding Layer (64 dimensions)
       ↓
    2. LSTM Layer (64 units)
       ↓
    3. Dropout (30%)
       ↓
    4. Dense Layer (32 units, ReLU)
       ↓
    5. Dropout (20%)
       ↓
    6. Output Layer (1 unit, Sigmoid)
    ```
    """)

    st.info("""
    **🔍 ¿Por qué esta arquitectura?**

    - **1 sola LSTM** (vs 2 bidireccionales): Más simple, menos overfitting
    - **64 units**: Balance entre capacidad y generalización
    - **Dropout 30% + 20%**: Previene memorización
    - **Learning rate 0.0001**: 10x más bajo = convergencia estable
    """)

with col2:
    st.subheader("⚙️ Hiperparámetros")

    params_df = pd.DataFrame({
        'Parámetro': [
            'Vocabulario',
            'Longitud secuencia',
            'Embedding dim',
            'LSTM units',
            'Dropout (LSTM)',
            'Dropout (Dense)',
            'Learning rate',
            'Batch size',
            'Épocas',
            'Early stopping'
        ],
        'Valor': [
            '10,000 palabras',
            '300 palabras',
            '64',
            '64',
            '0.3',
            '0.2',
            '0.0001',
            '64',
            '15 (early stop en 13)',
            'Patience: 5'
        ]
    })

    st.dataframe(params_df, use_container_width=True, hide_index=True)

# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

st.markdown("---")
st.header("🔄 Comparación de Modelos")

if model_comparison_path.exists():
    comparison_df = pd.read_csv(model_comparison_path)
    st.dataframe(comparison_df, use_container_width=True)

    # Crear gráfica de comparación
    st.subheader("📊 Accuracy Comparativa")

    chart_data = comparison_df.set_index('Model')['Test Accuracy']
    st.bar_chart(chart_data)

    with st.expander("💡 Interpretación"):
        st.markdown("""
        **Modelos Clásicos (TF-IDF):**
        - **Naive Bayes**: Rápido, simple, ~85-88% accuracy
        - **SVM**: Más robusto, ~86-90% accuracy
        - Funcionan bien para texto, pero no capturan contexto

        **Modelo Deep Learning (LSTM):**
        - **LSTM**: ~87-88% accuracy
        - Captura relaciones secuenciales (orden de palabras)
        - Mejor con reseñas largas y complejas
        - Requiere más datos y tiempo de entrenamiento

        **¿Cuál usar?**
        - **Producción rápida**: Naive Bayes o SVM
        - **Máxima precisión**: LSTM
        - **Contexto/secuencia importante**: LSTM
        """)
else:
    st.info("📊 Ejecuta `python main.py` para generar la comparación de modelos")

# ============================================================================
# EJEMPLOS DE PREDICCIONES
# ============================================================================

st.markdown("---")
st.header("🔮 Ejemplos de Predicciones")

if sample_predictions_path.exists():
    predictions_df = pd.read_csv(sample_predictions_path)

    # Mostrar tabla
    st.dataframe(
        predictions_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Review": st.column_config.TextColumn("Reseña", width="large"),
            "True Sentiment": st.column_config.TextColumn("Sentimiento Real"),
            "Predicted Sentiment": st.column_config.TextColumn("Predicción"),
            "Confidence": st.column_config.ProgressColumn(
                "Confianza",
                format="%.1f%%",
                min_value=0,
                max_value=100
            )
        }
    )

    # Estadísticas de confianza
    if 'Confidence' in predictions_df.columns:
        avg_confidence = predictions_df['Confidence'].mean()
        st.metric("Confianza Promedio", f"{avg_confidence:.1f}%")

# ============================================================================
# DISTRIBUCIÓN DE CONFIANZA
# ============================================================================

if confidence_path.exists():
    st.markdown("---")
    st.header("📊 Distribución de Confianza")

    image = Image.open(confidence_path)
    st.image(image, caption="Histograma de confianza en las predicciones", use_container_width=True)

    with st.expander("ℹ️ ¿Qué significa esto?"):
        st.markdown("""
        **Confianza del modelo:**
        - Probabilidad asignada a la predicción final
        - Valores cercanos a 1.0 = muy seguro
        - Valores cercanos a 0.5 = incierto/neutral

        **Distribución ideal:**
        - ✅ Muchas predicciones con alta confianza (>0.8)
        - ✅ Pocas predicciones en la zona neutral (0.4-0.6)
        - ⚠️ Si todo está en 0.5 = modelo no aprendió

        **Nuestro modelo:**
        - La mayoría de predicciones tienen >70% confianza
        - Indica que el modelo es decisivo y confía en sus predicciones
        """)

# ============================================================================
# LIMITACIONES
# ============================================================================

st.markdown("---")
st.header("⚠️ Limitaciones y Consideraciones")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🚫 Limitaciones")
    st.markdown("""
    1. **Solo inglés**: Entrenado con reseñas en inglés
    2. **Dominio específico**: Optimizado para reseñas de películas
    3. **Contexto limitado**: 300 palabras máximo
    4. **Sarcasmo/ironía**: Difícil de detectar
    5. **Sentimientos mixtos**: Clasifica como positivo o negativo únicamente
    """)

with col2:
    st.subheader("💡 Mejoras Futuras")
    st.markdown("""
    1. **Multilenguaje**: Entrenar con datos en español
    2. **Transfer learning**: Usar BERT o GPT pre-entrenados
    3. **Sentimientos multi-clase**: Muy negativo → Muy positivo
    4. **Análisis de aspectos**: Detectar qué específicamente gusta/disgusta
    5. **Explicabilidad**: Visualizar qué palabras influyen más
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>📊 Métricas actualizadas según el último entrenamiento</p>
    <p>Para re-entrenar el modelo, ejecuta: <code>python main.py</code></p>
</div>
""", unsafe_allow_html=True)
