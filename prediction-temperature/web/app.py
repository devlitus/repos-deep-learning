"""
═══════════════════════════════════════════════════════════════
APLICACIÓN WEB: PREDICTOR DE TEMPERATURA
═══════════════════════════════════════════════════════════════

Aplicación Streamlit para predecir temperaturas mínimas en Melbourne
utilizando un modelo LSTM entrenado.

Uso:
    streamlit run web/app.py

═══════════════════════════════════════════════════════════════
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
import sys
import pickle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Agregar ruta del proyecto para importar módulos
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Intentar diferentes formas de importar keras
    try:
        from tensorflow import keras
    except (ImportError, AttributeError):
        import keras

    from src.preprocessing import create_sequences
    from data.load_data import load_melbourne_data
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE STREAMLIT
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Predictor de Temperatura",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def load_model():
    """Cargar modelo LSTM entrenado"""
    model_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'models',
        'lstm_temperatura.keras'
    )
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Cargar normalizador (MinMaxScaler)"""
    scaler_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'models',
        'scaler.pkl'
    )
    try:
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
    except Exception as e:
        st.warning(f"No se pudo cargar scaler: {e}")

    # Si no existe, crear uno con rangos típicos
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0], [30]]))
    return scaler


def generate_sample_data():
    """Generar datos de demostración realistas"""
    np.random.seed(42)

    # Crear una onda sinusoidal con ruido (patrón estacional)
    days = np.arange(60)
    base = 15 + 10 * np.sin(2 * np.pi * days / 365)  # Patrón anual
    noise = np.random.normal(0, 1, 60)
    temps = base + noise

    # Asegurar que estén en rango realista
    temps = np.clip(temps, 0, 30)

    return temps


def normalize_data(data, scaler):
    """Normalizar datos usando el scaler"""
    data_reshaped = data.reshape(-1, 1)
    normalized = scaler.transform(data_reshaped)
    return normalized.flatten()


def denormalize_prediction(pred, scaler):
    """Desnormalizar predicción"""
    pred_reshaped = pred.reshape(-1, 1)
    denorm = scaler.inverse_transform(pred_reshaped)
    return denorm[0, 0]


def make_prediction(temperatures, model, scaler):
    """
    Hacer predicción de temperatura para el día siguiente

    Args:
        temperatures: Array con los últimos 60 días de temperaturas
        model: Modelo LSTM cargado
        scaler: MinMaxScaler para normalizar

    Returns:
        prediccion: Temperatura predicha
    """
    try:
        # Validar que tengamos 60 temperaturas
        if len(temperatures) != 60:
            return None, f"Se requieren exactamente 60 temperaturas. Tienes {len(temperatures)}"

        # Normalizar
        normalized = normalize_data(temperatures, scaler)

        # Reshape para LSTM (1, 60, 1)
        X = normalized.reshape(1, 60, 1)

        # Predecir (valor normalizado)
        pred_normalized = model.predict(X, verbose=0)

        # Desnormalizar
        prediccion = denormalize_prediction(pred_normalized[0], scaler)

        return prediccion, None

    except Exception as e:
        return None, str(e)


# ═══════════════════════════════════════════════════════════════
# CARGAR RECURSOS
# ═══════════════════════════════════════════════════════════════

model = load_model()
scaler = load_scaler()

if model is None:
    st.error("No se pudo cargar el modelo. Asegúrate de haber ejecutado train.py primero.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ═══════════════════════════════════════════════════════════════

st.title("🌡️ Predictor de Temperatura - Melbourne")
st.markdown("### Predice la temperatura mínima del día siguiente basándose en los últimos 60 días")
st.markdown("---")

# Sidebar - Información del modelo
with st.sidebar:
    st.header("📊 Información del Modelo")
    st.info("""
    **Modelo:** LSTM (Long Short-Term Memory)

    **Arquitectura:**
    - LSTM Layer 1: 50 neuronas
    - Dropout: 20%
    - LSTM Layer 2: 50 neuronas
    - Dropout: 20%
    - Dense: 1 neurona (predicción)

    **Rendimiento:**
    - RMSE: 2.23°C
    - MAE: 1.75°C
    - R²: 0.706 (70.6% varianza explicada)

    **Dataset:**
    - 10 años de datos (1981-1990)
    - 3,650 observaciones
    - Melbourne, Australia
    """)

    st.markdown("---")
    st.subheader("💡 Cómo usar:")
    st.markdown("""
    1. Ingresa las temperaturas de los últimos 60 días
    2. Usa datos de demostración o ingresa los tuyos
    3. Haz clic en "Predecir"
    4. Visualiza la predicción y rango de confianza
    """)

# Tabs para diferentes formas de entrada
tab1, tab2, tab3 = st.tabs(["📝 Entrada Manual", "📊 Datos de Demostración", "📈 Información"])

# ═══════════════════════════════════════════════════════════════
# TAB 1: ENTRADA MANUAL
# ═══════════════════════════════════════════════════════════════

with tab1:
    st.header("Ingresa las temperaturas de los últimos 60 días")
    st.markdown("**Nota:** Ingresa temperaturas en Celsius")

    col1, col2 = st.columns([3, 1])

    with col1:
        input_method = st.radio(
            "Método de entrada:",
            ["Valores individuales", "Texto pegado", "Generar valores de demostración"]
        )

    temperatures_manual = None

    if input_method == "Valores individuales":
        st.markdown("**Ingresa hasta 60 valores (sliders)**")

        # Crear sliders en una grilla de 10 columnas
        col_width = st.columns(10)
        temperatures_manual = []

        for i in range(60):
            with col_width[i % 10]:
                temp = st.number_input(
                    f"Día {i+1}",
                    min_value=0.0,
                    max_value=30.0,
                    value=15.0,
                    step=0.1,
                    key=f"temp_{i}",
                    label_visibility="collapsed"
                )
                temperatures_manual.append(temp)

        temperatures_manual = np.array(temperatures_manual)

    elif input_method == "Texto pegado":
        st.markdown("**Pega temperaturas separadas por espacios, comas o saltos de línea**")
        text_input = st.text_area(
            "Pega aquí:",
            value="15 16 17 18 19 20 21 20 19 18 17 16 15 14 13 12 11 10 11 12 13 14 15 16 17 18 19 20 21 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24",
            height=100
        )

        try:
            # Intentar parsear
            text_clean = text_input.replace(',', ' ').replace('\n', ' ')
            temps = [float(x) for x in text_clean.split() if x]

            if len(temps) != 60:
                st.warning(f"Se detectaron {len(temps)} valores. Se requieren exactamente 60.")
                if len(temps) > 60:
                    temps = temps[:60]
                    st.info(f"Se usarán los primeros 60 valores")
                elif len(temps) < 60:
                    temps = temps + [15] * (60 - len(temps))
                    st.info(f"Se rellenarán los {60 - len(temps)} valores faltantes con 15°C")

            temperatures_manual = np.array(temps)
        except ValueError as e:
            st.error(f"Error al parsear temperaturas: {e}")
            temperatures_manual = None

    else:  # Generar datos de demostración
        st.info("Generando datos de demostración realistas...")
        temperatures_manual = generate_sample_data()
        st.success(f"✅ Generados 60 valores de demostración")
        st.write(f"Rango: {temperatures_manual.min():.1f}°C - {temperatures_manual.max():.1f}°C")

    # Visualizar datos ingresados
    if temperatures_manual is not None and len(temperatures_manual) == 60:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Estadísticas")
            stats_data = {
                'Mínima': f"{temperatures_manual.min():.1f}°C",
                'Máxima': f"{temperatures_manual.max():.1f}°C",
                'Promedio': f"{temperatures_manual.mean():.1f}°C",
                'Desv. Estándar': f"{temperatures_manual.std():.1f}°C",
            }
            for key, value in stats_data.items():
                st.metric(key, value)

        with col2:
            st.subheader("Gráfica de temperaturas")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(temperatures_manual, marker='o', linewidth=2, markersize=6, color='#FF6B6B')
            ax.fill_between(range(60), temperatures_manual, alpha=0.3, color='#FF6B6B')
            ax.set_xlabel('Día', fontsize=12)
            ax.set_ylabel('Temperatura (°C)', fontsize=12)
            ax.set_title('Últimos 60 días de temperaturas', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 30)
            st.pyplot(fig)

        # Botón de predicción
        st.markdown("---")
        if st.button("🔮 Predecir Temperatura del Día 61", type="primary", use_container_width=True):
            prediccion, error = make_prediction(temperatures_manual, model, scaler)

            if error is None:
                st.success("✅ Predicción completada")

                # Mostrar resultados
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label="🌡️ Temperatura Predicha",
                        value=f"{prediccion:.1f}°C"
                    )

                with col2:
                    st.metric(
                        label="📊 Rango (±RMSE 2.23°C)",
                        value=f"{prediccion-2.23:.1f}°C a {prediccion+2.23:.1f}°C"
                    )

                with col3:
                    st.metric(
                        label="🎯 Confianza del Modelo",
                        value="70.6%"
                    )

                # Mostrar comparación
                st.markdown("---")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Comparación")
                    comparison_data = {
                        'Promedio últimos 60 días': f"{temperatures_manual.mean():.1f}°C",
                        'Predicción día 61': f"{prediccion:.1f}°C",
                        'Diferencia': f"{prediccion - temperatures_manual.mean():.1f}°C"
                    }
                    for key, value in comparison_data.items():
                        st.write(f"**{key}:** {value}")

                with col2:
                    st.subheader("Visualización")
                    fig, ax = plt.subplots(figsize=(10, 5))

                    # Últimos 15 días
                    days_show = 15
                    ax.plot(
                        range(days_show),
                        temperatures_manual[-days_show:],
                        marker='o',
                        linewidth=2,
                        markersize=8,
                        label='Últimos 15 días',
                        color='#4ECDC4'
                    )

                    # Predicción
                    ax.scatter(
                        [days_show],
                        [prediccion],
                        marker='*',
                        s=500,
                        color='#FF6B6B',
                        label='Predicción',
                        zorder=5
                    )

                    # Rango de confianza
                    ax.fill_between(
                        [days_show-0.5, days_show+0.5],
                        prediccion-2.23,
                        prediccion+2.23,
                        alpha=0.3,
                        color='#FF6B6B',
                        label='Rango ±RMSE'
                    )

                    ax.set_xlabel('Días', fontsize=11)
                    ax.set_ylabel('Temperatura (°C)', fontsize=11)
                    ax.set_title('Últimos 15 días + Predicción', fontsize=12, fontweight='bold')
                    ax.legend(loc='best')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 30)
                    st.pyplot(fig)

            else:
                st.error(f"Error en la predicción: {error}")

# ═══════════════════════════════════════════════════════════════
# TAB 2: DATOS DE DEMOSTRACIÓN
# ═══════════════════════════════════════════════════════════════

with tab2:
    st.header("Predice usando datos de demostración")
    st.markdown("Estos son datos realistas basados en el patrón estacional de Melbourne")

    # Generar y mostrar datos de demostración
    demo_temps = generate_sample_data()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Estadísticas de Demostración")
        demo_stats = {
            'Mínima': f"{demo_temps.min():.1f}°C",
            'Máxima': f"{demo_temps.max():.1f}°C",
            'Promedio': f"{demo_temps.mean():.1f}°C",
            'Desv. Estándar': f"{demo_temps.std():.1f}°C",
        }
        for key, value in demo_stats.items():
            st.metric(key, value)

    with col2:
        st.subheader("Gráfica de Demostración")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(demo_temps, marker='o', linewidth=2, markersize=6, color='#4ECDC4')
        ax.fill_between(range(60), demo_temps, alpha=0.3, color='#4ECDC4')
        ax.set_xlabel('Día', fontsize=12)
        ax.set_ylabel('Temperatura (°C)', fontsize=12)
        ax.set_title('Datos de Demostración - Últimos 60 días', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 30)
        st.pyplot(fig)

    st.markdown("---")

    if st.button("🔮 Predecir con Datos de Demostración", type="primary", use_container_width=True):
        prediccion_demo, error_demo = make_prediction(demo_temps, model, scaler)

        if error_demo is None:
            st.success("✅ Predicción completada")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="🌡️ Temperatura Predicha",
                    value=f"{prediccion_demo:.1f}°C"
                )

            with col2:
                st.metric(
                    label="📊 Rango (±RMSE)",
                    value=f"{prediccion_demo-2.23:.1f}°C a {prediccion_demo+2.23:.1f}°C"
                )

            with col3:
                st.metric(
                    label="🎯 Confianza",
                    value="70.6% R²"
                )

            st.markdown("---")
            st.subheader("Análisis Detallado")

            col1, col2 = st.columns(2)

            with col1:
                st.write("""
                **Interpretación:**
                - La predicción se basa en los patrones de los últimos 60 días
                - El rango de ±2.23°C representa un error típico del modelo
                - Esta precisión es adecuada para aplicaciones de planificación
                """)

            with col2:
                analysis_data = {
                    'Promedio entrada': f"{demo_temps.mean():.1f}°C",
                    'Predicción': f"{prediccion_demo:.1f}°C",
                    'Cambio esperado': f"{prediccion_demo - demo_temps.mean():.1f}°C",
                    'Confiabilidad': '70.6%'
                }
                for key, value in analysis_data.items():
                    st.write(f"**{key}:** {value}")

        else:
            st.error(f"Error: {error_demo}")

# ═══════════════════════════════════════════════════════════════
# TAB 3: INFORMACIÓN
# ═══════════════════════════════════════════════════════════════

with tab3:
    st.header("🧠 Cómo Funciona el Modelo")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("¿Qué es LSTM?")
        st.write("""
        **LSTM** = Long Short-Term Memory

        Es una red neuronal especializada en trabajar con series de tiempo.

        **Características:**
        - Recuerda patrones a largo plazo
        - Ideal para temperaturas (patrón cíclico anual)
        - Mejor que modelos simples (promedio móvil, ARIMA)
        """)

        st.subheader("¿Por qué 60 días?")
        st.write("""
        Usamos las últimas 60 temperaturas porque:
        - 60 días ≈ 2 meses de historia
        - Captura patrones semanales (7 días × 8 semanas)
        - Captura tendencias mensuales
        - No es tan largo como para memorizar
        """)

    with col2:
        st.subheader("Rendimiento del Modelo")
        st.write("""
        **Métricas de Evaluación:**
        - **RMSE:** 2.23°C (error promedio)
        - **MAE:** 1.75°C (error absoluto)
        - **R²:** 0.706 (explica 70.6% varianza)

        **Qué significa:**
        Si predice 15°C, la temperatura real está entre 12.77°C y 17.23°C
        """)

        st.subheader("Dataset")
        st.write("""
        **Datos Usados:**
        - Período: 1981-1990 (10 años)
        - Localidad: Melbourne, Australia
        - Observaciones: 3,650 días
        - Variables: Temperatura mínima diaria
        """)

    st.markdown("---")

    st.subheader("🔄 Pipeline Completo")

    pipeline_steps = """
    1. **Carga de Datos** → 3,650 temperaturas históricas
    2. **Normalización** → Escala [0, 1] para la red neuronal
    3. **Secuencias** → Ventanas de 60 días
    4. **Entrenamiento** → 50 épocas, LSTM de 2 capas
    5. **Evaluación** → RMSE=2.23°C, R²=0.706
    6. **Predicción** → Modelo entrenado predice futuro
    """
    st.markdown(pipeline_steps)

    st.markdown("---")

    st.subheader("📊 Usos Prácticos")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("""
        **🌾 Agricultura**
        - Planificar riego
        - Proteger ante heladas
        - Optimizar cosechas
        """)

    with col2:
        st.write("""
        **⚡ Energía**
        - Predecir demanda calefacción
        - Predecir demanda refrigeración
        - Optimizar generación
        """)

    with col3:
        st.write("""
        **🏃 Eventos**
        - Planificar actividades
        - Preparar emergencias
        - Optimizar operaciones
        """)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px; padding: 20px;'>
    <p>Desarrollado con ❤️ usando Streamlit, TensorFlow/Keras y LSTM</p>
    <p>Modelo entrenado con 10 años de datos de temperaturas mínimas de Melbourne (1981-1990)</p>
    <p><strong>Disclaimer:</strong> Este modelo es educativo. Para aplicaciones críticas, consulta fuentes meteorológicas oficiales.</p>
</div>
""", unsafe_allow_html=True)
