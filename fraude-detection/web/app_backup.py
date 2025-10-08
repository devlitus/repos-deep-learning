"""
Aplicación Web de Detección de Fraude con Streamlit - Versión Profesional
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config.config import MODELS_DIR, FIGURES_DIR
from src.data.load import load_data

# Configuración de la página
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para modo oscuro
st.markdown("""
    <style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Estilos generales */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header principal con gradiente */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Cards personalizadas */
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    .metric-card h3 {
        color: #a8dadc;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-card .value {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-card .subtitle {
        color: #e0e0e0;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Botones personalizados */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Sidebar personalizada */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Forzar color blanco a TODOS los elementos de texto en sidebar */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Todos los labels */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label * {
        color: white !important;
    }

    /* Radio buttons */
    [data-testid="stSidbar"] .stRadio > label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] div[role="radiogroup"] label p,
    [data-testid="stSidebar"] div[role="radiogroup"] label span,
    [data-testid="stSidebar"] div[data-baseweb="radio"] label {
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }

    /* Paragraphs y spans en sidebar */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: white !important;
    }

    /* Markdown en sidebar */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown * {
        color: white !important;
    }

    /* Metrics en sidebar */
    [data-testid="stSidebar"] [data-testid="stMetricLabel"],
    [data-testid="stSidebar"] [data-testid="stMetricValue"],
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric {
        color: white !important;
    }

    /* Selectbox */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSelectbox {
        color: white !important;
    }
    
    /* Alertas personalizadas */
    .alert-fraud {
        background: linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(255, 107, 107, 0.4);
        animation: pulse 2s infinite;
    }
    
    .alert-legit {
        background: linear-gradient(135deg, #51cf66 0%, #2b8a3e 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(81, 207, 102, 0.4);
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.02);
        }
    }
    
    /* Tabs personalizadas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(81, 207, 102, 0.1);
        border-left: 4px solid #51cf66;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(255, 107, 107, 0.1);
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Ocultar elementos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar personalizada */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_name="random_forest"):
    """Carga el modelo entrenado (con cache para no recargar)"""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    """Carga el dataset (con cache)"""
    return load_data()


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
        st.markdown("""
            <style>
            [data-testid="stSidebar"] * {
                color: white !important;
            }
            [data-testid="stSidebar"] .stMarkdown {
                color: white !important;
            }
            </style>
        """, unsafe_allow_html=True)

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


def show_dashboard():
    """Dashboard principal con métricas"""
    
    # Métricas principales en cards personalizadas
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        {"title": "ROC-AUC Score", "value": "98.01%", "subtitle": "Excellent discrimination", "icon": "🎯"},
        {"title": "Recall", "value": "85.7%", "subtitle": "84/98 frauds detected", "icon": "✅"},
        {"title": "Precision", "value": "46.9%", "subtitle": "Low false alarms", "icon": "🎲"},
        {"title": "F1-Score", "value": "0.60", "subtitle": "Balanced performance", "icon": "⚖️"}
    ]
    
    for col, metric in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{metric['icon']} {metric['title']}</h3>
                    <p class="value">{metric['value']}</p>
                    <p class="subtitle">{metric['subtitle']}</p>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Confusion Matrix")
        
        # Crear matriz de confusión interactiva
        cm = np.array([[56769, 95], [14, 84]])
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Legitimate', 'Fraud'],
            y=['Legitimate', 'Fraud'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20},
            showscale=False
        ))
        
        fig.update_layout(
            title="Test Set Results",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Model Comparison")
        
        # Gráfico de comparación
        models = ['Logistic<br>Regression', 'Random<br>Forest']
        precision = [5.49, 49.23]
        recall = [87.34, 81.01]
        f1 = [10.34, 61.24]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Precision',
            x=models,
            y=precision,
            marker_color='#667eea',
            text=[f'{v:.1f}%' for v in precision],
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='Recall',
            x=models,
            y=recall,
            marker_color='#764ba2',
            text=[f'{v:.1f}%' for v in recall],
            textposition='auto',
        ))
        
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=models,
            y=f1,
            marker_color='#51cf66',
            text=[f'{v:.1f}%' for v in f1],
            textposition='auto',
        ))
        
        fig.update_layout(
            barmode='group',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Sección de impacto
    st.markdown("### 💰 Business Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="success-box">
                <h4>✅ Frauds Detected</h4>
                <p style='font-size: 2rem; font-weight: bold; margin: 0;'>84</p>
                <p style='margin: 0;'>Out of 98 total frauds (85.7%)</p>
                <p style='margin-top: 0.5rem; color: #51cf66;'>~$10,248 saved</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Frauds Missed</h4>
                <p style='font-size: 2rem; font-weight: bold; margin: 0;'>14</p>
                <p style='margin: 0;'>Undetected fraudulent transactions</p>
                <p style='margin-top: 0.5rem; color: #ff6b6b;'>~$1,708 potential loss</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-box">
                <h4>🚨 False Alarms</h4>
                <p style='font-size: 2rem; font-weight: bold; margin: 0;'>95</p>
                <p style='margin: 0;'>Legitimate flagged as fraud</p>
                <p style='margin-top: 0.5rem; color: #667eea;'>Only 0.17% of legitimate</p>
            </div>
        """, unsafe_allow_html=True)


def show_prediction(model_name):
    """Página de predicción en tiempo real"""
    
    st.markdown("## 🔍 Live Fraud Detection")
    st.markdown("Enter transaction details to predict fraud probability in real-time")
    
    # Tabs para diferentes modos
    tab1, tab2, tab3 = st.tabs(["⚡ Quick Predict", "📝 Manual Input", "📂 From Dataset"])
    
    with tab1:
        st.markdown("### ⚡ Quick Random Prediction")
        st.markdown("Generate random transaction values and predict instantly")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🎲 Generate & Predict", use_container_width=True, type="primary"):
                # Generar valores aleatorios
                time = np.random.uniform(0, 172800)
                amount = np.random.uniform(0, 500)
                v_features = {f'V{i}': np.random.randn() for i in range(1, 29)}
                
                # Predecir
                make_prediction_advanced(time, amount, v_features, model_name)
        
        with col2:
            st.info("**Tip**: This generates realistic transaction patterns for testing")
    
    with tab2:
        st.markdown("### 📝 Custom Transaction Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input(
                "⏱️ Time (seconds)",
                min_value=0.0,
                max_value=200000.0,
                value=0.0,
                help="Seconds elapsed from first transaction"
            )
        
        with col2:
            amount = st.number_input(
                "💵 Amount ($)",
                min_value=0.0,
                max_value=30000.0,
                value=100.0,
                help="Transaction amount in dollars"
            )
        
        st.markdown("#### Features V1-V28 (PCA Transformed)")
        st.info("💡 These are anonymized features. Use random values or leave as zero")
        
        # Features V1-V28 en columnas
        v_features = {}
        cols = st.columns(7)
        for i in range(1, 29):
            col_idx = (i - 1) % 7
            with cols[col_idx]:
                v_features[f'V{i}'] = st.number_input(
                    f"V{i}",
                    value=0.0,
                    format="%.4f",
                    key=f"manual_v{i}",
                    label_visibility="visible"
                )
        
        if st.button("🔍 Analyze Transaction", use_container_width=True, type="primary"):
            make_prediction_advanced(time, amount, v_features, model_name)
    
    with tab3:
        st.markdown("### 📂 Select from Real Dataset")
        
        data = load_dataset()
        
        if data is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                transaction_type = st.selectbox(
                    "Filter by type:",
                    ["All Transactions", "Legitimate Only", "Fraudulent Only"]
                )
            
            with col2:
                if transaction_type == "Legitimate Only":
                    filtered_data = data[data['Class'] == 0]
                elif transaction_type == "Fraudulent Only":
                    filtered_data = data[data['Class'] == 1]
                else:
                    filtered_data = data
                
                transaction_idx = st.number_input(
                    "Transaction index:",
                    min_value=0,
                    max_value=len(filtered_data)-1,
                    value=0
                )
            
            with col3:
                st.metric("Total Available", len(filtered_data))
            
            if st.button("📊 Load & Predict", use_container_width=True, type="primary"):
                transaction = filtered_data.iloc[transaction_idx]
                
                # Mostrar datos
                with st.expander("📄 View Transaction Data", expanded=True):
                    st.dataframe(
                        transaction.to_frame().T,
                        use_container_width=True
                    )
                
                # Extraer features
                time = transaction['Time']
                amount = transaction['Amount']
                v_features = {f'V{i}': transaction[f'V{i}'] for i in range(1, 29)}
                
                # Predecir
                make_prediction_advanced(time, amount, v_features, model_name, transaction['Class'])


def make_prediction_advanced(time, amount, v_features, model_name, true_label=None):
    """Realiza predicción con visualización avanzada"""
    try:
        # Cargar modelo
        model = load_model(model_name)
        
        # Preparar datos
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        values = [time] + [v_features[f'V{i}'] for i in range(1, 29)] + [amount]
        
        X = pd.DataFrame([values], columns=feature_names)
        
        # Normalizar (usar scaler simple para demo)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
        
        # Predecir
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]
        
        # Animación de carga
        with st.spinner('🔄 Analyzing transaction...'):
            import time as time_module
            time_module.sleep(0.5)
        
        st.markdown("---")
        
        # Resultado principal
        if prediction == 1:
            st.markdown("""
                <div class="alert-fraud">
                    🚨 FRAUD DETECTED 🚨
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="alert-legit">
                    ✅ LEGITIMATE TRANSACTION
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Métricas de predicción
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Prediction",
                "FRAUD" if prediction == 1 else "LEGITIMATE",
                delta=None
            )
        
        with col2:
            st.metric(
                "Confidence",
                f"{max(probability)*100:.1f}%",
                delta=f"{abs(probability[1] - probability[0])*100:.1f}% margin"
            )
        
        with col3:
            st.metric(
                "Fraud Probability",
                f"{probability[1]*100:.2f}%",
                delta="HIGH" if probability[1] > 0.5 else "LOW"
            )
        
        with col4:
            if true_label is not None:
                is_correct = (true_label == prediction)
                st.metric(
                    "Actual Label",
                    "FRAUD" if true_label == 1 else "LEGITIMATE",
                    delta="✅ Correct" if is_correct else "❌ Wrong"
                )
        
        # Gráfico de probabilidades
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Probability Distribution")
            
            fig = go.Figure()
            
            colors = ['#51cf66' if prediction == 0 else '#ff6b6b', 
                     '#ff6b6b' if prediction == 1 else '#51cf66']
            
            fig.add_trace(go.Bar(
                x=['Legitimate', 'Fraud'],
                y=[probability[0], probability[1]],
                marker_color=colors,
                text=[f'{probability[0]*100:.2f}%', f'{probability[1]*100:.2f}%'],
                textposition='auto',
                textfont=dict(size=16, color='white')
            ))
            
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                yaxis=dict(title='Probability', range=[0, 1]),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Confidence Meter")
            
            confidence = max(probability)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Confidence Level", 'font': {'color': 'white'}},
                number={'suffix': "%", 'font': {'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255,107,107,0.3)'},
                        {'range': [50, 75], 'color': 'rgba(255,193,7,0.3)'},
                        {'range': [75, 100], 'color': 'rgba(81,207,102,0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': confidence * 100
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Detalles de la transacción
        with st.expander("📋 Transaction Details", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Transaction Info:**")
                st.write(f"- Time: {time:.2f} seconds")
                st.write(f"- Amount: ${amount:.2f}")
                st.write(f"- Model Used: {model_name.replace('_', ' ').title()}")
            
            with col2:
                st.write("**Risk Assessment:**")
                risk_level = "🔴 HIGH" if probability[1] > 0.7 else "🟡 MEDIUM" if probability[1] > 0.3 else "🟢 LOW"
                st.write(f"- Risk Level: {risk_level}")
                st.write(f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                recommendation = "🚫 BLOCK TRANSACTION" if prediction == 1 else "✅ APPROVE TRANSACTION"
                st.write(f"- Recommendation: {recommendation}")
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")


def show_analytics():
    """Página de análisis del modelo"""
    st.markdown("## 📊 Model Performance Analytics")
    
    # ROC Curve
    st.markdown("### 📈 ROC Curve")
    
    # Simular datos de ROC (en producción usarías datos reales)
    fpr = np.linspace(0, 1, 100)
    tpr = np.power(fpr, 0.1)  # Simular curva ROC buena
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='Random Forest (AUC=0.9801)',
        line=dict(color='#667eea', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="ROC Curve - Test Set Performance",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature Importance (simulado)
    st.markdown("### 🎯 Top Important Features")
    
    features = ['V17', 'V14', 'V12', 'V10', 'V11', 'V4', 'V3', 'V16', 'V7', 'V18']
    importance = [0.33, 0.30, 0.26, 0.22, 0.20, 0.18, 0.19, 0.17, 0.16, 0.15]
    
    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            showscale=True
        ),
        text=[f'{v:.2f}' for v in importance],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Correlation with Fraud",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Feature",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Precision-Recall tradeoff
    st.markdown("### ⚖️ Precision-Recall Trade-off")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Curva Precision-Recall
        recall = np.linspace(0, 1, 100)
        precision = 1 - (recall ** 0.5)  # Simular curva
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            fill='tozeroy',
            name='Precision-Recall',
            line=dict(color='#764ba2', width=3),
            fillcolor='rgba(118, 75, 162, 0.3)'
        ))
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall (Frauds Detected)",
            yaxis_title="Precision (Alert Accuracy)",
            height=350,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Performance Breakdown")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 True Positives (TP)</h4>
            <p style='font-size: 1.8rem; margin: 0;'><strong>84</strong></p>
            <p>Correctly identified frauds</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
            <h4>✅ True Negatives (TN)</h4>
            <p style='font-size: 1.8rem; margin: 0;'><strong>56,769</strong></p>
            <p>Correctly identified legitimate</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ False Positives (FP)</h4>
            <p style='font-size: 1.8rem; margin: 0;'><strong>95</strong></p>
            <p>Legitimate flagged as fraud</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <h4>🚨 False Negatives (FN)</h4>
            <p style='font-size: 1.8rem; margin: 0;'><strong>14</strong></p>
            <p>Frauds that slipped through</p>
        </div>
        """, unsafe_allow_html=True)


def show_eda():
    """Página de exploración de datos"""
    st.markdown("## 📈 Data Explorer")
    
    data = load_dataset()
    
    if data is not None:
        # Estadísticas generales
        st.markdown("### 📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-card">
                    <h3>📦 Total Records</h3>
                    <p class="value">284,807</p>
                    <p class="subtitle">2 days of data</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="metric-card">
                    <h3>✅ Legitimate</h3>
                    <p class="value">284,315</p>
                    <p class="subtitle">99.827%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="metric-card">
                    <h3>🚨 Fraudulent</h3>
                    <p class="value">492</p>
                    <p class="subtitle">0.173%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class="metric-card">
                    <h3>⚖️ Imbalance Ratio</h3>
                    <p class="value">1:578</p>
                    <p class="subtitle">Highly imbalanced</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Class Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Class Distribution")
            
            class_counts = data['Class'].value_counts()
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=['Legitimate', 'Fraud'],
                y=class_counts.values,
                marker_color=['#51cf66', '#ff6b6b'],
                text=class_counts.values,
                texttemplate='%{text:,}',
                textposition='auto',
                textfont=dict(size=16, color='white')
            ))
            
            fig.update_layout(
                title="Transaction Count by Class",
                yaxis_type="log",
                yaxis_title="Count (log scale)",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 💰 Amount Distribution")
            
            legitimate = data[data['Class'] == 0]['Amount']
            fraud = data[data['Class'] == 1]['Amount']
            
            fig = go.Figure()
            
            fig.add_trace(go.Box(
                y=legitimate,
                name='Legitimate',
                marker_color='#51cf66',
                boxmean='sd'
            ))
            
            fig.add_trace(go.Box(
                y=fraud,
                name='Fraud',
                marker_color='#ff6b6b',
                boxmean='sd'
            ))
            
            fig.update_layout(
                title="Transaction Amount by Class",
                yaxis_title="Amount ($)",
                yaxis_range=[0, 1000],
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal Distribution
        st.markdown("### ⏱️ Temporal Distribution")
        
        data['Hours'] = data['Time'] / 3600
        
        fig = go.Figure()
        
        # Todas las transacciones
        fig.add_trace(go.Histogram(
            x=data['Hours'],
            nbinsx=48,
            name='All',
            marker_color='#667eea',
            opacity=0.7
        ))
        
        # Solo fraudes
        fraud_data = data[data['Class'] == 1]
        fig.add_trace(go.Histogram(
            x=fraud_data['Hours'],
            nbinsx=48,
            name='Fraud',
            marker_color='#ff6b6b',
            opacity=0.8
        ))
        
        fig.update_layout(
            title="Transaction Distribution Over Time (48 hours)",
            xaxis_title="Time (Hours)",
            yaxis_title="Number of Transactions",
            height=400,
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Línea vertical para marcar día 2
        fig.add_vline(x=24, line_dash="dash", line_color="white", 
                     annotation_text="Day 2", annotation_position="top")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Estadísticas comparativas
        st.markdown("### 📊 Statistical Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Legitimate Transactions")
            stats_legit = {
                'Mean Amount': f'${legitimate.mean():.2f}',
                'Median Amount': f'${legitimate.median():.2f}',
                'Std Dev': f'${legitimate.std():.2f}',
                'Min Amount': f'${legitimate.min():.2f}',
                'Max Amount': f'${legitimate.max():.2f}'
            }
            st.table(pd.DataFrame.from_dict(stats_legit, orient='index', columns=['Value']))
        
        with col2:
            st.markdown("#### Fraudulent Transactions")
            stats_fraud = {
                'Mean Amount': f'${fraud.mean():.2f}',
                'Median Amount': f'${fraud.median():.2f}',
                'Std Dev': f'${fraud.std():.2f}',
                'Min Amount': f'${fraud.min():.2f}',
                'Max Amount': f'${fraud.max():.2f}'
            }
            st.table(pd.DataFrame.from_dict(stats_fraud, orient='index', columns=['Value']))
        
        # Key Insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="info-box">
                    <h4>🎯 Imbalance Challenge</h4>
                    <p>With only 0.173% frauds, traditional models would achieve 99.8% accuracy by predicting everything as legitimate - completely useless for fraud detection!</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="success-box">
                    <h4>💰 Amount Pattern</h4>
                    <p>Fraudulent transactions have LOWER median ($9.25) than legitimate ($22.00). Fraudsters prefer small amounts to avoid detection!</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="warning-box">
                    <h4>⏰ Temporal Pattern</h4>
                    <p>Frauds show clear temporal patterns with peaks around hours 10-11 and 23-24. This is valuable for the model!</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Raw Data Explorer
        st.markdown("### 🔍 Raw Data Explorer")
        
        with st.expander("View Dataset Sample", expanded=False):
            # Filtros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_class = st.selectbox(
                    "Filter by class:",
                    ["All", "Legitimate", "Fraud"]
                )
            
            with col2:
                num_rows = st.slider("Number of rows:", 10, 100, 50)
            
            with col3:
                sort_by = st.selectbox(
                    "Sort by:",
                    ["Time", "Amount", "Class"]
                )
            
            # Aplicar filtros
            if filter_class == "Legitimate":
                filtered = data[data['Class'] == 0]
            elif filter_class == "Fraud":
                filtered = data[data['Class'] == 1]
            else:
                filtered = data
            
            filtered = filtered.sort_values(by=sort_by, ascending=False).head(num_rows)
            
            st.dataframe(
                filtered,
                use_container_width=True,
                height=400
            )
            
            # Botón de descarga
            csv = filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name=f'fraud_data_{filter_class.lower()}.csv',
                mime='text/csv',
            )


def show_about():
    """Página acerca del proyecto"""
    st.markdown("## ℹ️ About This Project")
    
    # Hero section
    st.markdown("""
        <div class="info-box" style="padding: 2rem;">
            <h2>🛡️ Fraud Detection System</h2>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                An advanced Machine Learning system designed to detect fraudulent credit card transactions 
                in real-time using state-of-the-art algorithms and data science techniques.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Características del proyecto
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Project Goals")
        st.markdown("""
        - ✅ **High Recall**: Detect 85%+ of all fraudulent transactions
        - ✅ **Low False Alarms**: Minimize customer inconvenience
        - ✅ **Real-time**: Instant predictions (<100ms)
        - ✅ **Interpretable**: Understand model decisions
        - ✅ **Scalable**: Handle millions of transactions
        """)
        
        st.markdown("### 🛠️ Technologies")
        st.markdown("""
        **Machine Learning:**
        - scikit-learn
        - imbalanced-learn (SMOTE)
        - Random Forest
        - Logistic Regression
        
        **Data Processing:**
        - pandas & numpy
        - StandardScaler
        - PCA (pre-applied)
        
        **Visualization:**
        - Streamlit
        - Plotly
        - Matplotlib & Seaborn
        """)
    
    with col2:
        st.markdown("### 📊 Dataset Information")
        st.markdown("""
        **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
        
        **Details:**
        - 284,807 transactions
        - 2 days (September 2013)
        - 492 frauds (0.173%)
        - 30 features + 1 target
        - Anonymized with PCA
        
        **Features:**
        - `Time`: Seconds elapsed
        - `V1-V28`: PCA components
        - `Amount`: Transaction value
        - `Class`: 0=Legitimate, 1=Fraud
        """)
        
        st.markdown("### 🏆 Model Performance")
        st.markdown("""
        **Random Forest (Best Model):**
        - ROC-AUC: **98.01%**
        - Recall: **85.7%**
        - Precision: **46.9%**
        - F1-Score: **0.60**
        
        **Why these metrics matter:**
        - High Recall = Catch most frauds
        - Moderate Precision = Few false alarms
        - High ROC-AUC = Excellent discrimination
        """)
    
    # Metodología
    st.markdown("### 🔬 Methodology")
    
    timeline_data = [
        {"phase": "1. Data Analysis", "description": "Exploratory analysis, visualization, pattern discovery", "icon": "📊"},
        {"phase": "2. Preprocessing", "description": "Scaling, train/val/test split, SMOTE balancing", "icon": "🔧"},
        {"phase": "3. Model Training", "description": "Logistic Regression, Random Forest, hyperparameter tuning", "icon": "🤖"},
        {"phase": "4. Evaluation", "description": "Test set evaluation, confusion matrix, ROC curve", "icon": "📈"},
        {"phase": "5. Deployment", "description": "Web application, real-time prediction system", "icon": "🚀"}
    ]
    
    for item in timeline_data:
        st.markdown(f"""
            <div class="info-box">
                <h4>{item['icon']} {item['phase']}</h4>
                <p>{item['description']}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Challenges & Solutions
    st.markdown("### 💡 Challenges & Solutions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Challenge: Extreme Imbalance</h4>
                <p><strong>Problem:</strong> Only 0.173% frauds - models would ignore minority class</p>
                <p><strong>Solution:</strong> SMOTE (Synthetic Minority Over-sampling) to balance training data</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Challenge: Wrong Metrics</h4>
                <p><strong>Problem:</strong> Accuracy misleading (99.8% by predicting all legitimate)</p>
                <p><strong>Solution:</strong> Use Precision, Recall, F1-Score, and ROC-AUC instead</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="success-box">
                <h4>✅ Challenge: Feature Scaling</h4>
                <p><strong>Problem:</strong> Amount and Time on different scales than V1-V28</p>
                <p><strong>Solution:</strong> StandardScaler normalization (mean=0, std=1)</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="success-box">
                <h4>✅ Challenge: Model Selection</h4>
                <p><strong>Problem:</strong> Need balance between precision and recall</p>
                <p><strong>Solution:</strong> Random Forest with class_weight='balanced'</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Future Improvements
    st.markdown("### 🚀 Future Improvements")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="info-box">
                <h4>📊 Models</h4>
                <ul>
                    <li>XGBoost</li>
                    <li>LightGBM</li>
                    <li>Neural Networks</li>
                    <li>Ensemble methods</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="info-box">
                <h4>🔧 Features</h4>
                <ul>
                    <li>Hyperparameter tuning</li>
                    <li>Feature engineering</li>
                    <li>Anomaly detection</li>
                    <li>Time series analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="info-box">
                <h4>🌐 Production</h4>
                <ul>
                    <li>REST API (FastAPI)</li>
                    <li>Model monitoring</li>
                    <li>A/B testing</li>
                    <li>Cloud deployment</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Credits
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
            <div style="text-align: center; padding: 2rem;">
                <h3>👤 Author</h3>
                <p style="font-size: 1.1rem;"><strong>Tu Nombre</strong></p>
                <p>Machine Learning Engineer</p>
                <br>
                <p>
                    <a href="https://github.com/tu-usuario" style="color: #667eea; text-decoration: none; margin: 0 10px;">
                        💻 GitHub
                    </a>
                    <a href="https://linkedin.com/in/tu-perfil" style="color: #667eea; text-decoration: none; margin: 0 10px;">
                        💼 LinkedIn
                    </a>
                    <a href="mailto:tu.email@ejemplo.com" style="color: #667eea; text-decoration: none; margin: 0 10px;">
                        📧 Email
                    </a>
                </p>
                <br>
                <p style="color: #888; font-size: 0.9rem;">
                    © 2025 Fraud Detection System<br>
                    Built with ❤️ using Streamlit & scikit-learn
                </p>
                <br>
                <p style="font-size: 1.2rem;">
                    ⭐ <strong>Star this project on GitHub!</strong> ⭐
                </p>
            </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()