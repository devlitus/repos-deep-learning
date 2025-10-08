"""Página de predicción en tiempo real"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler

from utils.data_loader import load_dataset, load_model


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
