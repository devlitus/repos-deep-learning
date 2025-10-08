"""Dashboard principal con métricas"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go


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
