"""Página de análisis del modelo"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go


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
        title="Feature Importance",
        xaxis_title="Importance Score",
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
