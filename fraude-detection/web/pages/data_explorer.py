"""Página de exploración de datos"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.data_loader import load_dataset


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