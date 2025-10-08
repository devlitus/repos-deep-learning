"""Página acerca del proyecto"""
import streamlit as st


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