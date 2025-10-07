import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Supervivencia - Titanic",
    page_icon="🚢",
    layout="wide"
)

# ============================================
# FUNCIÓN PARA CARGAR EL MODELO
# ============================================
@st.cache_resource
def load_model():
    """
    Carga el modelo entrenado desde la ruta configurada
    @st.cache_resource mantiene el modelo en memoria
    """
    if os.path.exists(config.MODEL_FILE):
        model = joblib.load(config.MODEL_FILE)
        # Mostrar solo el nombre del archivo para mejor UX
        display_name = os.path.basename(config.MODEL_FILE)
        st.success(f"✅ Modelo cargado: {display_name}")
        return model
    
    st.error("❌ No se encontró el modelo. Asegúrate de haber entrenado el modelo primero.")
    st.info("💡 Ejecuta: python main.py desde el directorio predictor-titanic")
    return None

# ============================================
# CARGAR MODELO
# ============================================
model = load_model()

# ============================================
# TÍTULO Y DESCRIPCIÓN
# ============================================
st.title("🚢 Predictor de Supervivencia del Titanic")
st.markdown("""
Esta aplicación usa **Machine Learning** para predecir si un pasajero del Titanic 
habría sobrevivido al hundimiento basándose en sus características.

**Modelo:** Random Forest Classifier  
**Accuracy:** 81%
""")

st.divider()

# ============================================
# SIDEBAR - INFORMACIÓN
# ============================================
with st.sidebar:
    st.header("📊 Información del Modelo")
    st.markdown("""
    ### Variables más importantes:
    1. **Sexo** (36%)
    2. **Precio del ticket** (21%)
    3. **Edad** (17%)
    4. **Clase social** (12%)
    
    ### Métricas del modelo:
    - **Accuracy:** 81%
    - **Precision:** 80%
    - **Recall:** 68%
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📖 Contexto Histórico
    El RMS Titanic se hundió el 15 de abril de 1912.
    De los 2,224 pasajeros y tripulación, solo sobrevivieron 710 personas (32%).
    """)

# ============================================
# FORMULARIO DE ENTRADA
# ============================================
st.header("👤 Ingresa los Datos del Pasajero")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Información Personal")
    
    # Sexo
    sex = st.selectbox(
        "Sexo",
        options=["Mujer", "Hombre"],
        help="El sexo fue el factor más importante en la supervivencia"
    )
    sex_encoded = 1 if sex == "Mujer" else 0
    
    # Edad
    age = st.slider(
        "Edad",
        min_value=0.0,
        max_value=80.0,
        value=28.0,
        step=1.0,
        help="Los niños tuvieron mayor probabilidad de sobrevivir"
    )
    
    # Clase
    pclass = st.selectbox(
        "Clase del Ticket",
        options=[1, 2, 3],
        format_func=lambda x: f"Clase {x} ({'Primera' if x==1 else 'Segunda' if x==2 else 'Tercera'})",
        help="La primera clase tenía más acceso a botes salvavidas"
    )

with col2:
    st.subheader("Información Familiar")
    
    # Hermanos/Cónyuges
    sibsp = st.number_input(
        "Número de Hermanos/Cónyuges a bordo",
        min_value=0,
        max_value=8,
        value=0,
        step=1,
        help="SibSp = Siblings/Spouses"
    )
    
    # Padres/Hijos
    parch = st.number_input(
        "Número de Padres/Hijos a bordo",
        min_value=0,
        max_value=6,
        value=0,
        step=1,
        help="Parch = Parents/Children"
    )
    
    # Calcular family_size y is_alone automáticamente
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    
    st.info(f"👨‍👩‍👧‍👦 Tamaño de familia: {family_size}")
    if is_alone:
        st.warning("⚠️ Viaja solo")
    else:
        st.success(f"✅ Viaja acompañado")

with col3:
    st.subheader("Información del Ticket")
    
    # Precio del ticket
    fare = st.number_input(
        "Precio del Ticket (£)",
        min_value=0.0,
        max_value=512.0,
        value=32.0,
        step=1.0,
        help="Precio en libras esterlinas. Promedio: £32"
    )
    
    # Puerto de embarque
    embarked = st.selectbox(
        "Puerto de Embarque",
        options=["Cherbourg", "Queenstown", "Southampton"],
        index=2,
        help="Puerto donde abordó el Titanic"
    )
    embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
    embarked_encoded = embarked_map[embarked]

st.divider()

# ============================================
# BOTÓN DE PREDICCIÓN
# ============================================
if st.button("🔮 Predecir Supervivencia", type="primary", use_container_width=True):
    
    if model is None:
        st.error("❌ No se puede hacer la predicción. El modelo no está cargado.")
    else:
        # Crear el array de features en el orden correcto
        # Orden: pclass, sex, age, sibsp, parch, fare, embarked, family_size, is_alone
        features = np.array([[
            pclass,
            sex_encoded,
            age,
            sibsp,
            parch,
            fare,
            embarked_encoded,
            family_size,
            is_alone
        ]])
        
        # Hacer predicción
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Probabilidades
        prob_no_survived = prediction_proba[0] * 100
        prob_survived = prediction_proba[1] * 100
        
        # ============================================
        # MOSTRAR RESULTADO
        # ============================================
        st.header("🎯 Resultado de la Predicción")
        
        # Crear dos columnas para el resultado
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 1:
                st.success("### ✅ SOBREVIVIÓ")
                st.balloons()
            else:
                st.error("### ❌ NO SOBREVIVIÓ")
        
        with result_col2:
            st.metric(
                label="Probabilidad de Supervivencia",
                value=f"{prob_survived:.1f}%",
                delta=f"{prob_survived - 38.4:.1f}% vs promedio"
            )
        
        # ============================================
        # BARRA DE PROBABILIDAD
        # ============================================
        st.subheader("📊 Probabilidades")
        
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("No Sobrevivió", f"{prob_no_survived:.1f}%")
            st.progress(prob_no_survived / 100)
        
        with prob_col2:
            st.metric("Sobrevivió", f"{prob_survived:.1f}%")
            st.progress(prob_survived / 100)
        
        # ============================================
        # ANÁLISIS DE FACTORES
        # ============================================
        st.divider()
        st.subheader("🔍 Análisis de Factores")
        
        # Factores positivos y negativos
        positive_factors = []
        negative_factors = []
        
        # Análisis de sexo
        if sex == "Mujer":
            positive_factors.append("✅ **Mujer** - 74% de las mujeres sobrevivieron")
        else:
            negative_factors.append("❌ **Hombre** - Solo 19% de los hombres sobrevivieron")
        
        # Análisis de clase
        if pclass == 1:
            positive_factors.append("✅ **Primera Clase** - 63% de supervivencia")
        elif pclass == 3:
            negative_factors.append("❌ **Tercera Clase** - Solo 24% de supervivencia")
        
        # Análisis de edad
        if age < 12:
            positive_factors.append("✅ **Niño** - Los niños tuvieron prioridad")
        elif age > 60:
            negative_factors.append("❌ **Mayor de 60 años** - Menor tasa de supervivencia")
        
        # Análisis de precio
        if fare > 50:
            positive_factors.append(f"✅ **Ticket caro (£{fare:.0f})** - Indica mejor acceso a botes")
        elif fare < 15:
            negative_factors.append(f"❌ **Ticket económico (£{fare:.0f})** - Menor acceso a botes")
        
        # Mostrar factores
        factor_col1, factor_col2 = st.columns(2)
        
        with factor_col1:
            st.markdown("### Factores Positivos 👍")
            if positive_factors:
                for factor in positive_factors:
                    st.markdown(factor)
            else:
                st.info("No se identificaron factores muy positivos")
        
        with factor_col2:
            st.markdown("### Factores Negativos 👎")
            if negative_factors:
                for factor in negative_factors:
                    st.markdown(factor)
            else:
                st.info("No se identificaron factores muy negativos")
        
        # ============================================
        # DATOS INGRESADOS
        # ============================================
        with st.expander("📋 Ver Datos Ingresados"):
            data_dict = {
                "Variable": ["Clase", "Sexo", "Edad", "Hermanos/Cónyuges", "Padres/Hijos", 
                            "Precio", "Puerto", "Tamaño Familia", "Viaja Solo"],
                "Valor": [pclass, sex, age, sibsp, parch, f"£{fare:.2f}", 
                         embarked, family_size, "Sí" if is_alone else "No"]
            }
            df_data = pd.DataFrame(data_dict)
            st.dataframe(df_data, use_container_width=True, hide_index=True)

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🚢 Proyecto Educativo - Predictor de Supervivencia del Titanic</p>
    <p>Modelo: Random Forest Classifier | Accuracy: 81%</p>
</div>
""", unsafe_allow_html=True)