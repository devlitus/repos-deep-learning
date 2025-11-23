#!/bin/bash

# Script para ejecutar la web app de Sentiment Analysis
# Uso: ./run.sh

echo "🚀 Iniciando Sentiment Analysis Web App..."
echo ""

# Verificar si streamlit está instalado
if ! command -v streamlit &> /dev/null; then
    echo "❌ Error: Streamlit no está instalado"
    echo "💡 Instala con: pip install streamlit"
    exit 1
fi

# Verificar si el modelo existe
MODEL_PATH="../models/lstm_sentiment_model.keras"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Advertencia: Modelo no encontrado en $MODEL_PATH"
    echo "💡 Ejecuta 'python ../main.py' para entrenar el modelo"
    echo ""
fi

# Ejecutar app
echo "✅ Lanzando aplicación en http://localhost:8501"
echo ""
streamlit run app.py
