@echo off
REM Script para ejecutar la web app de Sentiment Analysis en Windows
REM Uso: run.bat

echo.
echo 🚀 Iniciando Sentiment Analysis Web App...
echo.

REM Verificar si streamlit está instalado
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Error: Streamlit no está instalado
    echo 💡 Instala con: pip install streamlit
    pause
    exit /b 1
)

REM Verificar si el modelo existe
set MODEL_PATH=..\models\lstm_sentiment_model.keras
if not exist "%MODEL_PATH%" (
    echo ⚠️  Advertencia: Modelo no encontrado en %MODEL_PATH%
    echo 💡 Ejecuta 'python ..\main.py' para entrenar el modelo
    echo.
)

REM Ejecutar app
echo ✅ Lanzando aplicación en http://localhost:8501
echo.
streamlit run app.py
