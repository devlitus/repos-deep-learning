"""
═══════════════════════════════════════════════════════════════
SCRIPT PRINCIPAL: ENTRENAMIENTO DEL MODELO
═══════════════════════════════════════════════════════════════

Este script ejecuta TODO el pipeline de Machine Learning:

1. 📂 Carga de datos
2. 🔧 Preprocesamiento
3. 🧠 Construcción del modelo
4. 🏋️  Entrenamiento
5. 📊 Evaluación
6. 📈 Visualización
7. 💾 Guardado de resultados

═══════════════════════════════════════════════════════════════
CÓMO USAR ESTE SCRIPT
═══════════════════════════════════════════════════════════════

Desde terminal:
    python train.py

Desde Jupyter/Colab:
    %run train.py

O importar la función:
    from train import train_model
    train_model()

═══════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
print("\n" + "="*70)
print("🚀 INICIANDO PIPELINE DE MACHINE LEARNING")
print("="*70 + "\n")

print("📦 Importando módulos del proyecto...")

from data.load_data import load_melbourne_data
from src.preprocessing import create_sequences, split_data
from src.model import build_lstm_model
from src.evaluation import evaluate_model, create_evaluation_report
from src.visualization import (
    plot_temperature_history,
    plot_training_history,
    plot_predictions,
    plot_prediction_scatter,
    plot_errors
)

print("✅ Todos los módulos importados correctamente\n")


def train_model():
    """
    Función principal que ejecuta todo el pipeline
    """
    
    # PASO 1: CARGAR DATOS
    print("="*70)
    print("📂 PASO 1: CARGANDO DATOS")
    print("="*70 + "\n")
    
    df, data_normalized, scaler = load_melbourne_data()
    
    print(f"✅ Datos cargados: {len(df)} días de temperaturas")
    print(f"   Rango: {df['Temp'].min():.1f}°C - {df['Temp'].max():.1f}°C")
    print(f"   Promedio: {df['Temp'].mean():.1f}°C\n")
    
    # PASO 2: VISUALIZAR DATOS ORIGINALES
    print("="*70)
    print("📈 PASO 2: VISUALIZANDO DATOS HISTÓRICOS")
    print("="*70 + "\n")
    
    plot_temperature_history(df)
    
    # PASO 3: PREPROCESAR DATOS
    print("\n" + "="*70)
    print("🔧 PASO 3: PREPROCESANDO DATOS")
    print("="*70 + "\n")
    
    print("🔄 Creando secuencias de 60 días...")
    X, y = create_sequences(data_normalized, time_steps=60)
    
    print(f"✅ Secuencias creadas: {len(X)}")
    print(f"   Cada secuencia: 60 días → predice siguiente día\n")
    
    print("✂️  Dividiendo en conjuntos Train/Val/Test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    print(f"✅ Datos divididos:")
    print(f"   Entrenamiento: {len(X_train)} secuencias ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Validación:    {len(X_val)} secuencias ({len(X_val)/len(X)*100:.1f}%)")
    print(f"   Prueba:        {len(X_test)} secuencias ({len(X_test)/len(X)*100:.1f}%)\n")
    
    # PASO 4: CONSTRUIR MODELO
    print("="*70)
    print("🧠 PASO 4: CONSTRUYENDO MODELO LSTM")
    print("="*70 + "\n")
    
    model = build_lstm_model(time_steps=60)
    
    print("✅ Modelo construido")
    print(f"   Parámetros entrenables: {model.count_params():,}\n")
    
    print("📐 Arquitectura del modelo:")
    model.summary()
    print()
    
    # PASO 5: ENTRENAR MODELO
    print("\n" + "="*70)
    print("🏋️  PASO 5: ENTRENANDO MODELO")
    print("="*70 + "\n")
    
    print("🔥 Iniciando entrenamiento...")
    print("   Esto puede tomar 5-10 minutos dependiendo del hardware")
    print("   Progreso:\n")
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1,
        shuffle=False
    )
    
    print("\n✅ Entrenamiento completado!\n")
    
    # PASO 6: EVALUAR MODELO
    print("="*70)
    print("📊 PASO 6: EVALUANDO MODELO")
    print("="*70 + "\n")
    
    predictions, metrics = evaluate_model(model, X_test, y_test, scaler)
    
    create_evaluation_report(metrics)
    
    # PASO 7: CREAR VISUALIZACIONES
    print("\n" + "="*70)
    print("📈 PASO 7: CREANDO VISUALIZACIONES")
    print("="*70 + "\n")
    
    print("🎨 Generando gráficas...")
    
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    predictions_flat = predictions.flatten()
    
    print("   1/4 Progreso del entrenamiento...")
    plot_training_history(history)
    
    print("   2/4 Predicciones vs Realidad...")
    plot_predictions(y_test_original, predictions_flat)
    
    print("   3/4 Gráfica de dispersión...")
    plot_prediction_scatter(y_test_original, predictions_flat)
    
    print("   4/4 Análisis de errores...")
    plot_errors(y_test_original, predictions_flat)
    
    print("\n✅ Todas las gráficas guardadas en carpeta 'reports/'\n")
    
    # PASO 8: GUARDAR MODELO
    print("="*70)
    print("💾 PASO 8: GUARDANDO MODELO")
    print("="*70 + "\n")
    
    os.makedirs('models', exist_ok=True)
    
    model_path = 'models/lstm_temperatura.keras'
    model.save(model_path)
    
    print(f"✅ Modelo guardado en: {model_path}")
    print(f"   Tamaño: {os.path.getsize(model_path) / (1024*1024):.2f} MB\n")
    
    # RESUMEN FINAL
    print("="*70)
    print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70 + "\n")
    
    print("📁 Archivos generados:")
    print("   📊 reports/temperatura_historica.png - Datos originales")
    print("   📈 reports/entrenamiento.png         - Progreso del entrenamiento")
    print("   📉 reports/predicciones.png          - Predicciones vs Realidad")
    print("   🔍 reports/scatter.png               - Gráfica de dispersión")
    print("   ⚠️  reports/errores.png              - Análisis de errores")
    print("   📄 reports/metricas.txt              - Reporte de métricas")
    print("   🧠 models/lstm_temperatura.keras     - Modelo entrenado\n")
    
    print("📊 Resumen de resultados:")
    print(f"   RMSE:  {metrics['RMSE']:.4f}°C")
    print(f"   MAE:   {metrics['MAE']:.4f}°C")
    print(f"   MAPE:  {metrics['MAPE']:.2f}%")
    print(f"   R²:    {metrics['R2']:.4f} ({metrics['R2']*100:.2f}% varianza explicada)")
    
    print("\n💡 Interpretación:")
    if metrics['R2'] >= 0.9:
        print("   ✅ ¡Excelente! El modelo tiene muy buen desempeño")
    elif metrics['R2'] >= 0.7:
        print("   👍 Buen modelo, predicciones confiables")
    elif metrics['R2'] >= 0.5:
        print("   ⚠️  Modelo aceptable, puede mejorar")
    else:
        print("   ❌ Modelo necesita mejoras")
    
    print("\n🚀 Próximos pasos sugeridos:")
    print("   1. Revisar las gráficas en carpeta 'reports/'")
    print("   2. Analizar si hay overfitting en entrenamiento.png")
    print("   3. Verificar patrones de error en errores.png")
    print("   4. Si R² < 0.8, considera:")
    print("      - Aumentar épocas (epochs=100)")
    print("      - Ajustar arquitectura (más neuronas)")
    print("      - Probar diferentes sequence_length")
    
    print("\n" + "="*70)
    print("✨ ¡Gracias por usar este sistema de predicción!")
    print("="*70 + "\n")
    
    return model, history, metrics


if __name__ == "__main__":
    try:
        model, history, metrics = train_model()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Entrenamiento interrumpido por el usuario")
        print("   Los archivos parciales pueden estar en carpeta 'reports/'\n")
        
    except Exception as e:
        print(f"\n\n❌ ERROR durante el entrenamiento:")
        print(f"   {str(e)}")
        print("\n   Revisa que:")
        print("   1. El archivo daily-min-temperatures.csv existe en data/")
        print("   2. Todas las dependencias están instaladas (requirements.txt)")
        print("   3. Hay suficiente memoria RAM disponible\n")
        raise