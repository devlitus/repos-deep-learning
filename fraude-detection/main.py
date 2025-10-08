"""
Pipeline Principal del Proyecto de Detección de Fraude
Ejecuta todo el flujo de trabajo de principio a fin
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import argparse
from src.data.load import load_data, explore_data
from src.data.preprocess import preprocess_pipeline
from src.visualization.plots import generate_all_plots
from src.models.train import (
    train_logistic_regression,
    train_random_forest,
    save_model,
    compare_models
)
from src.models.evaluate import (
    load_model,
    evaluate_on_test,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)


def print_header(text):
    """Imprime un encabezado visual"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def run_full_pipeline(skip_eda=False, skip_training=False, model_name="random_forest"):
    """
    Ejecuta el pipeline completo del proyecto
    
    Args:
        skip_eda (bool): Si True, omite el análisis exploratorio
        skip_training (bool): Si True, omite el entrenamiento (usa modelos guardados)
        model_name (str): Nombre del modelo a evaluar
    """
    
    print_header("🚀 PROYECTO: DETECCIÓN DE FRAUDE CON TARJETAS DE CRÉDITO")
    
    # =========================================================================
    # PASO 1: CARGA DE DATOS
    # =========================================================================
    print_header("📂 PASO 1: CARGA DE DATOS")
    data = load_data()
    
    if data is None:
        print("❌ Error: No se pudo cargar el dataset. Abortando pipeline.")
        return
    
    explore_data(data)
    
    # =========================================================================
    # PASO 2: ANÁLISIS EXPLORATORIO (EDA)
    # =========================================================================
    if not skip_eda:
        print_header("📊 PASO 2: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("⏳ Generando visualizaciones (esto puede tardar 1-2 minutos)...")
        generate_all_plots(data)
        print("✅ Visualizaciones generadas en reports/figures/")
    else:
        print_header("⏭️ PASO 2: EDA OMITIDO")
    
    # =========================================================================
    # PASO 3: PREPROCESAMIENTO
    # =========================================================================
    print_header("🔧 PASO 3: PREPROCESAMIENTO DE DATOS")
    processed_data = preprocess_pipeline(data, apply_balancing=True)
    
    X_train = processed_data['X_train']
    y_train = processed_data['y_train']
    X_val = processed_data['X_val']
    y_val = processed_data['y_val']
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']
    
    print("\n✅ Datos preprocesados y listos para entrenamiento")
    
    # =========================================================================
    # PASO 4: ENTRENAMIENTO DE MODELOS
    # =========================================================================
    if not skip_training:
        print_header("🤖 PASO 4: ENTRENAMIENTO DE MODELOS")
        
        all_metrics = []
        
        # Entrenar Logistic Regression
        print("\n1️⃣ Entrenando Logistic Regression...")
        lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
        save_model(lr_model, "logistic_regression")
        all_metrics.append(lr_metrics)
        
        # Entrenar Random Forest
        print("\n2️⃣ Entrenando Random Forest...")
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
        save_model(rf_model, "random_forest")
        all_metrics.append(rf_metrics)
        
        # Comparar modelos
        compare_models(all_metrics)
        
        print("\n✅ Modelos entrenados y guardados en models/")
    else:
        print_header("⏭️ PASO 4: ENTRENAMIENTO OMITIDO (usando modelos guardados)")
    
    # =========================================================================
    # PASO 5: EVALUACIÓN FINAL EN TEST SET
    # =========================================================================
    print_header("🧪 PASO 5: EVALUACIÓN FINAL EN TEST SET")
    
    try:
        # Cargar el mejor modelo
        print(f"\n📦 Cargando modelo: {model_name}")
        model = load_model(model_name)
        
        # Evaluar en test
        y_pred, y_pred_proba, cm = evaluate_on_test(model, X_test, y_test, model_name.replace('_', ' ').title())
        
        # Generar visualizaciones
        print("\n🎨 Generando visualizaciones finales...")
        plot_confusion_matrix(cm, model_name.replace('_', ' ').title())
        plot_roc_curve(y_test, y_pred_proba, model_name.replace('_', ' ').title())
        plot_precision_recall_curve(y_test, y_pred_proba, model_name.replace('_', ' ').title())
        
        print("\n✅ Evaluación completada")
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el modelo '{model_name}.pkl'")
        print("   Ejecuta el pipeline con '--train' para entrenar los modelos primero.")
        return
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print_header("📋 RESUMEN DEL PROYECTO")
    
    print("\n✅ Pipeline completado exitosamente!")
    print("\n📁 Archivos generados:")
    print("   • Visualizaciones EDA: reports/figures/")
    print("   • Modelos entrenados: models/")
    print("   • Gráficos de evaluación: reports/figures/")
    
    print("\n🎯 Próximos pasos sugeridos:")
    print("   1. Revisar las visualizaciones en reports/figures/")
    print("   2. Analizar las métricas del modelo")
    print("   3. Ajustar hiperparámetros si es necesario")
    print("   4. Considerar probar modelos adicionales (XGBoost, etc.)")
    
    print("\n" + "="*70)
    print("  ¡Gracias por usar el proyecto de Detección de Fraude!")
    print("="*70 + "\n")


def main():
    """
    Función principal con argumentos de línea de comandos
    """
    parser = argparse.ArgumentParser(
        description='Pipeline de Detección de Fraude con Tarjetas de Crédito',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                              # Ejecutar pipeline completo
  python main.py --skip-eda                   # Omitir análisis exploratorio
  python main.py --skip-training              # Usar modelos ya entrenados
  python main.py --model logistic_regression  # Evaluar modelo específico
  python main.py --skip-eda --skip-training   # Solo evaluación
        """
    )
    
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Omitir el análisis exploratorio de datos (EDA)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Omitir el entrenamiento de modelos (usar modelos guardados)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='random_forest',
        choices=['logistic_regression', 'random_forest'],
        help='Modelo a evaluar en el test set (default: random_forest)'
    )
    
    args = parser.parse_args()
    
    # Ejecutar pipeline
    run_full_pipeline(
        skip_eda=args.skip_eda,
        skip_training=args.skip_training,
        model_name=args.model
    )


if __name__ == "__main__":
    main()