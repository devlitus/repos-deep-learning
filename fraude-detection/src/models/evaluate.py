"""
Módulo para evaluación final de modelos en el test set
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc
)
import joblib
from config.config import MODELS_DIR, FIGURES_DIR


def load_model(model_name):
    """
    Carga un modelo guardado
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Modelo cargado
    """
    model_path = MODELS_DIR / f"{model_name}.pkl"
    print(f"📂 Cargando modelo desde: {model_path}")
    model = joblib.load(model_path)
    print(f"✅ Modelo cargado: {model_name}")
    return model


def evaluate_on_test(model, X_test, y_test, model_name):
    """
    Evalúa el modelo en el test set
    
    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Target de test
        model_name: Nombre del modelo
    """
    print("\n" + "="*60)
    print(f"🧪 EVALUACIÓN FINAL EN TEST SET - {model_name}")
    print("="*60)
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Reporte de clasificación
    print("\n📊 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=['Legítima', 'Fraude']))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n📋 Matriz de Confusión:")
    print(f"                 Predicho")
    print(f"               Legítima  Fraude")
    print(f"Real Legítima    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"     Fraude      {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    # Análisis de errores
    tn, fp, fn, tp = cm.ravel()
    print(f"\n💰 ANÁLISIS DE IMPACTO:")
    print(f"   • Fraudes detectados: {tp}/{tp+fn} ({tp/(tp+fn)*100:.1f}%)")
    print(f"   • Fraudes perdidos: {fn} ⚠️")
    print(f"   • Falsas alarmas: {fp}")
    print(f"   • Clientes correctos: {tn:,}")
    
    return y_pred, y_pred_proba, cm


def plot_confusion_matrix(cm, model_name, save=True):
    """
    Visualiza la matriz de confusión
    
    Args:
        cm: Matriz de confusión
        model_name: Nombre del modelo
        save: Si True, guarda la figura
    """
    plt.figure(figsize=(8, 6))
    
    # Normalizar para mostrar porcentajes
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Crear heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legítima', 'Fraude'],
                yticklabels=['Legítima', 'Fraude'],
                cbar_kws={'label': 'Cantidad'})
    
    plt.title(f'Matriz de Confusión - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Clase Real', fontsize=12)
    plt.xlabel('Clase Predicha', fontsize=12)
    
    # Añadir porcentajes
    for i in range(2):
        for j in range(2):
            plt.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]*100:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save:
        filename = FIGURES_DIR / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {filename}")
    
    plt.show()


def plot_roc_curve(y_test, y_pred_proba, model_name, save=True):
    """
    Grafica la curva ROC
    
    ROC Curve: Muestra el trade-off entre True Positive Rate y False Positive Rate
    - Eje X: False Positive Rate (FPR) = FP / (FP + TN)
    - Eje Y: True Positive Rate (TPR) = TP / (TP + FN) = Recall
    - AUC: Área bajo la curva (1.0 = perfecto, 0.5 = aleatorio)
    
    Propósito: Ver cómo varía el rendimiento según el umbral de decisión
    
    Args:
        y_test: Valores reales
        y_pred_proba: Probabilidades predichas
        model_name: Nombre del modelo
        save: Si True, guarda la figura
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
             label='Línea Base (Aleatorio)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Tasa de Falsas Alarmas)', fontsize=12)
    plt.ylabel('True Positive Rate (Recall)', fontsize=12)
    plt.title(f'Curva ROC - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save:
        filename = FIGURES_DIR / f'roc_curve_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {filename}")
    
    plt.show()


def plot_precision_recall_curve(y_test, y_pred_proba, model_name, save=True):
    """
    Grafica la curva Precision-Recall
    
    Precision-Recall Curve: Especialmente útil para datos desbalanceados
    - Eje X: Recall (¿cuántos fraudes detectamos?)
    - Eje Y: Precision (¿cuántas alertas son correctas?)
    
    Propósito: Entender el trade-off entre precision y recall
    
    Args:
        y_test: Valores reales
        y_pred_proba: Probabilidades predichas
        model_name: Nombre del modelo
        save: Si True, guarda la figura
    """
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='#3498db', lw=2, label=f'{model_name}')
    
    plt.xlabel('Recall (Fraudes Detectados)', fontsize=12)
    plt.ylabel('Precision (Alertas Correctas)', fontsize=12)
    plt.title(f'Curva Precision-Recall - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    
    if save:
        filename = FIGURES_DIR / f'precision_recall_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {filename}")
    
    plt.show()


if __name__ == "__main__":
    from src.data.load import load_data
    from src.data.preprocess import preprocess_pipeline
    
    print("🚀 Iniciando evaluación final en test set...")
    
    # 1. Cargar y preprocesar datos
    data = load_data()
    if data is not None:
        processed_data = preprocess_pipeline(data, apply_balancing=True)
        
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # 2. Evaluar Random Forest (el mejor modelo)
        rf_model = load_model("random_forest")
        y_pred, y_pred_proba, cm = evaluate_on_test(rf_model, X_test, y_test, "Random Forest")
        
        # 3. Visualizaciones
        print("\n🎨 Generando visualizaciones...")
        plot_confusion_matrix(cm, "Random Forest")
        plot_roc_curve(y_test, y_pred_proba, "Random Forest")
        plot_precision_recall_curve(y_test, y_pred_proba, "Random Forest")
        
        print("\n✅ Evaluación completada")