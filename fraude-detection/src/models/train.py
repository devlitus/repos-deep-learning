"""
Módulo para entrenamiento de modelos
"""
import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import joblib
from config.config import MODELS_DIR, RANDOM_STATE


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo de Regresión Logística
    
    Logistic Regression: Modelo lineal que predice probabilidades
    - Simple y rápido
    - Buena línea base
    - Funciona bien cuando las clases son linealmente separables
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        
    Returns:
        tuple: (modelo entrenado, métricas en validación)
    """
    print("\n" + "="*60)
    print("🔵 ENTRENANDO LOGISTIC REGRESSION")
    print("="*60)
    
    # Crear y entrenar modelo
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight='balanced'  # Dar más peso a la clase minoritaria
    )
    
    print("⏳ Entrenando modelo...")
    model.fit(X_train, y_train)
    print("✅ Modelo entrenado")
    
    # Evaluar en validación
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calcular métricas
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Logistic Regression")
    
    return model, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Entrena un modelo de Random Forest
    
    Random Forest: Ensemble de muchos árboles de decisión
    - Más robusto que un solo árbol
    - Maneja bien interacciones no lineales
    - Resistente al overfitting
    
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        X_val: Features de validación
        y_val: Target de validación
        
    Returns:
        tuple: (modelo entrenado, métricas en validación)
    """
    print("\n" + "="*60)
    print("🌲 ENTRENANDO RANDOM FOREST")
    print("="*60)
    
    # Crear y entrenar modelo
    model = RandomForestClassifier(
        n_estimators=100,  # Número de árboles
        max_depth=10,  # Profundidad máxima de cada árbol
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1  # Usar todos los cores del CPU
    )
    
    print("⏳ Entrenando modelo (esto puede tardar 1-2 minutos)...")
    model.fit(X_train, y_train)
    print("✅ Modelo entrenado")
    
    # Evaluar en validación
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Calcular métricas
    metrics = calculate_metrics(y_val, y_pred, y_pred_proba, "Random Forest")
    
    return model, metrics


def calculate_metrics(y_true, y_pred, y_pred_proba, model_name):
    """
    Calcula métricas de evaluación para clasificación
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones del modelo
        y_pred_proba: Probabilidades predichas
        model_name: Nombre del modelo
        
    Returns:
        dict: Diccionario con todas las métricas
    """
    print(f"\n📊 MÉTRICAS DE EVALUACIÓN - {model_name}")
    print("="*60)
    
    # Calcular métricas
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"\n🎯 Métricas Principales:")
    print(f"   • Precision: {precision:.4f} - De los que predice como fraude, {precision*100:.2f}% son correctos")
    print(f"   • Recall:    {recall:.4f} - Detecta {recall*100:.2f}% de todos los fraudes")
    print(f"   • F1-Score:  {f1:.4f} - Balance entre Precision y Recall")
    print(f"   • ROC-AUC:   {roc_auc:.4f} - Capacidad de distinguir entre clases")
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📋 Matriz de Confusión:")
    print(f"                 Predicho")
    print(f"               Legítima  Fraude")
    print(f"Real Legítima    {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"     Fraude      {cm[1,0]:6d}  {cm[1,1]:6d}")
    
    print(f"\n🔍 Interpretación:")
    print(f"   • Verdaderos Negativos (TN): {cm[0,0]:,} - Legítimas correctamente identificadas")
    print(f"   • Falsos Positivos (FP):     {cm[0,1]:,} - Legítimas marcadas como fraude")
    print(f"   • Falsos Negativos (FN):     {cm[1,0]:,} - Fraudes NO detectados ⚠️")
    print(f"   • Verdaderos Positivos (TP): {cm[1,1]:,} - Fraudes correctamente detectados ✅")
    
    metrics = {
        'model_name': model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    return metrics


def save_model(model, model_name):
    """
    Guarda el modelo entrenado
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
    """
    model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Modelo guardado en: {model_path}")


def compare_models(metrics_list):
    """
    Compara los resultados de múltiples modelos
    
    Args:
        metrics_list: Lista de diccionarios con métricas de cada modelo
    """
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*60)
    
    # Crear DataFrame para comparar
    comparison = pd.DataFrame([
        {
            'Modelo': m['model_name'],
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}"
        }
        for m in metrics_list
    ])
    
    print("\n", comparison.to_string(index=False))
    
    # Determinar el mejor modelo (por F1-Score)
    best_model_idx = np.argmax([m['f1_score'] for m in metrics_list])
    best_model = metrics_list[best_model_idx]['model_name']
    
    print(f"\n🏆 MEJOR MODELO: {best_model}")
    print(f"   (Basado en F1-Score)")


if __name__ == "__main__":
    # Prueba del módulo
    from src.data.load import load_data
    from src.data.preprocess import preprocess_pipeline
    
    print("🚀 Iniciando entrenamiento de modelos...")
    
    # 1. Cargar y preprocesar datos
    data = load_data()
    if data is not None:
        processed_data = preprocess_pipeline(data, apply_balancing=True)
        
        # Extraer conjuntos de datos
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
        X_val = processed_data['X_val']
        y_val = processed_data['y_val']
        
        # 2. Entrenar modelos
        all_metrics = []
        all_models = []
        
        # Logistic Regression
        lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_val, y_val)
        save_model(lr_model, "logistic_regression")
        all_metrics.append(lr_metrics)
        all_models.append(('Logistic Regression', lr_model))
        
        # Random Forest
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
        save_model(rf_model, "random_forest")
        all_metrics.append(rf_metrics)
        all_models.append(('Random Forest', rf_model))
        
        # 3. Comparar modelos
        compare_models(all_metrics)