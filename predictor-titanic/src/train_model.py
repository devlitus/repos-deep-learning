import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Importar nuestras funciones de preprocesamiento
import sys
sys.path.append('../src')
from data_preprocessing import preprocess_titanic_data, split_features_target


def train_random_forest_classifier(X_train, y_train, random_state=42):
    """
    Entrena un modelo Random Forest Classifier
    
    Parámetros:
    -----------
    X_train : DataFrame
        Features de entrenamiento
    y_train : Series
        Target de entrenamiento
    random_state : int
        Semilla para reproducibilidad
        
    Retorna:
    --------
    model : RandomForestClassifier
        Modelo entrenado
    """
    
    print("\n🌲 Entrenando Random Forest Classifier...")
    
    # Crear el modelo
    model = RandomForestClassifier(
        n_estimators=100,      # Número de árboles
        max_depth=10,          # Profundidad máxima de cada árbol
        min_samples_split=5,   # Mínimo de muestras para dividir un nodo
        min_samples_leaf=2,    # Mínimo de muestras en una hoja
        random_state=random_state
    )
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    print("✅ Modelo entrenado exitosamente")
    
    return model


def evaluate_classifier(model, X_test, y_test, X_train, y_train):
    """
    Evalúa el modelo de clasificación con múltiples métricas
    
    Parámetros:
    -----------
    model : RandomForestClassifier
        Modelo entrenado
    X_test : DataFrame
        Features de prueba
    y_test : Series
        Target de prueba
    X_train : DataFrame
        Features de entrenamiento (para comparar)
    y_train : Series
        Target de entrenamiento (para comparar)
    """
    
    print("\n" + "=" * 60)
    print("📊 EVALUACIÓN DEL MODELO")
    print("=" * 60)
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # ============================================
    # MÉTRICAS EN TRAIN
    # ============================================
    print("\n🔵 MÉTRICAS EN DATOS DE ENTRENAMIENTO:")
    print("-" * 60)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    prec_train = precision_score(y_train, y_pred_train)
    rec_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    
    print(f"Accuracy (Exactitud):  {acc_train:.4f} ({acc_train*100:.2f}%)")
    print(f"Precision (Precisión): {prec_train:.4f} ({prec_train*100:.2f}%)")
    print(f"Recall (Sensibilidad): {rec_train:.4f} ({rec_train*100:.2f}%)")
    print(f"F1-Score:              {f1_train:.4f}")
    
    # ============================================
    # MÉTRICAS EN TEST
    # ============================================
    print("\n🟢 MÉTRICAS EN DATOS DE PRUEBA:")
    print("-" * 60)
    
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    
    print(f"Accuracy (Exactitud):  {acc_test:.4f} ({acc_test*100:.2f}%)")
    print(f"Precision (Precisión): {prec_test:.4f} ({prec_test*100:.2f}%)")
    print(f"Recall (Sensibilidad): {rec_test:.4f} ({rec_test*100:.2f}%)")
    print(f"F1-Score:              {f1_test:.4f}")
    
    # ============================================
    # INTERPRETACIÓN DE MÉTRICAS
    # ============================================
    print("\n💡 INTERPRETACIÓN:")
    print("-" * 60)
    print(f"De cada 100 predicciones, {acc_test*100:.0f} son correctas")
    print(f"De cada 100 predicciones de 'sobrevivió', {prec_test*100:.0f} son correctas")
    print(f"De cada 100 que sí sobrevivieron, detectamos {rec_test*100:.0f}")
    
    # ============================================
    # MATRIZ DE CONFUSIÓN
    # ============================================
    print("\n📊 MATRIZ DE CONFUSIÓN:")
    print("-" * 60)
    
    cm = confusion_matrix(y_test, y_pred_test)
    print("\n                 Predicción")
    print("               No    Sí")
    print(f"Real  No      {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"      Sí      {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    print("\nSignificado:")
    print(f"  ✅ Verdaderos Negativos: {cm[0,0]} (predijo No y era No)")
    print(f"  ❌ Falsos Positivos: {cm[0,1]} (predijo Sí pero era No)")
    print(f"  ❌ Falsos Negativos: {cm[1,0]} (predijo No pero era Sí)")
    print(f"  ✅ Verdaderos Positivos: {cm[1,1]} (predijo Sí y era Sí)")
    
    # ============================================
    # REPORTE DE CLASIFICACIÓN DETALLADO
    # ============================================
    print("\n📋 REPORTE DETALLADO:")
    print("-" * 60)
    print(classification_report(y_test, y_pred_test, 
                                target_names=['No Sobrevivió', 'Sobrevivió']))
    
    # ============================================
    # IMPORTANCIA DE VARIABLES
    # ============================================
    print("\n🔍 IMPORTANCIA DE VARIABLES:")
    print("-" * 60)
    
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    # Visualizar
    visualize_results(model, X_test, y_test, y_pred_test, feature_importance)
    
    return {
        'accuracy': acc_test,
        'precision': prec_test,
        'recall': rec_test,
        'f1_score': f1_test,
        'confusion_matrix': cm
    }


def visualize_results(model, X_test, y_test, y_pred, feature_importance):
    """
    Visualiza los resultados del modelo
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ============================================
    # 1. MATRIZ DE CONFUSIÓN
    # ============================================
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['No Sobrevivió', 'Sobrevivió'],
                yticklabels=['No Sobrevivió', 'Sobrevivió'])
    axes[0].set_title('Matriz de Confusión', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Valor Real')
    axes[0].set_xlabel('Predicción')
    
    # ============================================
    # 2. IMPORTANCIA DE VARIABLES
    # ============================================
    top_features = feature_importance.head(10)
    axes[1].barh(range(len(top_features)), top_features['importance'])
    axes[1].set_yticks(range(len(top_features)))
    axes[1].set_yticklabels(top_features['feature'])
    axes[1].set_xlabel('Importancia')
    axes[1].set_title('Top 10 Variables Más Importantes', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()


def save_model(model, filename='titanic_model.pkl'):
    """
    Guarda el modelo entrenado
    """
    joblib.dump(model, filename)
    print(f"\n💾 Modelo guardado como: {filename}")


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================
if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("🚢 PROYECTO TITANIC - ENTRENAMIENTO DEL MODELO")
    print("=" * 60)
    
    # ============================================
    # 1. CARGAR Y PREPROCESAR DATOS
    # ============================================
    print("\n📥 PASO 1: Cargando y preprocesando datos...")
    
    titanic = sns.load_dataset('titanic')
    titanic_clean = preprocess_titanic_data(titanic)
    X, y = split_features_target(titanic_clean)
    
    # ============================================
    # 2. DIVIDIR EN TRAIN/TEST
    # ============================================
    print("\n📊 PASO 2: Dividiendo datos en Train/Test...")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 20% para test
        random_state=42,    # Para reproducibilidad
        stratify=y          # Mantener proporciones de clases
    )
    
    print(f"✅ Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✅ Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Verificar balance
    print(f"\nDistribución en Train:")
    print(f"  No sobrevivió: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
    print(f"  Sobrevivió:    {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
    
    # ============================================
    # 3. ENTRENAR MODELO
    # ============================================
    print("\n🌲 PASO 3: Entrenando el modelo...")
    model = train_random_forest_classifier(X_train, y_train)
    
    # ============================================
    # 4. EVALUAR MODELO
    # ============================================
    print("\n📊 PASO 4: Evaluando el modelo...")
    metrics = evaluate_classifier(model, X_test, y_test, X_train, y_train)
    
    # ============================================
    # 5. GUARDAR MODELO
    # ============================================
    save_model(model, 'models/titanic_random_forest.pkl')
    
    print("\n" + "=" * 60)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print(f"\n🎯 Accuracy Final: {metrics['accuracy']*100:.2f}%")