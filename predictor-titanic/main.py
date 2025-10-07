"""
Pipeline completo del proyecto Titanic
Ejecuta todo el flujo: carga → preprocesamiento → entrenamiento → evaluación
"""

import sys
from pathlib import Path

# Añadir src al path para imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data_loader import load_data, explore_data, prepare_data
from src.data_preprocessing import preprocess_titanic_data, split_features_target
from src.model import train_random_forest_classifier, evaluate_classifier, save_model
from sklearn.model_selection import train_test_split
import config


def main():
    """
    Pipeline completo del proyecto Titanic
    Sigue el orden estándar del proyecto
    """
    
    print("\n" + "=" * 60)
    print("🚢 PROYECTO TITANIC - PIPELINE COMPLETO")
    print("=" * 60)
    
    # ============================================
    # 1. CARGAR DATOS
    # ============================================
    print("\n📥 PASO 1: Cargando datos...")
    df = load_data()
    
    # ============================================
    # 2. EXPLORAR DATOS
    # ============================================
    print("\n🔍 PASO 2: Explorando datos...")
    df = explore_data(df)
    
    # ============================================
    # 3. PREPROCESAR DATOS
    # ============================================
    print("\n🧹 PASO 3: Preprocesando datos...")
    df_clean = preprocess_titanic_data(df)
    
    # ============================================
    # 4. PREPARAR FEATURES Y TARGET
    # ============================================
    print("\n📊 PASO 4: Separando features y target...")
    X, y = split_features_target(df_clean)
    
    # ============================================
    # 5. DIVIDIR EN TRAIN/TEST
    # ============================================
    print("\n✂️ PASO 5: Dividiendo en train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"✅ Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✅ Test:  {X_test.shape[0]} muestras ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # ============================================
    # 6. ENTRENAR MODELO
    # ============================================
    print("\n🌲 PASO 6: Entrenando modelo...")
    modelo = train_random_forest_classifier(X_train, y_train, config.RANDOM_STATE)
    
    # ============================================
    # 7. EVALUAR MODELO
    # ============================================
    print("\n📊 PASO 7: Evaluando modelo...")
    metrics = evaluate_classifier(modelo, X_test, y_test, X_train, y_train)
    
    # ============================================
    # 8. GUARDAR MODELO
    # ============================================
    print("\n💾 PASO 8: Guardando modelo...")
    save_model(modelo, config.MODEL_FILE)
    
    # ============================================
    # RESUMEN FINAL
    # ============================================
    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\n🎯 Accuracy Final: {metrics['accuracy']*100:.2f}%")
    
    # Mostrar solo nombre del archivo para mejor UX
    import os
    model_name = os.path.basename(config.MODEL_FILE)
    print(f"💾 Modelo: {model_name}")
    
    print("\n💡 Para hacer predicciones, usa: src/predictor.py")
    print("💡 Para visualizaciones, usa: src/visualization.py")
    print("💡 Para app interactiva, usa: streamlit run src/app.py")
    

if __name__ == "__main__":
    main()
