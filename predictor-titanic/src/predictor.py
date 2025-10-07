"""
Módulo para hacer predicciones con el modelo entrenado del Titanic
Permite predecir la supervivencia de nuevos pasajeros
"""

import pandas as pd
import joblib
import os
import sys
from pathlib import Path

# Añadir el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import config


def load_trained_model(model_path=None):
    """
    Carga el modelo entrenado desde archivo
    
    Parámetros:
    -----------
    model_path : str, opcional
        Ruta al archivo del modelo. Si no se proporciona, usa config.MODEL_FILE
        
    Retorna:
    --------
    model : RandomForestClassifier
        Modelo entrenado cargado
    """
    if model_path is None:
        model_path = config.MODEL_FILE
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo: {os.path.basename(model_path)}\n"
            "Por favor, entrena el modelo primero ejecutando: python main.py"
        )
    
    model = joblib.load(model_path)
    # Mostrar solo el nombre del archivo para mejor UX
    display_name = os.path.basename(model_path)
    print(f"✅ Modelo cargado: {display_name}")
    return model


def predict_survival(model, passenger_data):
    """
    Predice la supervivencia de uno o más pasajeros
    
    Parámetros:
    -----------
    model : RandomForestClassifier
        Modelo entrenado
    passenger_data : dict o DataFrame
        Datos del pasajero(s) con las siguientes columnas:
        - pclass: int (1, 2, o 3)
        - sex: int (0=male, 1=female)
        - age: float
        - sibsp: int
        - parch: int
        - fare: float
        - embarked: int (0=C, 1=Q, 2=S)
        - family_size: int
        - is_alone: int (0 o 1)
        
    Retorna:
    --------
    predictions : array
        Array con predicciones (0=No sobrevivió, 1=Sobrevivió)
    probabilities : array
        Array con probabilidades de supervivencia
    """
    
    # Convertir dict a DataFrame si es necesario
    if isinstance(passenger_data, dict):
        passenger_data = pd.DataFrame([passenger_data])
    
    # Verificar que tenga las columnas correctas
    required_cols = config.FEATURES
    missing_cols = set(required_cols) - set(passenger_data.columns)
    
    if missing_cols:
        raise ValueError(f"Faltan las siguientes columnas: {missing_cols}")
    
    # Asegurar el orden correcto de columnas
    passenger_data = passenger_data[required_cols]
    
    # Hacer predicciones
    predictions = model.predict(passenger_data)
    probabilities = model.predict_proba(passenger_data)[:, 1]  # Probabilidad de sobrevivir
    
    return predictions, probabilities


def predict_new_passengers(model=None):
    """
    Predice la supervivencia de pasajeros de ejemplo
    Función de demostración
    
    Parámetros:
    -----------
    model : RandomForestClassifier, opcional
        Modelo entrenado. Si no se proporciona, se carga desde archivo
    """
    
    # Cargar modelo si no se proporcionó
    if model is None:
        model = load_trained_model()
    
    print("\n" + "=" * 60)
    print("🔮 PREDICCIONES DE SUPERVIVENCIA - EJEMPLOS")
    print("=" * 60)
    
    # Definir pasajeros de ejemplo
    ejemplos = [
        {
            'nombre': 'Rose (1ra clase, mujer, joven)',
            'pclass': 1,
            'sex': 1,  # female
            'age': 17,
            'sibsp': 0,
            'parch': 2,
            'fare': 100.0,
            'embarked': 2,  # Southampton
            'family_size': 3,
            'is_alone': 0
        },
        {
            'nombre': 'Jack (3ra clase, hombre, joven)',
            'pclass': 3,
            'sex': 0,  # male
            'age': 20,
            'sibsp': 0,
            'parch': 0,
            'fare': 8.0,
            'embarked': 2,  # Southampton
            'family_size': 1,
            'is_alone': 1
        },
        {
            'nombre': 'Mujer mayor, 2da clase',
            'pclass': 2,
            'sex': 1,  # female
            'age': 60,
            'sibsp': 1,
            'parch': 0,
            'fare': 25.0,
            'embarked': 0,  # Cherbourg
            'family_size': 2,
            'is_alone': 0
        },
        {
            'nombre': 'Niño, 3ra clase',
            'pclass': 3,
            'sex': 0,  # male
            'age': 5,
            'sibsp': 1,
            'parch': 2,
            'fare': 15.0,
            'embarked': 2,  # Southampton
            'family_size': 4,
            'is_alone': 0
        }
    ]
    
    # Hacer predicciones para cada ejemplo
    for ejemplo in ejemplos:
        nombre = ejemplo.pop('nombre')
        
        # Predecir
        pred, prob = predict_survival(model, ejemplo)
        
        # Mostrar resultado
        resultado = "✅ SOBREVIVIÓ" if pred[0] == 1 else "❌ NO SOBREVIVIÓ"
        print(f"\n👤 {nombre}")
        print(f"   Predicción: {resultado}")
        print(f"   Probabilidad de supervivencia: {prob[0]*100:.1f}%")
        print(f"   Confianza: {'Alta' if abs(prob[0] - 0.5) > 0.3 else 'Media' if abs(prob[0] - 0.5) > 0.15 else 'Baja'}")
    
    print("\n" + "=" * 60)


# ============================================
# FUNCIÓN PRINCIPAL PARA TESTEAR
# ============================================
if __name__ == "__main__":
    
    print("\n🧪 EJECUTANDO PREDICTOR EN MODO TEST\n")
    
    try:
        # Cargar modelo
        modelo = load_trained_model()
        
        # Hacer predicciones de ejemplo
        predict_new_passengers(modelo)
        
        print("\n✅ Predictor funcionando correctamente")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 SOLUCIÓN: Ejecuta primero el entrenamiento:")
        print("   cd predictor-titanic")
        print("   python main.py")
