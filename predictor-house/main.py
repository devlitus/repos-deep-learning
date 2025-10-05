# main.py
from src.data_loader import load_data, explore_data, prepare_data
from src.model import split_data, train_model, evaluate_model, save_model
from src.predictor import predict_new_houses
from src.visualizations import plot_feature_vs_target, plot_predictions_vs_actual

def main():
    """Pipeline completo de entrenamiento"""
    
    # 1. Cargar datos
    df = load_data()
    
    # 2. Explorar datos
    df = explore_data(df)
    
    # 3. Preparar datos
    X, y = prepare_data(df)
    
    # 4. Visualizar relaciones
    plot_feature_vs_target(df)
    
    # 5. Dividir datos
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 6. Entrenar modelo
    modelo = train_model(X_train, y_train)
    
    # 7. Evaluar modelo
    y_pred, metricas = evaluate_model(modelo, X_test, y_test)
    
    # 8. Visualizar predicciones
    plot_predictions_vs_actual(y_test, y_pred)
    
    # 9. Guardar modelo
    save_model(modelo)
    
    # 10. Probar con casas nuevas
    casas_nuevas = [
        [130, 3, 2, 10, 5.0],
        [80, 2, 1, 20, 15.0],
        [200, 5, 4, 1, 2.0]
    ]
    predict_new_houses(modelo, casas_nuevas)
    
    print("\n🎉 Pipeline completado exitosamente")

if __name__ == "__main__":
    main()