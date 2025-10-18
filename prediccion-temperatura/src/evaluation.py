"""
═══════════════════════════════════════════════════════════════
MÓDULO: EVALUACIÓN DEL MODELO
═══════════════════════════════════════════════════════════════

Propósito: Calcular métricas para medir el rendimiento del modelo

Funciones:
1. calculate_metrics() → Calcula todas las métricas
2. print_metrics() → Muestra métricas formateadas
3. evaluate_model() → Evalúa el modelo completo
4. create_evaluation_report() → Genera reporte en archivo
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def calculate_metrics(y_true, y_pred):
    """
    Calcula todas las métricas de evaluación
    
    ═══════════════════════════════════════════════════════════
    MÉTRICAS CALCULADAS
    ═══════════════════════════════════════════════════════════
    
    1. MSE (Mean Squared Error):
       - Promedio de errores al cuadrado
       - Penaliza más los errores grandes
       - Unidades: °C²
       - Fórmula: promedio((y_true - y_pred)²)
    
    2. RMSE (Root Mean Squared Error):
       - Raíz cuadrada del MSE
       - Más fácil de interpretar (unidades: °C)
       - Fórmula: √MSE
       - Ejemplo: RMSE=2°C significa "me equivoco 2°C en promedio"
    
    3. MAE (Mean Absolute Error):
       - Promedio de errores absolutos
       - Todos los errores pesan igual
       - Unidades: °C
       - Fórmula: promedio(|y_true - y_pred|)
       - Menos sensible a outliers que RMSE
    
    4. MAPE (Mean Absolute Percentage Error):
       - Error en porcentaje
       - Fácil de interpretar
       - Sin unidades (%)
       - Fórmula: promedio(|y_true - y_pred| / |y_true|) × 100
       - Ejemplo: MAPE=5% significa "5% de error promedio"
    
    5. R² (R-squared / Coeficiente de Determinación):
       - Qué tan bien el modelo explica la variabilidad
       - Rango: -∞ a 1 (1 es perfecto)
       - Sin unidades
       - R²=0.85 significa "explica el 85% de la variación"
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        y_true: Valores reales (temperaturas reales)
        y_pred: Valores predichos (temperaturas predichas)
    
    Returns:
        dict: Diccionario con todas las métricas
              {'MSE': ..., 'RMSE': ..., 'MAE': ..., 'MAPE': ..., 'R2': ...}
    """
    
    # ═══════════════════════════════════════════════════════════
    # PREPARACIÓN DE DATOS
    # ═══════════════════════════════════════════════════════════
    
    # Aplanar arrays si tienen más de 1 dimensión
    # Ejemplo: [[20], [21], [19]] → [20, 21, 19]
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    """
    ¿Por qué flatten()?
    
    A veces los datos vienen en formato (n, 1):
    [[20],
     [21],
     [19]]
    
    Necesitamos formato (n,):
    [20, 21, 19]
    
    flatten() convierte cualquier dimensión a 1D
    """
    
    # ═══════════════════════════════════════════════════════════
    # CALCULAR MÉTRICAS
    # ═══════════════════════════════════════════════════════════
    
    # ───────────────────────────────────────────────────────────
    # 1. MSE (Mean Squared Error)
    # ───────────────────────────────────────────────────────────
    
    mse = mean_squared_error(y_true, y_pred)
    
    """
    ¿Qué hace mean_squared_error()?
    
    Paso a paso:
    1. Calcula errores: (y_true - y_pred)
    2. Eleva al cuadrado: (y_true - y_pred)²
    3. Promedia: suma(errores²) / n
    
    Ejemplo:
    y_true = [20, 21, 19]
    y_pred = [21, 22, 20]
    errores = [-1, -1, -1]
    errores² = [1, 1, 1]
    MSE = (1 + 1 + 1) / 3 = 1.0
    
    Interpretación:
    - MSE = 0 → Perfecto
    - MSE = 1 → Errores pequeños
    - MSE = 100 → Errores grandes
    """
    
    # ───────────────────────────────────────────────────────────
    # 2. RMSE (Root Mean Squared Error)
    # ───────────────────────────────────────────────────────────
    
    rmse = np.sqrt(mse)
    
    """
    ¿Por qué calcular RMSE?
    
    MSE tiene unidades raras (°C²)
    RMSE tiene unidades normales (°C)
    
    Ejemplo:
    MSE = 4.0 (difícil de interpretar)
    RMSE = √4.0 = 2.0°C (fácil: "me equivoco 2 grados")
    
    RMSE es LA métrica más usada para regresión
    porque es fácil de entender.
    """
    
    # ───────────────────────────────────────────────────────────
    # 3. MAE (Mean Absolute Error)
    # ───────────────────────────────────────────────────────────
    
    mae = mean_absolute_error(y_true, y_pred)
    
    """
    ¿Qué hace mean_absolute_error()?
    
    Paso a paso:
    1. Calcula errores: (y_true - y_pred)
    2. Toma valor absoluto: |errores|
    3. Promedia: suma(|errores|) / n
    
    Ejemplo:
    y_true = [20, 21, 28]
    y_pred = [21, 22, 20]
    errores = [-1, -1, 8]
    |errores| = [1, 1, 8]
    MAE = (1 + 1 + 8) / 3 = 3.33°C
    
    Diferencia con RMSE:
    MAE = 3.33 (todos los errores pesan igual)
    RMSE = 4.69 (penaliza más el error de 8)
    
    ¿Cuál usar?
    - MAE: Si todos los errores son igual de malos
    - RMSE: Si los errores grandes son peores
    """
    
    # ───────────────────────────────────────────────────────────
    # 4. MAPE (Mean Absolute Percentage Error)
    # ───────────────────────────────────────────────────────────
    
    # Calcular manualmente (scikit-learn no tiene MAPE built-in)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    """
    ¿Cómo se calcula MAPE?
    
    Paso a paso:
    1. Calcula errores: (y_true - y_pred)
    2. Toma valor absoluto: |errores|
    3. Divide por valor real: |errores| / y_true
    4. Promedia: promedio(...)
    5. Multiplica por 100 para obtener %
    
    Ejemplo:
    y_true = [20, 21]
    y_pred = [21, 22]
    
    errores = [-1, -1]
    |errores| = [1, 1]
    % errores = [1/20, 1/21] = [0.05, 0.0476]
    promedio = 0.0488
    MAPE = 0.0488 × 100 = 4.88%
    
    Interpretación:
    MAPE = 4.88% → "Me equivoco un 4.88% en promedio"
    
    Ventaja:
    - Fácil de entender
    - Permite comparar modelos en diferentes escalas
    
    Limitación:
    - No funciona bien si y_true tiene valores cercanos a 0
      (división por casi cero)
    """
    
    # ───────────────────────────────────────────────────────────
    # 5. R² (R-squared)
    # ───────────────────────────────────────────────────────────
    
    r2 = r2_score(y_true, y_pred)
    
    """
    ¿Qué mide R²?
    
    Compara tu modelo contra un modelo "tonto" que siempre
    predice el promedio.
    
    Fórmula conceptual:
    R² = 1 - (errores_tu_modelo / errores_modelo_promedio)
    
    Ejemplo:
    
    Datos reales: [10, 20, 30]
    Promedio: 20
    
    Modelo "tonto" (predice siempre 20):
    Predicciones: [20, 20, 20]
    Errores: [10, 0, 10]
    Suma errores²: 100 + 0 + 100 = 200
    
    Tu modelo LSTM:
    Predicciones: [11, 20, 29]
    Errores: [1, 0, 1]
    Suma errores²: 1 + 0 + 1 = 2
    
    R² = 1 - (2 / 200) = 1 - 0.01 = 0.99 = 99%
    
    Interpretación:
    "Tu modelo explica el 99% de la variabilidad de los datos"
    "Es un 99% mejor que predecir siempre el promedio"
    
    Valores posibles:
    - R² = 1.0 → Perfecto (0% error)
    - R² = 0.9 → Excelente (explica 90%)
    - R² = 0.5 → Regular (explica 50%)
    - R² = 0.0 → Malo (igual que predecir promedio)
    - R² < 0.0 → Peor que predecir el promedio
    """
    
    # ═══════════════════════════════════════════════════════════
    # CREAR DICCIONARIO CON RESULTADOS
    # ═══════════════════════════════════════════════════════════
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }
    
    return metrics


def print_metrics(metrics, dataset_name='Test'):
    """
    Imprime las métricas de forma bonita y legible
    
    ═══════════════════════════════════════════════════════════
    FORMATO DE SALIDA
    ═══════════════════════════════════════════════════════════
    
    Muestra:
    1. Valores numéricos de cada métrica
    2. Interpretación cualitativa (Excelente/Bueno/Malo)
    3. Consejos sobre el rendimiento
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        metrics: Diccionario con las métricas
        dataset_name: Nombre del conjunto ('Test', 'Train', 'Validation')
    """
    
    print(f"\n{'='*70}")
    print(f"📊 MÉTRICAS DE EVALUACIÓN - {dataset_name}")
    print(f"{'='*70}")
    
    # ───────────────────────────────────────────────────────────
    # SECCIÓN 1: Métricas de Error
    # ───────────────────────────────────────────────────────────
    
    print(f"\n🎯 Errores:")
    print(f"   MSE  (Mean Squared Error)        : {metrics['MSE']:.4f}")
    print(f"   RMSE (Root Mean Squared Error)   : {metrics['RMSE']:.4f} °C")
    print(f"   MAE  (Mean Absolute Error)       : {metrics['MAE']:.4f} °C")
    print(f"   MAPE (Mean Abs Percentage Error) : {metrics['MAPE']:.2f} %")
    
    """
    Formato de impresión:
    
    {metrics['RMSE']:.4f}
         ↑            ↑
         │            └─ 4 decimales
         └────────────── Valor de la métrica
    
    Ejemplo:
    Si RMSE = 2.123456
    Se imprime: 2.1235
    """
    
    # ───────────────────────────────────────────────────────────
    # SECCIÓN 2: Bondad de Ajuste
    # ───────────────────────────────────────────────────────────
    
    print(f"\n📈 Bondad de ajuste:")
    print(f"   R² (R-squared)                   : {metrics['R2']:.4f}")
    print(f"   Varianza explicada               : {metrics['R2']*100:.2f} %")
    
    """
    ¿Qué es "Varianza explicada"?
    
    Es R² expresado en porcentaje.
    
    Ejemplo:
    R² = 0.8542
    Varianza explicada = 85.42%
    
    Significa: "El modelo explica el 85.42% de las
                variaciones en la temperatura"
    """
    
    # ───────────────────────────────────────────────────────────
    # SECCIÓN 3: Interpretación Cualitativa
    # ───────────────────────────────────────────────────────────
    
    print(f"\n💡 Interpretación:")
    
    # Evaluar R²
    if metrics['R2'] >= 0.9:
        print("   ✅ Excelente: El modelo explica muy bien los datos")
    elif metrics['R2'] >= 0.7:
        print("   👍 Bueno: El modelo tiene buen desempeño")
    elif metrics['R2'] >= 0.5:
        print("   ⚠️  Aceptable: El modelo puede mejorar")
    else:
        print("   ❌ Malo: El modelo necesita mejoras significativas")
    
    """
    Escala de calidad basada en R²:
    
    R² ≥ 0.9  → Excelente (explica >90% de variación)
    R² ≥ 0.7  → Bueno (explica 70-90%)
    R² ≥ 0.5  → Aceptable (explica 50-70%)
    R² < 0.5  → Malo (explica <50%)
    
    Estas son guías generales.
    Para temperaturas, esperamos R² ≥ 0.8
    """
    
    # Evaluar MAPE
    if metrics['MAPE'] <= 5:
        print("   ✅ Error porcentual muy bajo (<5%)")
    elif metrics['MAPE'] <= 10:
        print("   👍 Error porcentual aceptable (5-10%)")
    else:
        print("   ⚠️  Error porcentual alto (>10%)")
    
    """
    Escala de calidad basada en MAPE:
    
    MAPE ≤ 5%   → Excelente
    MAPE ≤ 10%  → Bueno
    MAPE ≤ 20%  → Aceptable
    MAPE > 20%  → Malo
    
    Para predicción de temperatura:
    MAPE < 5% es un rendimiento excelente
    """
    
    # Evaluar RMSE (contexto: temperaturas en Melbourne 0-26°C)
    if metrics['RMSE'] <= 1.0:
        print("   ✅ RMSE excelente: errores <1°C")
    elif metrics['RMSE'] <= 2.0:
        print("   👍 RMSE bueno: errores 1-2°C")
    elif metrics['RMSE'] <= 3.0:
        print("   ⚠️  RMSE aceptable: errores 2-3°C")
    else:
        print("   ⚠️  RMSE alto: errores >3°C - considerar mejorar el modelo")
    
    """
    Contexto de RMSE para temperaturas:
    
    Rango de datos: 0-26°C (26°C de variación)
    
    RMSE ≤ 1°C  → Excelente (error <4% del rango)
    RMSE ≤ 2°C  → Bueno (error <8% del rango)
    RMSE ≤ 3°C  → Aceptable (error <12% del rango)
    RMSE > 3°C  → Malo (error >12% del rango)
    
    Estos umbrales dependen del contexto:
    - Predicción a corto plazo: RMSE < 1°C
    - Predicción a largo plazo: RMSE < 3°C aceptable
    """
    
    print(f"{'='*70}\n")


def evaluate_model(model, X_test, y_test, scaler):
    """
    Evalúa el modelo completo en datos de prueba
    
    ═══════════════════════════════════════════════════════════
    PROCESO DE EVALUACIÓN
    ═══════════════════════════════════════════════════════════
    
    1. Hacer predicciones con datos de test
    2. Desnormalizar predicciones y valores reales
    3. Calcular métricas
    4. Mostrar resultados
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        model: Modelo LSTM entrenado
        X_test: Datos de entrada de prueba (normalizados)
        y_test: Valores reales de prueba (normalizados)
        scaler: Scaler para desnormalizar
    
    Returns:
        tuple: (predicciones, métricas)
               - predicciones: Array con temperaturas predichas (°C)
               - métricas: Diccionario con todas las métricas
    """
    
    print("🔍 Evaluando modelo en datos de prueba...\n")
    
    # ═══════════════════════════════════════════════════════════
    # PASO 1: Hacer Predicciones
    # ═══════════════════════════════════════════════════════════
    
    predictions_scaled = model.predict(X_test, verbose=0)
    
    """
    ¿Qué hace model.predict()?
    
    Toma los datos de entrada (X_test) y genera predicciones
    
    X_test shape: (n_samples, 60, 1)
    predictions_scaled shape: (n_samples, 1)
    
    Ejemplo:
    X_test tiene 718 secuencias de 60 días
    predictions_scaled tendrá 718 predicciones
    
    verbose=0 → No mostrar barra de progreso
    """
    
    # ═══════════════════════════════════════════════════════════
    # PASO 2: Desnormalizar
    # ═══════════════════════════════════════════════════════════
    
    # Desnormalizar predicciones (de 0-1 a °C)
    predictions = scaler.inverse_transform(predictions_scaled)
    
    # Desnormalizar valores reales (de 0-1 a °C)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    """
    ¿Por qué desnormalizar?
    
    Durante el entrenamiento trabajamos con valores 0-1
    Para evaluar necesitamos volver a escala original (°C)
    
    Ejemplo:
    Predicción normalizada: 0.7
    Rango original: 0-26°C
    Predicción desnormalizada: 0.7 × 26 = 18.2°C
    
    scaler.inverse_transform() revierte la normalización:
    valor_original = valor_normalizado × (max - min) + min
    
    reshape(-1, 1) → Convierte (n,) a (n, 1)
    Porque scaler espera formato (n, 1)
    """
    
    # ═══════════════════════════════════════════════════════════
    # PASO 3: Calcular Métricas
    # ═══════════════════════════════════════════════════════════
    
    metrics = calculate_metrics(y_test_original, predictions)
    
    # ═══════════════════════════════════════════════════════════
    # PASO 4: Mostrar Resultados
    # ═══════════════════════════════════════════════════════════
    
    print_metrics(metrics)
    
    return predictions, metrics


def create_evaluation_report(metrics, save_path='reports/metricas.txt'):
    """
    Crea un reporte de texto con las métricas
    
    ¿Para qué?
    - Guardar resultados para comparar después
    - Documentar el rendimiento del modelo
    - Compartir resultados con otros
    
    Args:
        metrics: Diccionario con las métricas
        save_path: Ruta donde guardar el reporte
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("REPORTE DE EVALUACIÓN - PREDICCIÓN DE TEMPERATURA\n")
        f.write("="*60 + "\n\n")
        
        f.write("MÉTRICAS DE ERROR:\n")
        f.write("-"*60 + "\n")
        f.write(f"MSE  (Mean Squared Error)        : {metrics['MSE']:.4f}\n")
        f.write(f"RMSE (Root Mean Squared Error)   : {metrics['RMSE']:.4f} °C\n")
        f.write(f"MAE  (Mean Absolute Error)       : {metrics['MAE']:.4f} °C\n")
        f.write(f"MAPE (Mean Abs Percentage Error) : {metrics['MAPE']:.2f} %\n\n")
        
        f.write("BONDAD DE AJUSTE:\n")
        f.write("-"*60 + "\n")
        f.write(f"R² (R-squared)                   : {metrics['R2']:.4f}\n")
        f.write(f"Varianza explicada               : {metrics['R2']*100:.2f} %\n\n")
        
        f.write("INTERPRETACIÓN:\n")
        f.write("-"*60 + "\n")
        
        # Interpretación de R²
        if metrics['R2'] >= 0.9:
            f.write("✅ Excelente: El modelo explica muy bien los datos\n")
        elif metrics['R2'] >= 0.7:
            f.write("👍 Bueno: El modelo tiene buen desempeño\n")
        elif metrics['R2'] >= 0.5:
            f.write("⚠️  Aceptable: El modelo puede mejorar\n")
        else:
            f.write("❌ Malo: El modelo necesita mejoras significativas\n")
        
        # Interpretación de MAPE
        if metrics['MAPE'] <= 5:
            f.write("✅ Error porcentual muy bajo (<5%)\n")
        elif metrics['MAPE'] <= 10:
            f.write("👍 Error porcentual aceptable (5-10%)\n")
        else:
            f.write("⚠️  Error porcentual alto (>10%)\n")
        
        # Interpretación de RMSE
        if metrics['RMSE'] <= 1.0:
            f.write("✅ RMSE excelente: errores <1°C\n")
        elif metrics['RMSE'] <= 2.0:
            f.write("👍 RMSE bueno: errores 1-2°C\n")
        elif metrics['RMSE'] <= 3.0:
            f.write("⚠️  RMSE aceptable: errores 2-3°C\n")
        else:
            f.write("⚠️  RMSE alto: errores >3°C\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"✅ Reporte guardado en {save_path}")


# ═══════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 PROBANDO MÓDULO DE EVALUACIÓN")
    print("="*70 + "\n")
    
    # Crear datos de ejemplo
    np.random.seed(42)  # Para reproducibilidad
    
    # Simular predicciones y valores reales
    y_true = np.array([20.5, 21.0, 19.5, 22.0, 23.5, 24.0, 22.5, 21.5, 20.0, 19.0])
    y_pred = np.array([20.8, 21.2, 19.3, 22.3, 23.2, 24.5, 22.7, 21.3, 20.2, 18.8])
    
    print("📊 Datos de ejemplo:")
    print(f"   Valores reales     : {y_true[:5]}... (10 valores)")
    print(f"   Valores predichos  : {y_pred[:5]}... (10 valores)\n")
    
    # Calcular métricas
    print("📈 Calculando métricas...\n")
    metrics = calculate_metrics(y_true, y_pred)
    
    # Mostrar métricas
    print_metrics(metrics, dataset_name='Ejemplo')
    
    print("\n💡 Nota: Para evaluar el modelo real, usa el script train.py")
    print("="*70)