"""
Classical ML Models - Análisis de Sentimientos
===============================================

Este módulo contiene modelos clásicos de Machine Learning para clasificación de texto.

MODELOS INCLUIDOS:
------------------

1. NAIVE BAYES (Bayes Ingenuo):
   - Modelo probabilístico
   - Rápido y eficiente
   - Ideal para clasificación de texto
   - Funciona excepcionalmente bien con TF-IDF

2. SUPPORT VECTOR MACHINE (SVM):
   - Encuentra hiperplano óptimo que separa clases
   - Robusto y preciso
   - Funciona muy bien con datos de alta dimensión
   - Excelente con TF-IDF features

3. LOGISTIC REGRESSION:
   - Modelo lineal para clasificación
   - Interpretable (puedes ver qué palabras influyen más)
   - Rápido de entrenar
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
from typing import Tuple, Dict
import config

# ============================================================================
# 1. NAIVE BAYES (Teorema de Bayes aplicado a texto)
# ============================================================================

def train_naive_bayes(X_train, y_train, alpha: float = None):
    """
    Entrena un clasificador Naive Bayes.

    📊 NAIVE BAYES:
    ===============

    TEOREMA DE BAYES:
    P(clase|documento) = P(documento|clase) × P(clase) / P(documento)

    Para clasificación de texto:
    P(positivo|"excellent movie") = ?

    CÓMO FUNCIONA:
    --------------
    1. Calcula probabilidad de cada palabra dada una clase:
       P("excellent"|positivo) = 0.05 (alta)
       P("excellent"|negativo) = 0.001 (baja)

    2. Calcula probabilidad del documento:
       P("excellent movie"|positivo) = P("excellent"|pos) × P("movie"|pos)

    3. Aplica Bayes para obtener P(positivo|documento)

    ASUNCIÓN "NAIVE" (Ingenua):
    ---------------------------
    Las palabras son INDEPENDIENTES entre sí.
    P("excellent movie") = P("excellent") × P("movie")

    ⚠️ Esto NO es verdad en realidad:
    "not good" tiene significado diferente a ["not", "good"] por separado.

    Pero funciona sorprendentemente bien en la práctica!

    POR QUÉ FUNCIONA BIEN CON TEXTO:
    ---------------------------------
    ✅ Rápido de entrenar (solo cuenta frecuencias)
    ✅ Requiere pocos datos
    ✅ Funciona bien con alta dimensionalidad
    ✅ Robusto a features irrelevantes
    ✅ Probabilidades interpretables

    VARIANTES:
    ----------
    - MultinomialNB: Para conteos (Bag of Words, TF-IDF)
      ← USAMOS ESTA
    - BernoulliNB: Para datos binarios (palabra presente/ausente)
    - GaussianNB: Para datos continuos (no típico en texto)

    PARÁMETRO ALPHA (Suavizado de Laplace):
    ----------------------------------------
    Problema: ¿Qué pasa si una palabra NUNCA apareció en clase negativa?
    P("newword"|negativo) = 0/total = 0

    Entonces P(negativo|documento) = 0.5 × 0 × 0.3 = 0 ← PROBLEMA!

    Solución: Suavizado de Laplace
    P("newword"|negativo) = (0 + alpha) / (total + alpha × vocab_size)

    alpha = 1.0 → Suavizado estándar (recomendado)
    alpha → 0 → Sin suavizado (riesgoso)
    alpha > 1 → Más suavizado (más conservador)

    Args:
        X_train: Features de entrenamiento (TF-IDF o BoW)
        y_train: Etiquetas (0 o 1)
        alpha: Parámetro de suavizado de Laplace

    Returns:
        Modelo entrenado
    """
    print("\n" + "=" * 70)
    print("🧮 ENTRENANDO: NAIVE BAYES")
    print("=" * 70)

    if alpha is None:
        alpha = config.NAIVE_BAYES_ALPHA

    print(f"\n⚙️  Configuración:")
    print(f"   • Algoritmo: Multinomial Naive Bayes")
    print(f"   • Alpha (Laplace smoothing): {alpha}")
    print(f"   • Features shape: {X_train.shape}")
    print(f"   • Muestras: {len(y_train)}")

    # Crear y entrenar modelo
    model = MultinomialNB(alpha=alpha)

    print(f"\n🔄 Entrenando modelo...")
    model.fit(X_train, y_train)

    print(f"✅ Modelo entrenado!")

    # Información adicional
    print(f"\n📊 Estadísticas del modelo:")
    print(f"   • Clases: {model.classes_}")
    print(f"   • Features: {model.feature_count_.shape[1]}")

    # Probabilidades a priori
    class_priors = model.class_log_prior_
    print(f"\n📈 Probabilidades a priori de clases:")
    for cls, log_prior in zip(model.classes_, class_priors):
        prior = np.exp(log_prior)
        label = "Positivo" if cls == 1 else "Negativo"
        print(f"   • P({label}) = {prior:.3f}")

    return model


# ============================================================================
# 2. SUPPORT VECTOR MACHINE (Máquina de Vectores de Soporte)
# ============================================================================

def train_svm(X_train, y_train, C: float = None):
    """
    Entrena un clasificador SVM lineal.

    🎯 SUPPORT VECTOR MACHINE:
    ===========================

    IDEA CENTRAL:
    Encuentra el HIPERPLANO que mejor separa las clases con máximo margen.

    VISUALIZACIÓN 2D (simplificado):
    --------------------------------

           Clase Negativa (●)     |     Clase Positiva (○)
                                 |
              ●                  |              ○
                                 |
          ●       ●              |        ○         ○
                                 |
              ●                  |              ○
                          ◄──────|──────►
                           MARGEN (queremos maximizarlo)

    El hiperplano (línea en 2D) separa las clases.
    Los puntos más cercanos se llaman "support vectors" (vectores de soporte).

    EN ALTA DIMENSIÓN (texto):
    --------------------------
    Con 5000 features (palabras), el "hiperplano" está en 5000 dimensiones.
    Pero el concepto es el mismo: separar positivos de negativos.

    CÓMO FUNCIONA:
    --------------
    1. Encuentra el hiperplano que maximiza el margen
    2. Solo los "support vectors" (puntos críticos) importan
    3. Puede usar kernels para separaciones no lineales

    TIPOS DE KERNEL:
    ----------------
    - Linear: w₁x₁ + w₂x₂ + ... + b = 0  ← USAMOS ESTE
      ✅ Rápido
      ✅ Funciona muy bien con texto (datos de alta dimensión)
      ✅ Interpretable (pesos de palabras)

    - RBF (Radial Basis Function): No lineal
      ❌ Más lento
      ❌ Requiere más memoria
      ⚠️ Puede hacer overfitting con texto

    - Polynomial: No lineal
      Similar a RBF

    PARÁMETRO C (Regularización):
    -----------------------------
    Controla el trade-off entre:
    - Maximizar margen (generalización)
    - Minimizar errores de clasificación (precisión en training)

    C pequeño (ej: 0.1):
    ✅ Margen más amplio
    ✅ Mejor generalización
    ❌ Permite más errores en training
    → Prefiere simplicidad

    C grande (ej: 10):
    ✅ Menos errores en training
    ❌ Margen más estrecho
    ❌ Riesgo de overfitting
    → Prefiere precisión

    C = 1.0 → Balance estándar (recomendado)

    POR QUÉ SVM FUNCIONA BIEN CON TEXTO:
    -------------------------------------
    ✅ Excelente con datos de alta dimensión
    ✅ Robusto a overfitting (con C apropiado)
    ✅ Solo usa support vectors (eficiente)
    ✅ Funciona bien con features sparse (TF-IDF)

    Args:
        X_train: Features de entrenamiento (TF-IDF)
        y_train: Etiquetas (0 o 1)
        C: Parámetro de regularización

    Returns:
        Modelo entrenado
    """
    print("\n" + "=" * 70)
    print("🎯 ENTRENANDO: SUPPORT VECTOR MACHINE (SVM)")
    print("=" * 70)

    if C is None:
        C = config.SVM_C

    print(f"\n⚙️  Configuración:")
    print(f"   • Algoritmo: Linear SVM")
    print(f"   • C (regularización): {C}")
    print(f"   • Kernel: Linear")
    print(f"   • Features shape: {X_train.shape}")
    print(f"   • Muestras: {len(y_train)}")

    # Crear y entrenar modelo
    model = LinearSVC(
        C=C,
        max_iter=2000,  # Máximo de iteraciones
        random_state=config.RANDOM_STATE
    )

    print(f"\n🔄 Entrenando modelo...")
    model.fit(X_train, y_train)

    print(f"✅ Modelo entrenado!")

    # Información adicional
    print(f"\n📊 Estadísticas del modelo:")
    print(f"   • Clases: {model.classes_}")
    print(f"   • Features: {model.coef_.shape[1]}")
    print(f"   • Número de iteraciones: {model.n_iter_}")

    return model


# ============================================================================
# 3. LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(X_train, y_train, C: float = 1.0):
    """
    Entrena un clasificador Logistic Regression.

    📈 LOGISTIC REGRESSION:
    =======================

    FUNCIÓN SIGMOIDE:
    P(positivo|x) = 1 / (1 + e^(-z))
    donde z = w₁x₁ + w₂x₂ + ... + b

    VENTAJAS:
    ✅ Probabilidades calibradas
    ✅ Interpretable (coeficientes = importancia de palabras)
    ✅ Rápido de entrenar
    ✅ Funciona bien con regularización

    Args:
        X_train: Features de entrenamiento
        y_train: Etiquetas
        C: Inverso de regularización (más alto = menos regularización)

    Returns:
        Modelo entrenado
    """
    print("\n" + "=" * 70)
    print("📈 ENTRENANDO: LOGISTIC REGRESSION")
    print("=" * 70)

    print(f"\n⚙️  Configuración:")
    print(f"   • C (inverso de regularización): {C}")
    print(f"   • Solver: lbfgs")
    print(f"   • Features shape: {X_train.shape}")

    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=config.RANDOM_STATE
    )

    print(f"\n🔄 Entrenando modelo...")
    model.fit(X_train, y_train)

    print(f"✅ Modelo entrenado!")

    return model


# ============================================================================
# 4. EVALUACIÓN DE MODELOS
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> Dict:
    """
    Evalúa un modelo y retorna métricas.

    📊 MÉTRICAS DE CLASIFICACIÓN:
    =============================

    1. ACCURACY (Exactitud):
       ¿Qué porcentaje de predicciones son correctas?
       Accuracy = (VP + VN) / Total

    2. PRECISION (Precisión):
       De las reviews que predijimos como POSITIVAS, ¿cuántas son realmente positivas?
       Precision = VP / (VP + FP)
       Alta precisión → Pocas falsas alarmas

    3. RECALL (Sensibilidad/Exhaustividad):
       De todas las reviews REALMENTE positivas, ¿cuántas identificamos?
       Recall = VP / (VP + FN)
       Alto recall → No nos perdemos muchos positivos

    4. F1-SCORE:
       Media armónica de Precision y Recall
       F1 = 2 × (Precision × Recall) / (Precision + Recall)
       Balance entre precision y recall

    MATRIZ DE CONFUSIÓN:
    --------------------
                    Predicho Negativo   Predicho Positivo
    Real Negativo         VN (✅)             FP (❌)
    Real Positivo         FN (❌)             VP (✅)

    VN = Verdaderos Negativos (correcto: predijimos neg, es neg)
    VP = Verdaderos Positivos (correcto: predijimos pos, es pos)
    FP = Falsos Positivos (error: predijimos pos, es neg)
    FN = Falsos Negativos (error: predijimos neg, es pos)

    Args:
        model: Modelo entrenado
        X_test: Features de prueba
        y_test: Etiquetas reales
        model_name: Nombre del modelo para display

    Returns:
        Diccionario con métricas
    """
    print("\n" + "=" * 70)
    print(f"📊 EVALUANDO: {model_name}")
    print("=" * 70)

    # Predicciones
    print(f"\n🔮 Realizando predicciones...")
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n✅ Métricas calculadas!")

    # Mostrar métricas
    print(f"\n📈 MÉTRICAS DE CLASIFICACIÓN:")
    print("-" * 70)
    print(f"   • Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"   • Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"   • F1-Score:  {f1:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 MATRIZ DE CONFUSIÓN:")
    print("-" * 70)
    print(f"                Pred Neg    Pred Pos")
    print(f"   Real Neg     {cm[0,0]:>6}      {cm[0,1]:>6}    (FP)")
    print(f"   Real Pos     {cm[1,0]:>6}      {cm[1,1]:>6}")
    print(f"               (FN)         (VP)")

    # Interpretación
    print(f"\n💡 INTERPRETACIÓN:")
    print("-" * 70)
    print(f"   • Verdaderos Negativos (VN): {cm[0,0]} ✅")
    print(f"   • Verdaderos Positivos (VP): {cm[1,1]} ✅")
    print(f"   • Falsos Positivos (FP): {cm[0,1]} ❌ (predijimos pos, era neg)")
    print(f"   • Falsos Negativos (FN): {cm[1,0]} ❌ (predijimos neg, era pos)")

    # Reporte detallado
    print(f"\n📋 REPORTE DETALLADO:")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

    # Retornar métricas
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }

    return metrics


# ============================================================================
# 5. VALIDACIÓN CRUZADA
# ============================================================================

def cross_validate_model(model, X, y, cv: int = 5):
    """
    Realiza validación cruzada para evaluar robustez del modelo.

    🔄 CROSS-VALIDATION (Validación Cruzada):
    ==========================================

    En lugar de una sola división train/test, divide en K folds:

    Fold 1: [Train] [Train] [Train] [Train] [Test]
    Fold 2: [Train] [Train] [Train] [Test] [Train]
    Fold 3: [Train] [Train] [Test] [Train] [Train]
    Fold 4: [Train] [Test] [Train] [Train] [Train]
    Fold 5: [Test] [Train] [Train] [Train] [Train]

    Entrena K veces, cada vez con un fold diferente como test.
    Promedia los resultados → Estimación más robusta.

    VENTAJAS:
    ✅ Usa todos los datos para training Y testing
    ✅ Estimación más confiable del rendimiento
    ✅ Detecta overfitting

    Args:
        model: Modelo a evaluar
        X: Features completas
        y: Etiquetas completas
        cv: Número de folds

    Returns:
        Scores de cada fold
    """
    print(f"\n🔄 Validación cruzada ({cv} folds)...")

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    print(f"\n✅ Validación cruzada completada!")
    print(f"\n📊 Scores por fold:")
    for i, score in enumerate(scores, 1):
        print(f"   Fold {i}: {score:.4f} ({score*100:.2f}%)")

    print(f"\n📈 Resumen:")
    print(f"   • Media: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
    print(f"   • Desviación estándar: {scores.std():.4f}")
    print(f"   • Rango: [{scores.min():.4f}, {scores.max():.4f}]")

    return scores


# ============================================================================
# 6. ANÁLISIS DE CARACTERÍSTICAS
# ============================================================================

def analyze_feature_importance(model, vectorizer, top_n: int = 20):
    """
    Analiza qué palabras son más importantes para el modelo.

    🔍 INTERPRETABILIDAD:
    =====================

    Para modelos lineales (SVM, Logistic Regression), los coeficientes
    indican la importancia de cada feature (palabra).

    Coeficiente POSITIVO alto → Palabra asociada a sentimiento POSITIVO
    Coeficiente NEGATIVO alto → Palabra asociada a sentimiento NEGATIVO

    Ejemplo:
    "excellent": +2.5 → Fuerte indicador de sentimiento positivo
    "terrible": -2.1 → Fuerte indicador de sentimiento negativo

    Args:
        model: Modelo lineal entrenado
        vectorizer: TfidfVectorizer usado
        top_n: Número de palabras top a mostrar

    Returns:
        None (imprime resultados)
    """
    print("\n" + "=" * 70)
    print("🔍 ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
    print("=" * 70)

    # Obtener coeficientes
    try:
        coef = model.coef_[0]
    except AttributeError:
        print("⚠️  Este modelo no tiene coeficientes interpretables (ej: Naive Bayes)")
        return

    # Obtener nombres de features
    feature_names = vectorizer.get_feature_names_out()

    # Top palabras positivas
    top_positive_indices = np.argsort(coef)[-top_n:][::-1]

    print(f"\n🟢 TOP {top_n} PALABRAS MÁS POSITIVAS:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Palabra':<20} {'Coeficiente':<12}")
    print("-" * 70)
    for rank, idx in enumerate(top_positive_indices, 1):
        print(f"{rank:<6} {feature_names[idx]:<20} {coef[idx]:>11.4f}")

    # Top palabras negativas
    top_negative_indices = np.argsort(coef)[:top_n]

    print(f"\n🔴 TOP {top_n} PALABRAS MÁS NEGATIVAS:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Palabra':<20} {'Coeficiente':<12}")
    print("-" * 70)
    for rank, idx in enumerate(top_negative_indices, 1):
        print(f"{rank:<6} {feature_names[idx]:<20} {coef[idx]:>11.4f}")


# ============================================================================
# 7. GUARDAR Y CARGAR MODELOS
# ============================================================================

def save_model(model, filepath):
    """Guarda modelo entrenado."""
    joblib.dump(model, filepath)
    print(f"\n💾 Modelo guardado: {filepath.name}")


def load_model(filepath):
    """Carga modelo guardado."""
    model = joblib.load(filepath)
    print(f"\n📂 Modelo cargado: {filepath.name}")
    return model


# ============================================================================
# DEMO Y COMPARACIÓN DE MODELOS
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Comparación de modelos clásicos
    """
    print("\n" + "🤖" * 35)
    print(" DEMO: CLASSICAL ML MODELS")
    print("🤖" * 35)

    print("\n💡 Esta demo requiere datos reales.")
    print("   Ejecuta main.py para el pipeline completo.")
    print("\n✅ Modelos disponibles:")
    print("   1. Naive Bayes - Rápido y eficiente")
    print("   2. SVM - Preciso y robusto")
    print("   3. Logistic Regression - Interpretable")
