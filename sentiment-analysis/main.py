"""
Sentiment Analysis - Pipeline Completo
=======================================

Este es el archivo principal que ejecuta el pipeline completo end-to-end
de análisis de sentimientos.

PIPELINE:
---------
1. Cargar datos (IMDB dataset)
2. Explorar datos
3. Preprocesar texto (opcional)
4. Crear features (TF-IDF para modelos clásicos)
5. Entrenar modelos:
   - Naive Bayes
   - SVM
   - LSTM
6. Evaluar modelos
7. Visualizar resultados
8. Guardar modelos
9. Hacer predicciones de ejemplo

Similar a main.py de otros proyectos, pero adaptado para NLP.
"""

import numpy as np
from pathlib import Path
import config

# Importar módulos del proyecto
from src import data_loader
from src import text_preprocessing
from src import feature_extraction
from src import model
from src import deep_model
from src import visualizations
from src import predictor

# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================

def main():
    """
    Pipeline completo de análisis de sentimientos.
    """
    print("\n" + "🎬" * 35)
    print(" SENTIMENT ANALYSIS - PIPELINE COMPLETO")
    print("🎬" * 35)

    # Crear directorios si no existen
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    # ========================================================================
    # PASO 1: CARGAR DATOS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 1: CARGA DE DATOS")
    print("=" * 70)

    (X_train_raw, y_train), (X_test_raw, y_test) = data_loader.load_imdb_data()

    # ========================================================================
    # PASO 2: EXPLORAR DATOS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 2: EXPLORACIÓN DE DATOS")
    print("=" * 70)

    data_loader.explore_imdb_data(X_train_raw, y_train, X_test_raw, y_test)

    # Obtener diccionario de palabras
    word_index = data_loader.get_word_index()

    # Decodificar ejemplos
    print("\n📝 Ejemplo de review decodificada:")
    example_decoded = data_loader.decode_review(X_train_raw[0], word_index)
    print(f"   {example_decoded[:200]}...")

    # ========================================================================
    # PASO 3: VISUALIZACIONES INICIALES
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 3: VISUALIZACIONES INICIALES")
    print("=" * 70)

    # Decodificar todos los textos para visualizaciones
    print("\n🔄 Decodificando reviews para visualizaciones...")
    print("   (Esto puede tomar unos minutos...)")

    train_texts = [data_loader.decode_review(review, word_index) for review in X_train_raw[:5000]]
    train_labels = y_train[:5000]

    # Separar por sentimiento
    positive_texts = [text for text, label in zip(train_texts, train_labels) if label == 1]
    negative_texts = [text for text, label in zip(train_texts, train_labels) if label == 0]

    # Distribución de sentimientos
    visualizations.plot_sentiment_distribution(train_labels)

    # Distribución de longitudes
    visualizations.plot_text_length_distribution(train_texts, train_labels)

    # Word clouds (opcional - puede tardar)
    print("\n☁️  Generando word clouds...")
    visualizations.create_sentiment_wordclouds(
        positive_texts[:1000],
        negative_texts[:1000],
        save_dir=config.REPORTS_DIR
    )

    # ========================================================================
    # PASO 4A: ENTRENAR MODELOS CLÁSICOS (TF-IDF + ML)
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 4A: MODELOS CLÁSICOS (TF-IDF + ML)")
    print("=" * 70)

    # Preprocesar textos
    print("\n🧹 Preprocesando textos para modelos clásicos...")
    train_texts_processed = [
        text_preprocessing.preprocess_text(text) for text in train_texts
    ]
    test_texts = [data_loader.decode_review(review, word_index) for review in X_test_raw[:5000]]
    test_texts_processed = [
        text_preprocessing.preprocess_text(text) for text in test_texts
    ]
    test_labels = y_test[:5000]

    # Crear features TF-IDF
    print("\n📊 Creando features TF-IDF...")
    X_train_tfidf, tfidf_vectorizer = feature_extraction.create_tfidf_features(
        train_texts_processed
    )
    X_test_tfidf, _ = feature_extraction.create_tfidf_features(
        test_texts_processed,
        vectorizer=tfidf_vectorizer
    )

    # Entrenar Naive Bayes
    print("\n" + "-" * 70)
    print("4A.1: NAIVE BAYES")
    print("-" * 70)

    nb_model = model.train_naive_bayes(X_train_tfidf, train_labels)
    nb_metrics = model.evaluate_model(nb_model, X_test_tfidf, test_labels, "Naive Bayes")

    # Entrenar SVM
    print("\n" + "-" * 70)
    print("4A.2: SUPPORT VECTOR MACHINE (SVM)")
    print("-" * 70)

    svm_model = model.train_svm(X_train_tfidf, train_labels)
    svm_metrics = model.evaluate_model(svm_model, X_test_tfidf, test_labels, "SVM")

    # Análisis de características
    print("\n" + "-" * 70)
    print("4A.3: ANÁLISIS DE CARACTERÍSTICAS")
    print("-" * 70)

    model.analyze_feature_importance(svm_model, tfidf_vectorizer, top_n=20)

    # Guardar modelos clásicos
    print("\n💾 Guardando modelos clásicos...")
    model.save_model(nb_model, config.MODEL_TFIDF_NAIVE_BAYES)
    model.save_model(svm_model, config.MODEL_TFIDF_SVM)
    feature_extraction.save_vectorizer(tfidf_vectorizer, config.TFIDF_VECTORIZER_FILE)

    # ========================================================================
    # PASO 4B: ENTRENAR MODELO LSTM
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 4B: MODELO DEEP LEARNING (LSTM)")
    print("=" * 70)

    # Preparar datos para LSTM
    print("\n📊 Preparando datos para LSTM...")

    # Crear tokenizer
    all_texts_for_tokenizer = train_texts + test_texts
    tokenizer = deep_model.create_tokenizer(all_texts_for_tokenizer)

    # Convertir a secuencias
    X_train_seq = deep_model.texts_to_sequences_padded(train_texts, tokenizer)
    X_test_seq = deep_model.texts_to_sequences_padded(test_texts, tokenizer)

    # Split para validación
    from sklearn.model_selection import train_test_split
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
        X_train_seq, train_labels,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=train_labels
    )

    print(f"\n📊 Datos preparados:")
    print(f"   • Training: {X_train_lstm.shape}")
    print(f"   • Validation: {X_val_lstm.shape}")
    print(f"   • Test: {X_test_seq.shape}")

    # Construir modelo
    print("\n🧠 Construyendo modelo LSTM...")
    vocab_size = len(tokenizer.word_index) + 1
    lstm_model = deep_model.build_lstm_model(vocab_size)

    # Entrenar modelo
    print("\n🚀 Entrenando modelo LSTM...")
    print("   (Esto puede tomar varios minutos dependiendo de tu hardware...)")

    history = deep_model.train_lstm_model(
        lstm_model,
        X_train_lstm, y_train_lstm,
        X_val_lstm, y_val_lstm
    )

    # Evaluar modelo
    print("\n📊 Evaluando modelo LSTM...")
    lstm_metrics = deep_model.evaluate_lstm_model(lstm_model, X_test_seq, test_labels)

    # Visualizar historial de entrenamiento
    visualizations.plot_training_history(
        history,
        save_path=config.REPORT_TRAINING_HISTORY
    )

    # Guardar modelo LSTM
    print("\n💾 Guardando modelo LSTM...")
    lstm_model.save(config.MODEL_LSTM)
    deep_model.save_tokenizer(tokenizer, config.TOKENIZER_FILE)

    # ========================================================================
    # PASO 5: COMPARACIÓN DE MODELOS
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 5: COMPARACIÓN DE MODELOS")
    print("=" * 70)

    print("\n📊 RESUMEN DE RESULTADOS:")
    print("-" * 70)
    print(f"{'Modelo':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 70)

    # Naive Bayes
    print(f"{'Naive Bayes':<20} "
          f"{nb_metrics['accuracy']:<12.4f} "
          f"{nb_metrics['precision']:<12.4f} "
          f"{nb_metrics['recall']:<12.4f} "
          f"{nb_metrics['f1_score']:<12.4f}")

    # SVM
    print(f"{'SVM':<20} "
          f"{svm_metrics['accuracy']:<12.4f} "
          f"{svm_metrics['precision']:<12.4f} "
          f"{svm_metrics['recall']:<12.4f} "
          f"{svm_metrics['f1_score']:<12.4f}")

    # LSTM
    print(f"{'LSTM':<20} "
          f"{lstm_metrics['accuracy']:<12.4f} "
          f"{lstm_metrics.get('precision', 0):<12.4f} "
          f"{lstm_metrics.get('recall', 0):<12.4f} "
          f"{'N/A':<12}")

    print("-" * 70)

    # Determinar mejor modelo
    best_model_name = max(
        [('Naive Bayes', nb_metrics['accuracy']),
         ('SVM', svm_metrics['accuracy']),
         ('LSTM', lstm_metrics['accuracy'])],
        key=lambda x: x[1]
    )[0]

    print(f"\n🏆 MEJOR MODELO: {best_model_name}")

    # ========================================================================
    # PASO 6: PREDICCIONES DE EJEMPLO
    # ========================================================================
    print("\n" + "=" * 70)
    print("PASO 6: PREDICCIONES DE EJEMPLO")
    print("=" * 70)

    # Crear predictores
    print("\n🔮 Cargando predictores...")

    predictor_svm = predictor.SentimentPredictorClassic(
        model_path=config.MODEL_TFIDF_SVM,
        vectorizer_path=config.TFIDF_VECTORIZER_FILE
    )

    predictor_lstm = predictor.SentimentPredictorLSTM(
        model_path=config.MODEL_LSTM,
        tokenizer_path=config.TOKENIZER_FILE
    )

    # Ejemplos de prueba
    test_examples = [
        "This movie is absolutely excellent! Best film I've ever seen.",
        "Terrible waste of time. Awful acting and boring plot.",
        "It was okay, nothing special but not terrible either.",
        "I loved every minute of it! Brilliant performance!",
        "Disappointing. Expected much better based on the reviews."
    ]

    print("\n" + "-" * 70)
    print("Predicciones con SVM:")
    print("-" * 70)

    for i, text in enumerate(test_examples, 1):
        pred, conf, label = predictor_svm.predict_with_confidence(text)
        print(f"\n{i}. {label} (confianza: {conf:.2%})")
        print(f"   \"{text}\"")

    print("\n" + "-" * 70)
    print("Predicciones con LSTM:")
    print("-" * 70)

    for i, text in enumerate(test_examples, 1):
        pred, conf, label = predictor_lstm.predict_with_confidence(text)
        print(f"\n{i}. {label} (confianza: {conf:.2%})")
        print(f"   \"{text}\"")

    # ========================================================================
    # FINALIZACIÓN
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE!")
    print("=" * 70)

    print(f"\n📁 Archivos generados:")
    print(f"   • Modelos guardados en: {config.MODELS_DIR}")
    print(f"   • Reportes guardados en: {config.REPORTS_DIR}")

    print(f"\n💡 Próximos pasos:")
    print(f"   1. Revisar las visualizaciones en {config.REPORTS_DIR}")
    print(f"   2. Probar los predictores con tus propios textos")
    print(f"   3. Experimentar con hiperparámetros en config.py")
    print(f"   4. Considerar crear una web app con Streamlit")

    print("\n🎉 ¡Felicidades! Has completado un proyecto completo de NLP")


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def quick_demo():
    """
    Demo rápido sin entrenar modelos (requiere modelos pre-entrenados).
    """
    print("\n" + "🔮" * 35)
    print(" QUICK DEMO - PREDICCIONES")
    print("🔮" * 35)

    # Verificar si existen modelos
    if not config.MODEL_TFIDF_SVM.exists():
        print("\n⚠️  No se encontraron modelos entrenados.")
        print("   Ejecuta main() primero para entrenar los modelos.")
        return

    # Cargar predictores
    predictor_svm = predictor.SentimentPredictorClassic(
        model_path=config.MODEL_TFIDF_SVM,
        vectorizer_path=config.TFIDF_VECTORIZER_FILE
    )

    # Predicción interactiva
    test_text = "This movie was absolutely amazing! I loved every second of it."

    predictor.predict_sentiment_interactive(predictor_svm, test_text)


def train_only_lstm():
    """
    Entrena solo el modelo LSTM (más rápido si ya tienes los modelos clásicos).
    """
    print("\n" + "🧠" * 35)
    print(" ENTRENAMIENTO: SOLO LSTM")
    print("🧠" * 35)

    # Cargar datos
    (X_train_raw, y_train), (X_test_raw, y_test) = data_loader.load_imdb_data()

    # Obtener diccionario
    word_index = data_loader.get_word_index()

    # Decodificar
    print("\n🔄 Decodificando reviews...")
    train_texts = [data_loader.decode_review(review, word_index) for review in X_train_raw]
    test_texts = [data_loader.decode_review(review, word_index) for review in X_test_raw]

    # Crear tokenizer
    print("\n📊 Creando tokenizer...")
    tokenizer = deep_model.create_tokenizer(train_texts + test_texts)

    # Convertir a secuencias
    print("\n🔄 Convirtiendo a secuencias...")
    X_train_seq = deep_model.texts_to_sequences_padded(train_texts, tokenizer)
    X_test_seq = deep_model.texts_to_sequences_padded(test_texts, tokenizer)

    # Split para validación
    from sklearn.model_selection import train_test_split
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
        X_train_seq, y_train,
        test_size=config.VALIDATION_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_train
    )

    # Construir y entrenar
    vocab_size = len(tokenizer.word_index) + 1
    lstm_model = deep_model.build_lstm_model(vocab_size)

    history = deep_model.train_lstm_model(
        lstm_model,
        X_train_lstm, y_train_lstm,
        X_val_lstm, y_val_lstm
    )

    # Evaluar
    lstm_metrics = deep_model.evaluate_lstm_model(lstm_model, X_test_seq, y_test)

    # Guardar
    lstm_model.save(config.MODEL_LSTM)
    deep_model.save_tokenizer(tokenizer, config.TOKENIZER_FILE)

    print("\n✅ Modelo LSTM entrenado y guardado!")


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import sys

    # Verificar si hay argumentos
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "demo":
            quick_demo()
        elif command == "lstm":
            train_only_lstm()
        else:
            print(f"❌ Comando desconocido: {command}")
            print("\nComandos disponibles:")
            print("  python main.py       - Pipeline completo")
            print("  python main.py demo  - Demo rápido (requiere modelos entrenados)")
            print("  python main.py lstm  - Entrenar solo LSTM")
    else:
        # Pipeline completo
        main()
