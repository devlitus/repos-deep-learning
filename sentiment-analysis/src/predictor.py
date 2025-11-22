"""
Predictor - Análisis de Sentimientos
=====================================

Este módulo permite hacer predicciones de sentimientos en nuevos textos
usando modelos entrenados.

Similar a predictor.py de otros proyectos, pero adaptado para NLP.
"""

import numpy as np
import joblib
from tensorflow import keras
from pathlib import Path
from typing import List, Union, Tuple
import config

# ============================================================================
# PREDICTOR CON MODELOS CLÁSICOS (TF-IDF + SVM/NaiveBayes)
# ============================================================================

class SentimentPredictorClassic:
    """
    Predictor usando modelos clásicos (SVM, Naive Bayes).

    PIPELINE:
    ---------
    1. Texto → Preprocesamiento
    2. Texto preprocesado → TF-IDF vectorizer
    3. TF-IDF features → Modelo ML
    4. Modelo → Predicción (0 o 1)
    """

    def __init__(self, model_path: Path, vectorizer_path: Path):
        """
        Inicializa predictor.

        Args:
            model_path: Ruta al modelo guardado (.pkl)
            vectorizer_path: Ruta al vectorizer guardado (.pkl)
        """
        print("\n" + "=" * 70)
        print("🔮 INICIALIZANDO PREDICTOR CLÁSICO")
        print("=" * 70)

        print(f"\n📂 Cargando modelo: {model_path.name}")
        self.model = joblib.load(model_path)

        print(f"📂 Cargando vectorizer: {vectorizer_path.name}")
        self.vectorizer = joblib.load(vectorizer_path)

        print(f"\n✅ Predictor inicializado!")

    def preprocess(self, text: str) -> str:
        """
        Preprocesa texto para predicción.

        Aplica el mismo preprocesamiento que en entrenamiento.

        Args:
            text: Texto original

        Returns:
            Texto preprocesado
        """
        from src.text_preprocessing import preprocess_text

        return preprocess_text(
            text,
            remove_html=config.REMOVE_HTML_TAGS,
            remove_url=True,
            remove_punct=config.REMOVE_PUNCTUATION,
            remove_num=config.REMOVE_NUMBERS,
            lowercase=config.CONVERT_TO_LOWERCASE,
            remove_stops=config.REMOVE_STOPWORDS,
            apply_stemming=config.USE_STEMMING,
            apply_lemmatization=config.USE_LEMMATIZATION
        )

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predice sentimiento de texto(s).

        Args:
            text: Texto individual o lista de textos

        Returns:
            Array de predicciones (0=negativo, 1=positivo)
        """
        # Convertir a lista si es un solo texto
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Preprocesar
        processed_texts = [self.preprocess(t) for t in texts]

        # Vectorizar
        features = self.vectorizer.transform(processed_texts)

        # Predecir
        predictions = self.model.predict(features)

        if single_input:
            return predictions[0]
        return predictions

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predice probabilidades de sentimiento.

        Args:
            text: Texto individual o lista de textos

        Returns:
            Array de probabilidades [[prob_neg, prob_pos], ...]
        """
        # Convertir a lista si es un solo texto
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Preprocesar
        processed_texts = [self.preprocess(t) for t in texts]

        # Vectorizar
        features = self.vectorizer.transform(processed_texts)

        # Predecir probabilidades
        try:
            probabilities = self.model.predict_proba(features)
        except AttributeError:
            # Si el modelo no tiene predict_proba (ej: LinearSVC)
            # Usar decision_function y convertir a probabilidades
            decision = self.model.decision_function(features)
            # Convertir a probabilidades usando sigmoid
            prob_pos = 1 / (1 + np.exp(-decision))
            prob_neg = 1 - prob_pos
            probabilities = np.column_stack([prob_neg, prob_pos])

        if single_input:
            return probabilities[0]
        return probabilities

    def predict_with_confidence(self, text: str) -> Tuple[int, float, str]:
        """
        Predice sentimiento con nivel de confianza.

        Args:
            text: Texto a analizar

        Returns:
            (predicción, confianza, etiqueta)
        """
        prediction = self.predict(text)
        probas = self.predict_proba(text)

        confidence = max(probas)
        label = "Positivo ✅" if prediction == 1 else "Negativo ❌"

        return prediction, confidence, label


# ============================================================================
# PREDICTOR CON MODELO LSTM
# ============================================================================

class SentimentPredictorLSTM:
    """
    Predictor usando modelo LSTM.

    PIPELINE:
    ---------
    1. Texto → Preprocesamiento
    2. Texto preprocesado → Tokenizer (palabras → índices)
    3. Índices → Padding (mismo tamaño)
    4. Secuencia paddeada → Modelo LSTM
    5. Modelo → Probabilidad [0.0 - 1.0]
    """

    def __init__(self, model_path: Path, tokenizer_path: Path):
        """
        Inicializa predictor LSTM.

        Args:
            model_path: Ruta al modelo Keras (.keras)
            tokenizer_path: Ruta al tokenizer (.pkl)
        """
        print("\n" + "=" * 70)
        print("🔮 INICIALIZANDO PREDICTOR LSTM")
        print("=" * 70)

        print(f"\n📂 Cargando modelo LSTM: {model_path.name}")
        self.model = keras.models.load_model(model_path)

        print(f"📂 Cargando tokenizer: {tokenizer_path.name}")
        self.tokenizer = joblib.load(tokenizer_path)

        self.max_length = config.MAX_SEQUENCE_LENGTH

        print(f"\n✅ Predictor LSTM inicializado!")
        print(f"   • Longitud máxima de secuencia: {self.max_length}")

    def preprocess(self, text: str) -> str:
        """
        Preprocesa texto (más simple que para modelos clásicos).

        Para LSTM, no necesitamos remover stopwords ni hacer stemming
        porque el modelo aprende las relaciones entre palabras.

        Args:
            text: Texto original

        Returns:
            Texto preprocesado
        """
        from src.text_preprocessing import clean_text

        return clean_text(
            text,
            remove_html=True,
            remove_url=True,
            remove_punct=config.REMOVE_PUNCTUATION,
            remove_num=config.REMOVE_NUMBERS,
            lowercase=True
        )

    def texts_to_sequences(self, texts: List[str]) -> np.ndarray:
        """
        Convierte textos a secuencias numéricas paddeadas.

        Args:
            texts: Lista de textos

        Returns:
            Array numpy de secuencias
        """
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        # Tokenizar
        sequences = self.tokenizer.texts_to_sequences(texts)

        # Padding
        padded = pad_sequences(
            sequences,
            maxlen=self.max_length,
            padding='pre',
            truncating='post'
        )

        return padded

    def predict(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predice sentimiento de texto(s).

        Args:
            text: Texto individual o lista de textos

        Returns:
            Array de predicciones (0=negativo, 1=positivo)
        """
        # Convertir a lista si es un solo texto
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Preprocesar
        processed_texts = [self.preprocess(t) for t in texts]

        # Convertir a secuencias
        sequences = self.texts_to_sequences(processed_texts)

        # Predecir
        probabilities = self.model.predict(sequences, verbose=0)
        predictions = (probabilities > 0.5).astype(int).flatten()

        if single_input:
            return predictions[0]
        return predictions

    def predict_proba(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Predice probabilidad de sentimiento positivo.

        Args:
            text: Texto individual o lista de textos

        Returns:
            Array de probabilidades [0.0 - 1.0]
        """
        # Convertir a lista si es un solo texto
        if isinstance(text, str):
            texts = [text]
            single_input = True
        else:
            texts = text
            single_input = False

        # Preprocesar
        processed_texts = [self.preprocess(t) for t in texts]

        # Convertir a secuencias
        sequences = self.texts_to_sequences(processed_texts)

        # Predecir
        probabilities = self.model.predict(sequences, verbose=0).flatten()

        if single_input:
            return probabilities[0]
        return probabilities

    def predict_with_confidence(self, text: str) -> Tuple[int, float, str]:
        """
        Predice sentimiento con nivel de confianza.

        Args:
            text: Texto a analizar

        Returns:
            (predicción, probabilidad, etiqueta)
        """
        prediction = self.predict(text)
        probability = self.predict_proba(text)

        # Confianza es qué tan lejos está de 0.5
        confidence = abs(probability - 0.5) * 2

        label = "Positivo ✅" if prediction == 1 else "Negativo ❌"

        return prediction, confidence, label


# ============================================================================
# FUNCIÓN HELPER PARA PREDICCIÓN INTERACTIVA
# ============================================================================

def predict_sentiment_interactive(predictor,
                                 text: str,
                                 show_details: bool = True):
    """
    Predice y muestra resultado de forma interactiva.

    Args:
        predictor: SentimentPredictorClassic o SentimentPredictorLSTM
        text: Texto a analizar
        show_details: Mostrar detalles de la predicción
    """
    print("\n" + "=" * 70)
    print("🔮 PREDICCIÓN DE SENTIMIENTO")
    print("=" * 70)

    print(f"\n📝 Texto original:")
    print(f"   \"{text}\"")

    # Predecir
    prediction, confidence, label = predictor.predict_with_confidence(text)

    # Mostrar resultado
    print(f"\n🎯 Resultado:")
    print(f"   • Sentimiento: {label}")
    print(f"   • Confianza: {confidence:.2%}")

    if show_details:
        # Mostrar texto preprocesado
        processed = predictor.preprocess(text)
        print(f"\n🧹 Texto preprocesado:")
        print(f"   \"{processed}\"")

        # Mostrar probabilidades
        if hasattr(predictor, 'predict_proba'):
            if isinstance(predictor, SentimentPredictorLSTM):
                proba = predictor.predict_proba(text)
                print(f"\n📊 Probabilidades:")
                print(f"   • P(Positivo): {proba:.4f} ({proba*100:.2f}%)")
                print(f"   • P(Negativo): {1-proba:.4f} ({(1-proba)*100:.2f}%)")
            else:
                probas = predictor.predict_proba(text)
                print(f"\n📊 Probabilidades:")
                print(f"   • P(Negativo): {probas[0]:.4f} ({probas[0]*100:.2f}%)")
                print(f"   • P(Positivo): {probas[1]:.4f} ({probas[1]*100:.2f}%)")


def batch_predict(predictor, texts: List[str]):
    """
    Predice sentimientos para múltiples textos.

    Args:
        predictor: Predictor a usar
        texts: Lista de textos
    """
    print("\n" + "=" * 70)
    print(f"🔮 PREDICCIÓN EN LOTE: {len(texts)} textos")
    print("=" * 70)

    predictions = predictor.predict(texts)
    probabilities = predictor.predict_proba(texts)

    for i, (text, pred, proba) in enumerate(zip(texts, predictions, probabilities), 1):
        label = "Positivo ✅" if pred == 1 else "Negativo ❌"

        if isinstance(predictor, SentimentPredictorLSTM):
            confidence = abs(proba - 0.5) * 2
        else:
            confidence = max(proba)

        print(f"\n{i}. {label} (confianza: {confidence:.2%})")
        print(f"   \"{text[:80]}{'...' if len(text) > 80 else ''}\"")


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Predicción de sentimientos
    """
    print("\n" + "🔮" * 35)
    print(" DEMO: SENTIMENT PREDICTOR")
    print("🔮" * 35)

    print("\n💡 Este módulo requiere modelos entrenados.")
    print("   Ejecuta main.py primero para entrenar los modelos.")

    # Ejemplos de uso
    print("\n" + "=" * 70)
    print("📚 EJEMPLOS DE USO")
    print("=" * 70)

    print("\n1️⃣  Predictor Clásico (TF-IDF + SVM):")
    print("-" * 70)
    print("""
    from src.predictor import SentimentPredictorClassic
    import config

    predictor = SentimentPredictorClassic(
        model_path=config.MODEL_TFIDF_SVM,
        vectorizer_path=config.TFIDF_VECTORIZER_FILE
    )

    text = "This movie is absolutely excellent!"
    prediction, confidence, label = predictor.predict_with_confidence(text)
    print(f"{label} (confianza: {confidence:.2%})")
    """)

    print("\n2️⃣  Predictor LSTM:")
    print("-" * 70)
    print("""
    from src.predictor import SentimentPredictorLSTM
    import config

    predictor = SentimentPredictorLSTM(
        model_path=config.MODEL_LSTM,
        tokenizer_path=config.TOKENIZER_FILE
    )

    text = "Terrible waste of time, very disappointing"
    prediction, confidence, label = predictor.predict_with_confidence(text)
    print(f"{label} (confianza: {confidence:.2%})")
    """)

    print("\n3️⃣  Predicción interactiva:")
    print("-" * 70)
    print("""
    from src.predictor import predict_sentiment_interactive

    predict_sentiment_interactive(predictor, "Amazing movie, loved it!")
    """)

    print("\n✅ Demo completada!")
