"""
Deep Learning Models - Análisis de Sentimientos
================================================

Este módulo contiene modelos de Deep Learning para clasificación de texto.

CONCEPTOS QUE YA CONOCES (de prediccion-temperatura):
------------------------------------------------------
✅ LSTM (Long Short-Term Memory)
✅ Secuencias de datos
✅ Callbacks (EarlyStopping, ReduceLROnPlateau)
✅ Validación durante entrenamiento

DIFERENCIAS CON PREDICCION-TEMPERATURA:
----------------------------------------
📊 prediccion-temperatura:
   - Secuencias de NÚMEROS (temperaturas)
   - Input: [23.5, 24.1, 23.8, ...]
   - Output: Próxima temperatura

📝 sentiment-analysis:
   - Secuencias de PALABRAS (como índices)
   - Input: [42, 789, 123, ...]  (índices de palabras)
   - Embedding layer: Convierte índices a vectores densos
   - Output: Probabilidad de sentimiento positivo

ARQUITECTURA TÍPICA:
--------------------
1. Input: [word_idx_1, word_idx_2, ..., word_idx_200]
   ↓
2. Embedding Layer: Convierte índices a vectores
   [word_vec_1, word_vec_2, ..., word_vec_200]
   ↓
3. LSTM Layer: Procesa secuencia de vectores
   Aprende dependencias temporales (orden de palabras importa!)
   ↓
4. Dense Layer: Clasificación final
   ↓
5. Output: Probabilidad [0.0 - 1.0]
   < 0.5 → Negativo
   ≥ 0.5 → Positivo
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout,
    Bidirectional, GlobalMaxPooling1D
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import joblib
from typing import Tuple, Dict
import config

# ============================================================================
# 1. CREAR TOKENIZER (Convertir texto a secuencias numéricas)
# ============================================================================

def create_tokenizer(texts: list, num_words: int = None):
    """
    Crea y entrena un tokenizer de Keras.

    🔤 KERAS TOKENIZER:
    ===================

    Similar al concepto que viste en data_loader.py, pero más completo.

    FUNCIONES:
    ----------
    1. fit_on_texts(texts): Construye vocabulario
       - Cuenta frecuencia de cada palabra
       - Asigna índices (1 = palabra más frecuente)

    2. texts_to_sequences(texts): Convierte texto a números
       "I love this movie" → [42, 789, 123, 456]

    EJEMPLO:
    --------
    Corpus:
    - "I love movies"
    - "I hate bad movies"

    Vocabulario (por frecuencia):
    1. "I" (2 veces)
    2. "movies" (2 veces)
    3. "love" (1 vez)
    4. "hate" (1 vez)
    5. "bad" (1 vez)

    "I love movies" → [1, 3, 2]
    "I hate bad movies" → [1, 4, 5, 2]

    Args:
        texts: Lista de textos para entrenar vocabulario
        num_words: Límite de vocabulario (top N palabras)

    Returns:
        Tokenizer entrenado
    """
    print("\n" + "=" * 70)
    print("🔤 CREANDO TOKENIZER")
    print("=" * 70)

    if num_words is None:
        num_words = config.NUM_WORDS

    print(f"\n⚙️  Configuración:")
    print(f"   • Vocabulario máximo: {num_words} palabras")
    print(f"   • Textos de entrenamiento: {len(texts)}")

    # Crear tokenizer
    tokenizer = Tokenizer(
        num_words=num_words,
        oov_token='<UNK>',  # Token para palabras desconocidas
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',  # Caracteres a remover
        lower=True  # Convertir a minúsculas
    )

    print(f"\n🔄 Entrenando tokenizer en corpus...")
    tokenizer.fit_on_texts(texts)

    print(f"\n✅ Tokenizer entrenado!")
    print(f"   • Vocabulario total: {len(tokenizer.word_index)} palabras únicas")
    print(f"   • Vocabulario usado: {min(num_words, len(tokenizer.word_index))}")

    # Mostrar ejemplos
    print(f"\n📚 Ejemplos de tokenización:")
    sample_words = list(tokenizer.word_index.items())[:10]
    for word, idx in sample_words:
        print(f"   '{word}' → {idx}")

    return tokenizer


def texts_to_sequences_padded(texts: list,
                              tokenizer,
                              maxlen: int = None) -> np.ndarray:
    """
    Convierte textos a secuencias numéricas con padding.

    🔧 PROCESO COMPLETO:
    ====================

    1. Texto → Secuencia de índices (tokenizer)
    2. Secuencias → Mismo tamaño (padding)

    EJEMPLO:
    --------
    Textos:
    - "I love this movie"
    - "Great"
    - "Terrible waste of time"

    Paso 1 - Tokenización:
    - [42, 789, 123, 456]  (4 palabras)
    - [234]  (1 palabra)
    - [567, 890, 12, 345]  (4 palabras)

    Paso 2 - Padding (maxlen=5):
    - [0, 42, 789, 123, 456]  (padding al inicio)
    - [0, 0, 0, 0, 234]
    - [0, 567, 890, 12, 345]  (truncado si fuera más largo)

    RESULTADO: Matriz (3, 5) lista para LSTM!

    Args:
        texts: Lista de textos
        tokenizer: Tokenizer entrenado
        maxlen: Longitud máxima de secuencia

    Returns:
        Array numpy con secuencias paddeadas
    """
    if maxlen is None:
        maxlen = config.MAX_SEQUENCE_LENGTH

    print(f"\n🔄 Convirtiendo textos a secuencias...")

    # Tokenizar
    sequences = tokenizer.texts_to_sequences(texts)

    print(f"✅ Textos tokenizados: {len(sequences)} secuencias")

    # Aplicar padding
    print(f"\n🔧 Aplicando padding (longitud={maxlen})...")
    padded = pad_sequences(
        sequences,
        maxlen=maxlen,
        padding='pre',  # Padding al inicio
        truncating='post'  # Truncar al final si es muy largo
    )

    print(f"✅ Padding aplicado: forma {padded.shape}")

    return padded


# ============================================================================
# 2. CONSTRUIR MODELO LSTM
# ============================================================================

def build_lstm_model(vocab_size: int,
                    embedding_dim: int = None,
                    embedding_matrix: np.ndarray = None,
                    max_length: int = None):
    """
    Construye un modelo LSTM para clasificación de sentimientos.

    🧠 ARQUITECTURA DEL MODELO:
    ===========================

    1. EMBEDDING LAYER:
       ---------------------
       Input: Secuencia de índices [42, 789, 123, ...]
       Output: Secuencia de vectores [[0.2, -0.5, ...], [0.8, 0.1, ...], ...]

       ¿QUÉ HACE?
       Convierte cada índice de palabra en un vector denso.
       Similar a un "lookup table" pero los vectores se aprenden.

       ANALOGÍA con prediccion-temperatura:
       - prediccion-temperatura: Input directo son números
       - sentiment-analysis: Input son índices → Embedding → vectores

    2. LSTM LAYER (ya conoces esto!):
       --------------------------------
       Procesa la secuencia de vectores de palabras.

       DIFERENCIA CLAVE:
       - prediccion-temperatura: [temp_t-3, temp_t-2, temp_t-1] → temp_t
       - sentiment-analysis: [word_1_vec, word_2_vec, ..., word_n_vec] → sentimiento

       LSTM "recuerda" contexto:
       "not" + "good" → LSTM entiende que "not" invierte el sentimiento

       Sin LSTM (solo bag of words):
       "not good" = ["not", "good"] → Puede confundir con "good"

       Con LSTM:
       "not" → hidden_state_1
       "good" + hidden_state_1 → hidden_state_2 (negativo!)

    3. BIDIRECTIONAL LSTM (opcional):
       ---------------------------------
       Lee la secuencia en ambas direcciones:
       → Forward: Izquierda a derecha
       ← Backward: Derecha a izquierda

       VENTAJA:
       Captura contexto de ambos lados.

       Ejemplo:
       "The movie was not good"
       Forward: "was not" → negativo
       Backward: "not good" → negativo
       Ambos: CONTEXTO COMPLETO → Más robusto!

    4. DROPOUT:
       ----------
       Previene overfitting desactivando neuronas aleatoriamente.
       Similar a prediccion-temperatura!

    5. DENSE LAYER + SIGMOID:
       -------------------------
       Output: Probabilidad entre 0 y 1
       < 0.5 → Negativo
       ≥ 0.5 → Positivo

    COMPARACIÓN CON prediccion-temperatura:
    ----------------------------------------
    prediccion-temperatura:
    - Input: (batch, sequence_length, 1)  ← 1 feature (temperatura)
    - LSTM: Procesa secuencia de temperaturas
    - Output: (batch, 1)  ← Próxima temperatura

    sentiment-analysis:
    - Input: (batch, sequence_length)  ← Índices de palabras
    - Embedding: (batch, sequence_length, embedding_dim)
    - LSTM: Procesa secuencia de word vectors
    - Output: (batch, 1)  ← Probabilidad de sentimiento

    Args:
        vocab_size: Tamaño del vocabulario
        embedding_dim: Dimensión de embeddings
        embedding_matrix: Embeddings pre-entrenados (opcional)
        max_length: Longitud máxima de secuencia

    Returns:
        Modelo compilado
    """
    print("\n" + "=" * 70)
    print("🧠 CONSTRUYENDO MODELO LSTM")
    print("=" * 70)

    if embedding_dim is None:
        embedding_dim = config.EMBEDDING_DIM
    if max_length is None:
        max_length = config.MAX_SEQUENCE_LENGTH

    print(f"\n⚙️  Configuración:")
    print(f"   • Vocabulario: {vocab_size} palabras")
    print(f"   • Embedding dimension: {embedding_dim}")
    print(f"   • Longitud máxima: {max_length} palabras")
    print(f"   • LSTM units (capa 1): {config.LSTM_UNITS_1}")
    print(f"   • LSTM units (capa 2): {config.LSTM_UNITS_2}")
    print(f"   • Dropout: {config.DROPOUT_RATE}")

    # Construir modelo
    model = Sequential(name='SentimentLSTM')

    # 1. Embedding Layer
    if embedding_matrix is not None:
        print(f"\n   💡 Usando embeddings PRE-ENTRENADOS")
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            weights=[embedding_matrix],
            trainable=False,  # No entrenar embeddings (frozen)
            name='embedding'
        ))
    else:
        print(f"\n   💡 Usando embeddings ALEATORIOS (trainable)")
        model.add(Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=max_length,
            name='embedding'
        ))

    # 2. Primera capa LSTM (bidireccional)
    model.add(Bidirectional(
        LSTM(
            config.LSTM_UNITS_1,
            return_sequences=True,  # Retornar secuencia completa para siguiente LSTM
            dropout=config.DROPOUT_RATE,
            recurrent_dropout=config.RECURRENT_DROPOUT
        ),
        name='bidirectional_lstm_1'
    ))

    # 3. Segunda capa LSTM
    model.add(Bidirectional(
        LSTM(
            config.LSTM_UNITS_2,
            return_sequences=False,  # Solo retornar último output
            dropout=config.DROPOUT_RATE,
            recurrent_dropout=config.RECURRENT_DROPOUT
        ),
        name='bidirectional_lstm_2'
    ))

    # 4. Capa densa intermedia
    model.add(Dense(
        config.DENSE_UNITS,
        activation=config.ACTIVATION_HIDDEN,
        name='dense_hidden'
    ))
    model.add(Dropout(config.DROPOUT_RATE, name='dropout'))

    # 5. Capa de salida
    model.add(Dense(
        1,  # 1 neurona para clasificación binaria
        activation=config.ACTIVATION_OUTPUT,  # Sigmoid: output entre 0 y 1
        name='output'
    ))

    # Compilar modelo
    print(f"\n🔧 Compilando modelo...")
    model.compile(
        optimizer=config.OPTIMIZER,
        loss=config.LOSS,  # binary_crossentropy para clasificación binaria
        metrics=config.METRICS
    )

    print(f"✅ Modelo compilado!")

    # Mostrar arquitectura
    print(f"\n📊 ARQUITECTURA DEL MODELO:")
    print("=" * 70)
    model.summary()

    return model


# ============================================================================
# 3. ENTRENAR MODELO
# ============================================================================

def train_lstm_model(model,
                    X_train, y_train,
                    X_val, y_val,
                    epochs: int = None,
                    batch_size: int = None):
    """
    Entrena el modelo LSTM.

    📚 CALLBACKS (ya los conoces!):
    ================================

    1. EarlyStopping:
       Para el entrenamiento si no hay mejora
       (igual que en prediccion-temperatura)

    2. ReduceLROnPlateau:
       Reduce learning rate si no hay mejora
       (igual que en prediccion-temperatura)

    3. ModelCheckpoint:
       Guarda el mejor modelo durante entrenamiento

    Args:
        model: Modelo LSTM
        X_train: Secuencias de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_val: Secuencias de validación
        y_val: Etiquetas de validación
        epochs: Número de épocas
        batch_size: Tamaño de batch

    Returns:
        Historia de entrenamiento
    """
    print("\n" + "=" * 70)
    print("🚀 ENTRENANDO MODELO LSTM")
    print("=" * 70)

    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    print(f"\n⚙️  Configuración de entrenamiento:")
    print(f"   • Épocas: {epochs}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Training samples: {len(X_train)}")
    print(f"   • Validation samples: {len(X_val)}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(config.MODEL_LSTM),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\n📋 Callbacks configurados:")
    print(f"   • EarlyStopping (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"   • ReduceLROnPlateau")
    print(f"   • ModelCheckpoint")

    # Entrenar
    print(f"\n🔄 Iniciando entrenamiento...")
    print("=" * 70)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO!")
    print("=" * 70)

    # Resultados finales
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"\n📊 Resultados finales:")
    print(f"   • Training Loss: {final_train_loss:.4f}")
    print(f"   • Validation Loss: {final_val_loss:.4f}")
    print(f"   • Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   • Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

    return history


# ============================================================================
# 4. EVALUAR MODELO
# ============================================================================

def evaluate_lstm_model(model, X_test, y_test) -> Dict:
    """
    Evalúa el modelo LSTM en datos de prueba.

    Args:
        model: Modelo entrenado
        X_test: Secuencias de prueba
        y_test: Etiquetas reales

    Returns:
        Diccionario con métricas
    """
    print("\n" + "=" * 70)
    print("📊 EVALUANDO MODELO LSTM")
    print("=" * 70)

    # Evaluar
    print(f"\n🔮 Realizando predicciones...")
    results = model.evaluate(X_test, y_test, verbose=0)

    # Obtener nombres de métricas
    metric_names = model.metrics_names

    print(f"\n✅ Evaluación completada!")
    print(f"\n📈 MÉTRICAS EN TEST SET:")
    print("-" * 70)
    for name, value in zip(metric_names, results):
        if 'loss' in name:
            print(f"   • {name.capitalize()}: {value:.4f}")
        else:
            print(f"   • {name.capitalize()}: {value:.4f} ({value*100:.2f}%)")

    # Predicciones para matriz de confusión
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    # Matriz de confusión
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n📊 MATRIZ DE CONFUSIÓN:")
    print("-" * 70)
    print(f"                Pred Neg    Pred Pos")
    print(f"   Real Neg     {cm[0,0]:>6}      {cm[0,1]:>6}")
    print(f"   Real Pos     {cm[1,0]:>6}      {cm[1,1]:>6}")

    # Reporte detallado
    print(f"\n📋 REPORTE DETALLADO:")
    print("-" * 70)
    print(classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo']))

    # Crear diccionario de métricas
    metrics = {name: value for name, value in zip(metric_names, results)}
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


# ============================================================================
# 5. GUARDAR Y CARGAR
# ============================================================================

def save_tokenizer(tokenizer, filepath):
    """Guarda tokenizer."""
    joblib.dump(tokenizer, filepath)
    print(f"\n💾 Tokenizer guardado: {filepath.name}")


def load_tokenizer(filepath):
    """Carga tokenizer."""
    tokenizer = joblib.load(filepath)
    print(f"\n📂 Tokenizer cargado: {filepath.name}")
    return tokenizer


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Construcción del modelo LSTM
    """
    print("\n" + "🧠" * 35)
    print(" DEMO: LSTM MODEL FOR SENTIMENT ANALYSIS")
    print("🧠" * 35)

    # Construir modelo demo
    print(f"\n💡 Construyendo modelo demo...")

    model = build_lstm_model(
        vocab_size=10000,  # 10k palabras
        embedding_dim=100,  # 100 dimensiones
        max_length=200  # 200 palabras max
    )

    print(f"\n✅ Modelo construido exitosamente!")
    print(f"\n💡 Este modelo está listo para entrenar con datos reales.")
    print(f"   Ejecuta main.py para el pipeline completo.")

    # Información adicional
    print(f"\n📚 CONCEPTOS CLAVE:")
    print("=" * 70)
    print(f"\n1. EMBEDDING LAYER:")
    print(f"   Convierte índices de palabras a vectores densos")
    print(f"   palabra_idx → vector de {config.EMBEDDING_DIM} dimensiones")

    print(f"\n2. BIDIRECTIONAL LSTM:")
    print(f"   Lee la secuencia en ambas direcciones")
    print(f"   → Forward + ← Backward = Mejor contexto")

    print(f"\n3. DROPOUT:")
    print(f"   Previene overfitting ({config.DROPOUT_RATE*100}% de neuronas desactivadas)")

    print(f"\n4. SIGMOID OUTPUT:")
    print(f"   Probabilidad entre 0 y 1")
    print(f"   < 0.5 → Negativo, ≥ 0.5 → Positivo")
