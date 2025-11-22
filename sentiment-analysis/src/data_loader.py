"""
Data Loader - Análisis de Sentimientos
=========================================

Este módulo carga y explora datos de texto para análisis de sentimientos.

CONCEPTOS NUEVOS QUE APRENDISTE:
--------------------------------
1. TOKENIZACIÓN: Dividir texto en palabras (tokens)
   "Me encanta esta película" → ["Me", "encanta", "esta", "película"]

2. DATASETS DE TEXTO: A diferencia de datos tabulares (CSV), trabajamos con:
   - Secuencias de texto variable (reviews de 10 o 500 palabras)
   - Etiquetas de sentimiento (positivo/negativo)

3. VOCABULARIO: Conjunto único de palabras en el dataset
   Ejemplo: 10,000 palabras más frecuentes

4. PADDING: Hacer todas las secuencias del mismo tamaño
   "Buena película" → [42, 789, 0, 0, 0] (rellenado con ceros)
   "Excelente actuación increíble" → [12, 456, 923, 0, 0]
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import config

# ============================================================================
# OPCIÓN 1: CARGAR DATASET IMDB (incluido en Keras)
# ============================================================================

def load_imdb_data(num_words: int = None) -> Tuple:
    """
    Carga el dataset IMDB de reviews de películas.

    🎬 DATASET IMDB:
    - 50,000 reviews de películas
    - 25,000 para entrenamiento, 25,000 para prueba
    - Clasificación binaria: 0 = negativo, 1 = positivo
    - Ya viene pre-tokenizado (palabras → números)

    📊 EJEMPLO DE DATOS:
    Review original: "This movie was excellent and the acting superb!"
    Review tokenizado: [1, 23, 4, 567, 3, 1, 234, 89]
    Sentimiento: 1 (positivo)

    Args:
        num_words: Usar solo las top N palabras más frecuentes
                   Palabras raras se reemplazan por <UNK> (unknown)

    Returns:
        (X_train, y_train), (X_test, y_test)
        - X_train: Lista de reviews (cada review es lista de índices de palabras)
        - y_train: Lista de sentimientos (0 o 1)
    """
    print("=" * 70)
    print("📥 CARGANDO DATASET IMDB")
    print("=" * 70)

    if num_words is None:
        num_words = config.NUM_WORDS

    print(f"\n⚙️  Configuración:")
    print(f"   • Vocabulario: top {num_words} palabras más frecuentes")
    print(f"   • Palabras raras → <UNK> (unknown)")

    try:
        from tensorflow.keras.datasets import imdb

        # Cargar datos pre-tokenizados
        (X_train, y_train), (X_test, y_test) = imdb.load_data(
            num_words=num_words
        )

        print(f"\n✅ Datos cargados exitosamente!")
        print(f"\n📊 Tamaño del dataset:")
        print(f"   • Entrenamiento: {len(X_train)} reviews")
        print(f"   • Prueba: {len(X_test)} reviews")
        print(f"   • Total: {len(X_train) + len(X_test)} reviews")

        return (X_train, y_train), (X_test, y_test)

    except Exception as e:
        print(f"\n❌ Error al cargar datos: {e}")
        raise


def explore_imdb_data(X_train: List, y_train: np.ndarray,
                     X_test: List, y_test: np.ndarray) -> None:
    """
    Explora las características del dataset IMDB.

    Similar a explore_data() de otros proyectos, pero adaptado para texto.
    """
    print("\n" + "=" * 70)
    print("🔍 EXPLORACIÓN DE DATOS")
    print("=" * 70)

    # 1. Distribución de clases
    print("\n1️⃣  DISTRIBUCIÓN DE SENTIMIENTOS")
    print("-" * 70)

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    print("\n📊 Entrenamiento:")
    for sentiment, count in zip(unique_train, counts_train):
        label = "Positivo" if sentiment == 1 else "Negativo"
        percentage = (count / len(y_train)) * 100
        print(f"   • {label}: {count} reviews ({percentage:.1f}%)")

    print("\n📊 Prueba:")
    for sentiment, count in zip(unique_test, counts_test):
        label = "Positivo" if sentiment == 1 else "Negativo"
        percentage = (count / len(y_test)) * 100
        print(f"   • {label}: {count} reviews ({percentage:.1f}%)")

    # 2. Longitud de las reviews
    print("\n\n2️⃣  LONGITUD DE REVIEWS (en palabras)")
    print("-" * 70)

    lengths_train = [len(review) for review in X_train]
    lengths_test = [len(review) for review in X_test]

    print("\n📏 Estadísticas de longitud (entrenamiento):")
    print(f"   • Mínima: {min(lengths_train)} palabras")
    print(f"   • Máxima: {max(lengths_train)} palabras")
    print(f"   • Media: {np.mean(lengths_train):.1f} palabras")
    print(f"   • Mediana: {np.median(lengths_train):.1f} palabras")
    print(f"   • Desviación estándar: {np.std(lengths_train):.1f}")

    # 3. Ejemplo de review tokenizado
    print("\n\n3️⃣  EJEMPLO DE REVIEW TOKENIZADO")
    print("-" * 70)

    example_review = X_train[0]
    example_sentiment = y_train[0]

    print(f"\n🎬 Review #{0}:")
    print(f"   • Longitud: {len(example_review)} palabras")
    print(f"   • Sentimiento: {'Positivo ✅' if example_sentiment == 1 else 'Negativo ❌'}")
    print(f"   • Primeros 20 tokens: {example_review[:20]}")
    print(f"\n   💡 Cada número representa una palabra:")
    print(f"      Ejemplo: 1='the', 2='and', 4='to', 14='movie', etc.")

    # 4. Palabras únicas
    print("\n\n4️⃣  VOCABULARIO")
    print("-" * 70)

    # Encontrar el índice de palabra más alto (tamaño del vocabulario)
    max_index_train = max([max(review) for review in X_train])
    max_index_test = max([max(review) for review in X_test])
    vocab_size = max(max_index_train, max_index_test) + 1

    print(f"\n📚 Tamaño del vocabulario: {vocab_size} palabras únicas")
    print(f"   • Cada palabra tiene un índice único (1 a {vocab_size})")
    print(f"   • Índice 0 reservado para padding (relleno)")
    print(f"   • Índice 1 reservado para <START>")
    print(f"   • Índice 2 reservado para <UNK> (unknown)")


def get_word_index() -> Dict[str, int]:
    """
    Obtiene el diccionario de palabras → índices del dataset IMDB.

    📖 WORD INDEX:
    - Mapea cada palabra a su índice numérico
    - Ejemplo: {"the": 1, "and": 2, "movie": 14, "excellent": 89}
    - Permite convertir entre texto y números

    Returns:
        Diccionario {palabra: índice}
    """
    print("\n📖 Cargando diccionario de palabras...")

    from tensorflow.keras.datasets import imdb
    word_index = imdb.get_word_index()

    print(f"✅ Diccionario cargado: {len(word_index)} palabras")

    return word_index


def decode_review(encoded_review: List[int], word_index: Dict[str, int]) -> str:
    """
    Convierte una review tokenizada (números) de vuelta a texto.

    🔄 DECODIFICACIÓN:
    [1, 14, 22, 16] → "the movie was excellent"

    Esto es útil para inspeccionar los datos y entender qué estamos procesando.

    Args:
        encoded_review: Lista de índices de palabras
        word_index: Diccionario palabra → índice

    Returns:
        Review en texto legible
    """
    # Invertir el diccionario: índice → palabra
    reverse_word_index = {value: key for key, value in word_index.items()}

    # Reservar índices especiales
    reverse_word_index[0] = '<PAD>'  # Padding
    reverse_word_index[1] = '<START>'  # Inicio
    reverse_word_index[2] = '<UNK>'  # Unknown
    reverse_word_index[3] = '<UNUSED>'

    # Decodificar
    decoded = ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

    return decoded


# ============================================================================
# OPCIÓN 2: CARGAR DATASET PERSONALIZADO (CSV)
# ============================================================================

def load_custom_dataset(filepath: Path) -> pd.DataFrame:
    """
    Carga un dataset personalizado de sentimientos desde CSV.

    📄 FORMATO ESPERADO:
    - Columna 'text': El texto de la review
    - Columna 'sentiment': 0 (negativo) o 1 (positivo)

    Ejemplo:
    text,sentiment
    "This movie is amazing!",1
    "Terrible acting and plot",0

    Args:
        filepath: Ruta al archivo CSV

    Returns:
        DataFrame con columnas [text, sentiment]
    """
    print(f"\n📂 Cargando dataset desde: {filepath.name}")

    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()  # Limpiar nombres de columnas

        print(f"✅ Dataset cargado: {len(df)} reviews")

        # Verificar columnas requeridas
        required_cols = [config.TEXT_COLUMN, config.LABEL_COLUMN]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Columnas faltantes: {missing_cols}")

        return df

    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        raise


def explore_custom_dataset(df: pd.DataFrame) -> None:
    """
    Explora un dataset personalizado de sentimientos.
    """
    print("\n" + "=" * 70)
    print("🔍 EXPLORACIÓN DE DATOS PERSONALIZADOS")
    print("=" * 70)

    print(f"\n📊 Tamaño del dataset: {len(df)} reviews")
    print(f"\n📋 Columnas: {list(df.columns)}")

    # Distribución de sentimientos
    print("\n1️⃣  DISTRIBUCIÓN DE SENTIMIENTOS")
    print("-" * 70)
    sentiment_counts = df[config.LABEL_COLUMN].value_counts()
    for sentiment, count in sentiment_counts.items():
        label = "Positivo" if sentiment == 1 else "Negativo"
        percentage = (count / len(df)) * 100
        print(f"   • {label}: {count} reviews ({percentage:.1f}%)")

    # Ejemplos de reviews
    print("\n2️⃣  EJEMPLOS DE REVIEWS")
    print("-" * 70)

    for i in range(min(3, len(df))):
        text = df[config.TEXT_COLUMN].iloc[i]
        sentiment = df[config.LABEL_COLUMN].iloc[i]

        # Truncar texto si es muy largo
        display_text = text[:100] + "..." if len(text) > 100 else text

        print(f"\n📝 Review #{i+1}:")
        print(f"   Texto: {display_text}")
        print(f"   Sentimiento: {'Positivo ✅' if sentiment == 1 else 'Negativo ❌'}")

    # Valores faltantes
    print("\n3️⃣  VALORES FALTANTES")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   ✅ No hay valores faltantes")
    else:
        print(missing[missing > 0])


# ============================================================================
# PADDING: HACER TODAS LAS SECUENCIAS DEL MISMO TAMAÑO
# ============================================================================

def pad_sequences(sequences: List[List[int]],
                 maxlen: int = None) -> np.ndarray:
    """
    Aplica padding a las secuencias para que todas tengan la misma longitud.

    🔧 PADDING (Relleno):
    Los modelos de ML necesitan inputs del mismo tamaño, pero las reviews
    tienen longitudes variables. Padding soluciona esto.

    ANTES (longitudes variables):
    Review 1: [14, 22, 16, 43, 530]  (5 palabras)
    Review 2: [14, 22]  (2 palabras)
    Review 3: [11, 52, 23, 89, 12, 45, 67]  (7 palabras)

    DESPUÉS (todas de longitud 7):
    Review 1: [0, 0, 14, 22, 16, 43, 530]  (rellenado al inicio)
    Review 2: [0, 0, 0, 0, 0, 14, 22]  (rellenado al inicio)
    Review 3: [11, 52, 23, 89, 12, 45, 67]  (sin cambios)

    Args:
        sequences: Lista de secuencias de longitud variable
        maxlen: Longitud objetivo (se truncan secuencias más largas)

    Returns:
        Array numpy de forma (num_sequences, maxlen)
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad

    if maxlen is None:
        maxlen = config.MAX_SEQUENCE_LENGTH

    print(f"\n🔧 Aplicando padding...")
    print(f"   • Longitud objetivo: {maxlen} palabras")
    print(f"   • Secuencias más cortas → rellenadas con 0s")
    print(f"   • Secuencias más largas → truncadas")

    padded = keras_pad(
        sequences,
        maxlen=maxlen,
        padding='pre',  # Relleno al inicio
        truncating='post'  # Truncar al final
    )

    print(f"✅ Padding completado: forma {padded.shape}")

    return padded


if __name__ == "__main__":
    """
    Demo: Cargar y explorar el dataset IMDB
    """
    print("\n" + "🎬" * 35)
    print(" DEMO: DATA LOADER - ANÁLISIS DE SENTIMIENTOS")
    print("🎬" * 35)

    # Cargar datos
    (X_train, y_train), (X_test, y_test) = load_imdb_data()

    # Explorar datos
    explore_imdb_data(X_train, y_train, X_test, y_test)

    # Obtener diccionario de palabras
    word_index = get_word_index()

    # Decodificar una review de ejemplo
    print("\n\n5️⃣  DECODIFICANDO REVIEW DE EJEMPLO")
    print("-" * 70)
    example_review = X_train[0]
    decoded_text = decode_review(example_review, word_index)
    print(f"\n🎬 Review original (tokenizada):")
    print(f"   {example_review[:30]}...")
    print(f"\n📝 Review decodificada (texto):")
    print(f"   {decoded_text[:200]}...")

    # Aplicar padding
    print("\n\n6️⃣  APLICANDO PADDING")
    print("-" * 70)
    X_train_padded = pad_sequences(X_train[:5])  # Solo 5 ejemplos para demo
    print(f"\n📊 Forma antes: lista de listas de longitud variable")
    print(f"📊 Forma después: {X_train_padded.shape}")

    print("\n✅ Demo completada!")
