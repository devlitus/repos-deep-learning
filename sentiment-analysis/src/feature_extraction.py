"""
Feature Extraction - Análisis de Sentimientos
==============================================

Este módulo convierte texto en representaciones numéricas (features).

🔑 CONCEPTO CLAVE: Los modelos de ML solo entienden números
------------------------------------------------------------
El texto "I love this movie" no se puede usar directamente.
Necesitamos convertirlo a vectores numéricos.

MÉTODOS DE CONVERSIÓN:

1. BAG OF WORDS (Bolsa de Palabras): Conteo simple
   Vocabulario: ["love", "hate", "movie", "good", "bad"]
   "I love this movie" → [1, 0, 1, 0, 0]  (love=1, movie=1, resto=0)

2. TF-IDF: Bag of Words ponderado por importancia
   Palabras raras → mayor peso
   Palabras comunes → menor peso

3. WORD EMBEDDINGS: Representación vectorial densa
   Cada palabra → vector de N dimensiones
   Palabras similares → vectores similares
   "excellent" y "great" estarán cerca en el espacio vectorial
"""

import sys
from pathlib import Path

# Agregar la raíz del proyecto al PATH (solo cuando se ejecuta como script)
try:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except NameError:
    # __file__ no está definido en notebooks de Jupyter
    pass

import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import config

# ============================================================================
# 1. BAG OF WORDS (Bolsa de Palabras)
# ============================================================================

def create_bow_features(texts: List[str],
                       max_features: int = None,
                       vectorizer=None) -> Tuple:
    """
    Convierte textos a vectores Bag of Words.

    📊 BAG OF WORDS:
    Representa cada documento como vector de conteos de palabras.
    Ignora el orden y la gramática, solo cuenta frecuencias.

    EJEMPLO:
    --------
    Vocabulario: ["love", "hate", "movie", "excellent", "terrible"]

    Documento 1: "I love this movie"
    Vector: [1, 0, 1, 0, 0]  (love=1, movie=1)

    Documento 2: "Excellent movie, love it"
    Vector: [1, 0, 1, 1, 0]  (love=1, movie=1, excellent=1)

    Documento 3: "Terrible movie, I hate it"
    Vector: [0, 1, 1, 0, 1]  (hate=1, movie=1, terrible=1)

    VENTAJAS:
    ✅ Simple y rápido
    ✅ Funciona bien para clasificación de texto

    DESVENTAJAS:
    ❌ Ignora orden de palabras ("not good" = "good not")
    ❌ Ignora similitud semántica ("excellent" ≠ "great")
    ❌ Vectores muy grandes y dispersos (sparse)

    Args:
        texts: Lista de textos preprocesados
        max_features: Número máximo de palabras en vocabulario
        vectorizer: Vectorizer pre-entrenado (para predicción)

    Returns:
        (features, vectorizer)
        - features: Matriz sparse de forma (n_documents, n_features)
        - vectorizer: Vectorizer entrenado (para guardar)
    """
    print("\n" + "=" * 70)
    print("📊 CREANDO FEATURES: BAG OF WORDS")
    print("=" * 70)

    if max_features is None:
        max_features = config.TFIDF_MAX_FEATURES

    if vectorizer is None:
        # Entrenar nuevo vectorizer
        print(f"\n⚙️  Configuración:")
        print(f"   • Máximo de features: {max_features}")
        print(f"   • Min document frequency: 2 (palabra debe aparecer en 2+ docs)")

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=2,  # Ignorar palabras que aparecen en < 2 documentos
            ngram_range=(1, 2)  # Unigramas y bigramas
        )

        print(f"\n🔄 Entrenando vectorizer...")
        features = vectorizer.fit_transform(texts)

        print(f"\n✅ Vectorizer entrenado!")
        print(f"   • Vocabulario: {len(vectorizer.vocabulary_)} palabras")
        print(f"   • Forma de features: {features.shape}")
        print(f"   • Sparsity: {(1 - features.nnz / (features.shape[0] * features.shape[1])) * 100:.2f}%")

    else:
        # Usar vectorizer pre-entrenado
        print(f"\n🔄 Transformando con vectorizer pre-entrenado...")
        features = vectorizer.transform(texts)
        print(f"✅ Features creadas: {features.shape}")

    return features, vectorizer


# ============================================================================
# 2. TF-IDF (Term Frequency - Inverse Document Frequency)
# ============================================================================

def create_tfidf_features(texts: List[str],
                         max_features: int = None,
                         vectorizer=None) -> Tuple:
    """
    Convierte textos a vectores TF-IDF.

    📈 TF-IDF:
    Mejora Bag of Words ponderando palabras por importancia.

    TF-IDF = TF × IDF

    1. TF (Term Frequency):
       Qué tan frecuente es la palabra en ESTE documento
       TF(word, doc) = count(word in doc) / total_words(doc)

    2. IDF (Inverse Document Frequency):
       Qué tan rara es la palabra en TODOS los documentos
       IDF(word) = log(total_docs / docs_containing_word)

    INTUICIÓN:
    ----------
    Palabra común ("movie" aparece en todas las reviews):
    - TF: alta (aparece mucho)
    - IDF: baja (aparece en todos los docs)
    - TF-IDF: BAJA → No es discriminativa

    Palabra rara ("masterpiece" aparece en pocas reviews):
    - TF: media (aparece algunas veces)
    - IDF: alta (aparece en pocos docs)
    - TF-IDF: ALTA → Muy discriminativa!

    EJEMPLO NUMÉRICO:
    -----------------
    Dataset: 1000 reviews

    Palabra "movie":
    - Aparece en 950 reviews (muy común)
    - IDF = log(1000/950) = 0.05 (bajo)

    Palabra "masterpiece":
    - Aparece en 10 reviews (rara)
    - IDF = log(1000/10) = 2.0 (alto)

    Review: "This movie is a masterpiece"
    TF-IDF:
    - "movie": 0.33 × 0.05 = 0.017 (bajo)
    - "masterpiece": 0.33 × 2.0 = 0.66 (alto) ← MÁS IMPORTANTE!

    VENTAJAS sobre Bag of Words:
    ✅ Palabras raras tienen más peso
    ✅ Palabras comunes tienen menos peso
    ✅ Mejor para clasificación

    DESVENTAJAS:
    ❌ Sigue ignorando orden
    ❌ Sigue ignorando semántica

    Args:
        texts: Lista de textos preprocesados
        max_features: Número máximo de palabras en vocabulario
        vectorizer: Vectorizer pre-entrenado (para predicción)

    Returns:
        (features, vectorizer)
        - features: Matriz sparse de TF-IDF scores
        - vectorizer: Vectorizer entrenado
    """
    print("\n" + "=" * 70)
    print("📈 CREANDO FEATURES: TF-IDF")
    print("=" * 70)

    if max_features is None:
        max_features = config.TFIDF_MAX_FEATURES

    if vectorizer is None:
        # Entrenar nuevo vectorizer
        print(f"\n⚙️  Configuración:")
        print(f"   • Máximo de features: {max_features}")
        print(f"   • Min document frequency: 2")
        print(f"   • N-gramas: (1, 2) - unigramas y bigramas")
        print(f"   • Sublinear TF: True (usa log(TF) en lugar de TF)")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=2,
            max_df=0.8,  # Ignorar palabras en >80% de docs (demasiado comunes)
            ngram_range=(1, 2),  # Unigramas ("excellent") y bigramas ("not good")
            sublinear_tf=True  # Usa log(TF) para reducir impacto de TF muy altos
        )

        print(f"\n🔄 Entrenando TF-IDF vectorizer...")
        features = vectorizer.fit_transform(texts)

        print(f"\n✅ Vectorizer entrenado!")
        print(f"   • Vocabulario: {len(vectorizer.vocabulary_)} features")
        print(f"   • Forma: {features.shape}")
        print(f"   • Sparsity: {(1 - features.nnz / (features.shape[0] * features.shape[1])) * 100:.2f}%")

        # Mostrar palabras con IDF más alto (más discriminativas)
        feature_names = vectorizer.get_feature_names_out()
        idf_scores = vectorizer.idf_
        top_indices = np.argsort(idf_scores)[-10:][::-1]

        print(f"\n🔝 Top 10 palabras más discriminativas (IDF alto):")
        for idx in top_indices:
            print(f"   • '{feature_names[idx]}': {idf_scores[idx]:.3f}")

    else:
        # Usar vectorizer pre-entrenado
        print(f"\n🔄 Transformando con vectorizer pre-entrenado...")
        features = vectorizer.transform(texts)
        print(f"✅ Features creadas: {features.shape}")

    return features, vectorizer


# ============================================================================
# 3. WORD EMBEDDINGS (Representación vectorial densa)
# ============================================================================

def create_embedding_matrix(word_index: dict,
                           embedding_dim: int = None,
                           use_pretrained: bool = False) -> np.ndarray:
    """
    Crea matriz de embeddings para vocabulario.

    🧠 WORD EMBEDDINGS:
    Representación vectorial densa de palabras en espacio continuo.
    A diferencia de TF-IDF (sparse), embeddings son densos.

    CONCEPTO:
    ---------
    Cada palabra → vector de N dimensiones (típicamente 50-300)

    Ejemplo con 3 dimensiones (real: 100-300):
    "excellent" → [0.8, 0.2, 0.1]
    "great" → [0.75, 0.25, 0.15]  ← Similar a "excellent"
    "terrible" → [-0.7, 0.1, -0.2]  ← Opuesto a "excellent"

    PROPIEDADES MATEMÁTICAS:
    ------------------------
    1. Similitud semántica:
       similar_words("king") = ["queen", "monarch", "prince"]

    2. Analogías:
       king - man + woman ≈ queen
       Paris - France + Italy ≈ Rome

    3. Distancia en espacio vectorial:
       distance("excellent", "great") < distance("excellent", "terrible")

    OPCIONES:
    ---------
    1. Embeddings aleatorios (trainable):
       - Se entrenan durante el entrenamiento del modelo
       - Específicos para tu dataset
       - Requieren más datos

    2. Embeddings pre-entrenados (frozen o fine-tuned):
       - GloVe (Global Vectors): Entrenado en Wikipedia + Gigaword
       - Word2Vec: Entrenado en Google News
       - FastText: Subword embeddings
       - Capturan conocimiento lingüístico general
       - Mejor con pocos datos

    COMPARACIÓN CON TF-IDF:
    -----------------------
    TF-IDF:
    - Vector sparse: [0, 0, 0.5, 0, 0, 0.8, 0, ...]  (mayoría ceros)
    - Dimensión = vocabulario (5000-50000)
    - No captura semántica

    Embeddings:
    - Vector denso: [0.23, -0.45, 0.12, 0.67, ...]  (todos no-cero)
    - Dimensión fija: 50-300
    - Captura semántica

    Args:
        word_index: Diccionario {palabra: índice}
        embedding_dim: Dimensión del embedding
        use_pretrained: Usar GloVe pre-entrenado

    Returns:
        Matriz de embeddings de forma (vocab_size, embedding_dim)
    """
    print("\n" + "=" * 70)
    print("🧠 CREANDO EMBEDDING MATRIX")
    print("=" * 70)

    if embedding_dim is None:
        embedding_dim = config.EMBEDDING_DIM

    vocab_size = len(word_index) + 1  # +1 para padding (índice 0)

    if not use_pretrained:
        # Embeddings aleatorios (se entrenarán durante el training)
        print(f"\n⚙️  Configuración: Embeddings ALEATORIOS (trainable)")
        print(f"   • Vocabulario: {vocab_size} palabras")
        print(f"   • Dimensión: {embedding_dim}")
        print(f"   • Inicialización: Uniforme [-0.05, 0.05]")
        print(f"\n💡 Estos embeddings se entrenarán con tu modelo")

        embedding_matrix = np.random.uniform(
            low=-0.05,
            high=0.05,
            size=(vocab_size, embedding_dim)
        )

        # Padding vector (índice 0) debe ser ceros
        embedding_matrix[0] = np.zeros(embedding_dim)

        print(f"\n✅ Embedding matrix creada: {embedding_matrix.shape}")

    else:
        # Cargar embeddings pre-entrenados (GloVe)
        print(f"\n⚙️  Configuración: Embeddings PRE-ENTRENADOS (GloVe)")
        print(f"   • Vocabulario: {vocab_size} palabras")
        print(f"   • Dimensión: {embedding_dim}")
        print(f"\n💡 Estos embeddings ya conocen semántica de millones de textos")

        embedding_matrix = load_glove_embeddings(word_index, embedding_dim)

        print(f"\n✅ Embedding matrix creada: {embedding_matrix.shape}")

    return embedding_matrix


def load_glove_embeddings(word_index: dict,
                         embedding_dim: int) -> np.ndarray:
    """
    Carga embeddings pre-entrenados GloVe.

    📚 GLOVE (Global Vectors for Word Representation):
    Embeddings entrenados en:
    - Wikipedia (6B tokens)
    - Gigaword (42B tokens)
    - Common Crawl (840B tokens)

    Disponibles en: https://nlp.stanford.edu/projects/glove/

    Dimensiones disponibles: 50, 100, 200, 300

    Args:
        word_index: Diccionario {palabra: índice}
        embedding_dim: Dimensión (50, 100, 200, 300)

    Returns:
        Matriz de embeddings
    """
    print(f"\n📥 Cargando GloVe embeddings ({embedding_dim}d)...")

    glove_path = config.BASE_DIR / f'data/glove.6B.{embedding_dim}d.txt'

    if not glove_path.exists():
        print(f"\n⚠️  Archivo GloVe no encontrado: {glove_path}")
        print(f"\n💡 Descarga GloVe desde:")
        print(f"   https://nlp.stanford.edu/projects/glove/")
        print(f"\n   Usa embeddings aleatorios por ahora...")
        return create_embedding_matrix(word_index, embedding_dim, use_pretrained=False)

    # Cargar embeddings de archivo
    embeddings_index = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f"✅ Cargados {len(embeddings_index)} word vectors")

    # Crear matriz
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    found = 0
    for word, idx in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            found += 1

    print(f"✅ {found}/{len(word_index)} palabras encontradas en GloVe")
    print(f"   {len(word_index) - found} palabras sin embedding (usarán ceros)")

    return embedding_matrix


# ============================================================================
# UTILIDADES
# ============================================================================

def save_vectorizer(vectorizer, filepath):
    """
    Guarda vectorizer entrenado.

    Args:
        vectorizer: TfidfVectorizer o CountVectorizer
        filepath: Ruta donde guardar
    """
    joblib.dump(vectorizer, filepath)
    print(f"\n💾 Vectorizer guardado: {filepath.name}")


def load_vectorizer(filepath):
    """
    Carga vectorizer guardado.

    Args:
        filepath: Ruta del vectorizer

    Returns:
        Vectorizer cargado
    """
    vectorizer = joblib.load(filepath)
    print(f"\n📂 Vectorizer cargado: {filepath.name}")
    return vectorizer


# ============================================================================
# DEMO Y EJEMPLOS
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Feature extraction comparando métodos
    """
    print("\n" + "🔢" * 35)
    print(" DEMO: FEATURE EXTRACTION")
    print("🔢" * 35)

    # Corpus de ejemplo
    corpus = [
        "I love this movie excellent acting",  # Positivo
        "Great film loved the story",  # Positivo
        "Terrible movie waste of time",  # Negativo
        "Awful acting and boring plot",  # Negativo
        "Masterpiece brilliant performance"  # Positivo
    ]

    print(f"\n📚 Corpus de ejemplo ({len(corpus)} documentos):")
    print("-" * 70)
    for i, doc in enumerate(corpus, 1):
        print(f"{i}. \"{doc}\"")

    # 1. Bag of Words
    print(f"\n\n1️⃣  BAG OF WORDS")
    print("=" * 70)
    bow_features, bow_vectorizer = create_bow_features(corpus, max_features=20)

    vocab = bow_vectorizer.get_feature_names_out()
    print(f"\n📖 Vocabulario ({len(vocab)} palabras):")
    print(f"   {list(vocab)}")

    print(f"\n📊 Features del primer documento:")
    print(f"   Documento: \"{corpus[0]}\"")
    print(f"   Vector BoW: {bow_features[0].toarray()[0]}")

    # 2. TF-IDF
    print(f"\n\n2️⃣  TF-IDF")
    print("=" * 70)
    tfidf_features, tfidf_vectorizer = create_tfidf_features(corpus, max_features=20)

    print(f"\n📊 Features del primer documento:")
    print(f"   Documento: \"{corpus[0]}\"")
    print(f"   Vector TF-IDF: {tfidf_features[0].toarray()[0][:10]}... (primeros 10)")

    # 3. Comparación BoW vs TF-IDF
    print(f"\n\n3️⃣  COMPARACIÓN: BoW vs TF-IDF")
    print("=" * 70)

    feature_names = tfidf_vectorizer.get_feature_names_out()
    doc_idx = 0  # Primer documento

    bow_vector = bow_features[doc_idx].toarray()[0]
    tfidf_vector = tfidf_features[doc_idx].toarray()[0]

    print(f"\nDocumento: \"{corpus[doc_idx]}\"")
    print(f"\n{'Palabra':<15} {'BoW':<10} {'TF-IDF':<10}")
    print("-" * 35)

    # Mostrar solo palabras presentes
    for i, word in enumerate(feature_names):
        if bow_vector[i] > 0:
            print(f"{word:<15} {bow_vector[i]:<10.0f} {tfidf_vector[i]:<10.3f}")

    # 4. Word Embeddings
    print(f"\n\n4️⃣  WORD EMBEDDINGS (Demo conceptual)")
    print("=" * 70)

    # Crear vocabulario simple
    word_index = {word: i+1 for i, word in enumerate(vocab)}

    embedding_matrix = create_embedding_matrix(
        word_index,
        embedding_dim=50,
        use_pretrained=False
    )

    print(f"\n📊 Ejemplo de embedding para palabra 'love' (índice {word_index.get('love', 'N/A')}):")
    if 'love' in word_index:
        love_embedding = embedding_matrix[word_index['love']]
        print(f"   Vector (50 dimensiones): {love_embedding[:10]}... (primeras 10)")
        print(f"   Estos 50 números capturan el 'significado' de 'love'")

    print("\n\n✅ Demo completada!")

    # Resumen comparativo
    print(f"\n\n" + "=" * 70)
    print("📊 RESUMEN COMPARATIVO")
    print("=" * 70)

    print(f"\n{'Método':<20} {'Dimensión':<15} {'Tipo':<10} {'Semántica'}")
    print("-" * 60)
    print(f"{'Bag of Words':<20} {bow_features.shape[1]:<15} {'Sparse':<10} {'No'}")
    print(f"{'TF-IDF':<20} {tfidf_features.shape[1]:<15} {'Sparse':<10} {'No'}")
    print(f"{'Word Embeddings':<20} {embedding_matrix.shape[1]:<15} {'Denso':<10} {'Sí'}")

    print(f"\n💡 Cuándo usar cada método:")
    print(f"   • BoW/TF-IDF + ML clásico: Rápido, interpretable, buenos resultados")
    print(f"   • Embeddings + Deep Learning: Mejor con muchos datos, captura semántica")
