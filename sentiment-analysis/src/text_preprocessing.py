"""
Text Preprocessing - Análisis de Sentimientos
==============================================

Este módulo contiene funciones para limpiar y preprocesar texto.

CONCEPTOS NUEVOS QUE APRENDERÁS:
---------------------------------

1. LIMPIEZA DE TEXTO:
   - Remover HTML tags: "<p>Excelente</p>" → "Excelente"
   - Remover puntuación: "¡Increíble!" → "Increíble"
   - Convertir a minúsculas: "BUENO" → "bueno"
   - Remover números: "5 estrellas" → "estrellas"

2. STOPWORDS (Palabras vacías):
   Palabras muy comunes que no aportan significado:
   - Inglés: "the", "a", "an", "is", "are", "was", "were"
   - Español: "el", "la", "de", "que", "y", "en"

   Ejemplo:
   ANTES: "The movie was really excellent"
   DESPUÉS: "movie really excellent"  (removidas: the, was)

3. STEMMING (Derivación):
   Reduce palabras a su raíz (stem) de forma agresiva:
   - "running", "runs", "ran" → "run"
   - "better", "best" → "bet"
   - "movies" → "movi"

   ⚠️ A veces produce palabras que no existen ("movi")

4. LEMMATIZATION (Lematización):
   Reduce palabras a su forma base (lemma) usando diccionario:
   - "running", "runs", "ran" → "run"
   - "better" → "good"
   - "movies" → "movie"

   ✅ Siempre produce palabras válidas (más lento que stemming)

5. TOKENIZACIÓN:
   Dividir texto en palabras individuales:
   "I love this movie" → ["I", "love", "this", "movie"]
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

import re
import string
from typing import List, Union
from bs4 import BeautifulSoup
import config

# ============================================================================
# INICIALIZACIÓN DE HERRAMIENTAS NLP
# ============================================================================

def download_nltk_resources():
    """
    Descarga recursos necesarios de NLTK.

    NLTK requiere descargar recursos adicionales:
    - punkt: Tokenizador de oraciones y palabras
    - stopwords: Listas de palabras vacías en múltiples idiomas
    - wordnet: Diccionario léxico para lemmatization
    - averaged_perceptron_tagger: Etiquetador de partes del discurso (POS)
    """
    import nltk

    resources = [
        'punkt',  # Tokenización
        'stopwords',  # Palabras vacías
        'wordnet',  # Lemmatization
        'averaged_perceptron_tagger',  # POS tagging
        'omw-1.4'  # Open Multilingual Wordnet
    ]

    print("📦 Descargando recursos de NLTK...")
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            print(f"   ✅ {resource}")
        except Exception as e:
            print(f"   ⚠️  {resource}: {e}")


# Importar después de descargar recursos
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer
    from nltk.stem import WordNetLemmatizer

    # Descargar recursos si no existen
    try:
        stopwords.words('english')
    except LookupError:
        download_nltk_resources()

except ImportError as e:
    print(f"⚠️  Error al importar NLTK: {e}")
    print("   Instala con: pip install nltk")


# ============================================================================
# 1. LIMPIEZA BÁSICA DE TEXTO
# ============================================================================

def remove_html_tags(text: str) -> str:
    """
    Remueve tags HTML del texto.

    📄 HTML en reviews:
    Muchos datasets de reviews vienen de sitios web y contienen HTML:
    "<p>This movie is <b>excellent</b>!</p>"

    Args:
        text: Texto con posible HTML

    Returns:
        Texto sin HTML

    Ejemplo:
        >>> remove_html_tags("<p>Great movie!</p>")
        "Great movie!"
    """
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


def remove_urls(text: str) -> str:
    """
    Remueve URLs del texto.

    🔗 URLs no aportan sentimiento:
    "Check out http://example.com, great movie!"
    → "Check out , great movie!"

    Args:
        text: Texto con posibles URLs

    Returns:
        Texto sin URLs
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_punctuation(text: str) -> str:
    """
    Remueve signos de puntuación.

    🔤 Puntuación:
    En muchos casos, la puntuación no aporta al análisis de sentimientos.
    "Amazing!" → "Amazing"
    "Great, but..." → "Great but"

    ⚠️ CUIDADO: A veces la puntuación SÍ importa:
    "Good?" (duda) vs "Good!" (afirmación)

    Args:
        text: Texto con puntuación

    Returns:
        Texto sin puntuación
    """
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def remove_numbers(text: str) -> str:
    """
    Remueve números del texto.

    🔢 Números:
    En sentiment analysis, los números suelen no aportar:
    "10 out of 10 stars" → "out of stars"

    Args:
        text: Texto con números

    Returns:
        Texto sin números
    """
    return re.sub(r'\d+', '', text)


def remove_extra_whitespace(text: str) -> str:
    """
    Remueve espacios en blanco extras.

    Después de limpiar HTML, URLs, etc., pueden quedar espacios dobles.
    "Great  movie   here" → "Great movie here"

    Args:
        text: Texto con espacios extras

    Returns:
        Texto con espacios normalizados
    """
    return ' '.join(text.split())


def to_lowercase(text: str) -> str:
    """
    Convierte texto a minúsculas.

    🔡 CASE FOLDING:
    Para el modelo, "Movie", "movie", "MOVIE" son palabras diferentes.
    Convertir todo a minúsculas las unifica.

    Args:
        text: Texto con mayúsculas/minúsculas

    Returns:
        Texto en minúsculas
    """
    return text.lower()


# ============================================================================
# 2. FUNCIÓN DE LIMPIEZA COMPLETA
# ============================================================================

def clean_text(text: str,
               remove_html: bool = True,
               remove_url: bool = True,
               remove_punct: bool = True,
               remove_num: bool = True,
               lowercase: bool = True) -> str:
    """
    Aplica todas las limpiezas de texto.

    🧹 PIPELINE DE LIMPIEZA:
    1. HTML tags → removidos
    2. URLs → removidas
    3. Puntuación → removida
    4. Números → removidos
    5. Lowercase → aplicado
    6. Espacios extras → removidos

    Args:
        text: Texto a limpiar
        remove_html: Remover tags HTML
        remove_url: Remover URLs
        remove_punct: Remover puntuación
        remove_num: Remover números
        lowercase: Convertir a minúsculas

    Returns:
        Texto limpio

    Ejemplo:
        >>> text = "<p>I LOVED this movie! 10/10 ⭐</p>"
        >>> clean_text(text)
        "i loved this movie"
    """
    # Aplicar limpiezas en orden
    if remove_html:
        text = remove_html_tags(text)

    if remove_url:
        text = remove_urls(text)

    if lowercase:
        text = to_lowercase(text)

    if remove_punct:
        text = remove_punctuation(text)

    if remove_num:
        text = remove_numbers(text)

    # Siempre remover espacios extras al final
    text = remove_extra_whitespace(text)

    return text


# ============================================================================
# 3. TOKENIZACIÓN
# ============================================================================

def tokenize(text: str) -> List[str]:
    """
    Divide texto en tokens (palabras).

    🔤 TOKENIZACIÓN:
    Convierte texto en lista de palabras.

    "I love this movie" → ["I", "love", "this", "movie"]

    NLTK maneja casos especiales:
    - Contracciones: "don't" → ["do", "n't"]
    - Posesivos: "John's" → ["John", "'s"]

    Args:
        text: Texto a tokenizar

    Returns:
        Lista de tokens
    """
    return word_tokenize(text)


# ============================================================================
# 4. STOPWORDS (Palabras vacías)
# ============================================================================

def get_stopwords(language: str = 'english') -> set:
    """
    Obtiene conjunto de stopwords para un idioma.

    📚 STOPWORDS:
    Palabras muy frecuentes con poco significado semántico.

    Inglés (179 palabras):
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'as', 'by', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', ...

    Args:
        language: 'english', 'spanish', 'french', etc.

    Returns:
        Conjunto de stopwords
    """
    return set(stopwords.words(language))


def remove_stopwords(tokens: List[str], language: str = 'english') -> List[str]:
    """
    Remueve stopwords de una lista de tokens.

    🗑️ EJEMPLO:
    ANTES: ["the", "movie", "was", "really", "excellent"]
    DESPUÉS: ["movie", "really", "excellent"]

    Removidas: "the", "was" (stopwords)

    ⚠️ CONSIDERACIÓN:
    A veces las stopwords SÍ importan para el sentimiento:
    "not good" vs "good" tienen significados opuestos!

    Args:
        tokens: Lista de palabras
        language: Idioma de las stopwords

    Returns:
        Lista de tokens sin stopwords
    """
    stop_words = get_stopwords(language)
    return [token for token in tokens if token not in stop_words]


# ============================================================================
# 5. STEMMING (Derivación)
# ============================================================================

def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Aplica stemming a una lista de tokens.

    🌱 STEMMING (Porter Stemmer):
    Reduce palabras a su raíz de forma algorítmica (sin diccionario).

    EJEMPLOS:
    - "running" → "run"
    - "runner" → "runner"
    - "runs" → "run"
    - "movies" → "movi"  (⚠️ palabra inexistente)
    - "better" → "better"
    - "loving" → "love"

    VENTAJAS:
    ✅ Muy rápido
    ✅ Reduce vocabulario

    DESVENTAJAS:
    ❌ Produce palabras inexistentes
    ❌ Puede perder significado

    Args:
        tokens: Lista de palabras

    Returns:
        Lista de stems
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]


# ============================================================================
# 6. LEMMATIZATION (Lematización)
# ============================================================================

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Aplica lemmatization a una lista de tokens.

    📖 LEMMATIZATION (WordNet Lemmatizer):
    Reduce palabras a su forma base usando un diccionario léxico.

    EJEMPLOS:
    - "running" → "run"
    - "runner" → "runner"
    - "runs" → "run"
    - "movies" → "movie"  (✅ palabra válida)
    - "better" → "good"  (✅ entiende comparativos)
    - "loving" → "love"

    VENTAJAS:
    ✅ Siempre produce palabras válidas
    ✅ Preserva significado
    ✅ Más preciso que stemming

    DESVENTAJAS:
    ❌ Más lento que stemming
    ❌ Requiere recursos adicionales (WordNet)

    CUÁNDO USAR LEMMATIZATION vs STEMMING:
    - Lemmatization: Cuando necesitas precisión (análisis de sentimientos)
    - Stemming: Cuando necesitas velocidad (búsqueda de documentos)

    Args:
        tokens: Lista de palabras

    Returns:
        Lista de lemmas
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


# ============================================================================
# 7. PIPELINE COMPLETO DE PREPROCESAMIENTO
# ============================================================================

def preprocess_text(text: str,
                   remove_html: bool = True,
                   remove_url: bool = True,
                   remove_punct: bool = True,
                   remove_num: bool = True,
                   lowercase: bool = True,
                   remove_stops: bool = True,
                   apply_stemming: bool = False,
                   apply_lemmatization: bool = True) -> str:
    """
    Pipeline completo de preprocesamiento de texto.

    🔄 FLUJO COMPLETO:

    1. Texto original:
       "<p>I LOVED this movie! 10/10 stars ⭐</p>"

    2. Limpieza:
       "i loved this movie stars"

    3. Tokenización:
       ["i", "loved", "this", "movie", "stars"]

    4. Remover stopwords:
       ["loved", "movie", "stars"]

    5. Lemmatization:
       ["love", "movie", "star"]

    6. Resultado final:
       "love movie star"

    Args:
        text: Texto original
        remove_html: Remover HTML
        remove_url: Remover URLs
        remove_punct: Remover puntuación
        remove_num: Remover números
        lowercase: Convertir a minúsculas
        remove_stops: Remover stopwords
        apply_stemming: Aplicar stemming
        apply_lemmatization: Aplicar lemmatization

    Returns:
        Texto preprocesado
    """
    # 1. Limpiar texto
    text = clean_text(
        text,
        remove_html=remove_html,
        remove_url=remove_url,
        remove_punct=remove_punct,
        remove_num=remove_num,
        lowercase=lowercase
    )

    # 2. Tokenizar
    tokens = tokenize(text)

    # 3. Remover stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # 4. Stemming o Lemmatization (no ambos)
    if apply_stemming:
        tokens = stem_tokens(tokens)
    elif apply_lemmatization:
        tokens = lemmatize_tokens(tokens)

    # 5. Unir tokens de vuelta en texto
    processed_text = ' '.join(tokens)

    return processed_text


def preprocess_texts(texts: List[str], **kwargs) -> List[str]:
    """
    Preprocesa una lista de textos.

    Útil para procesar todo el dataset de una vez.

    Args:
        texts: Lista de textos
        **kwargs: Parámetros para preprocess_text()

    Returns:
        Lista de textos preprocesados
    """
    return [preprocess_text(text, **kwargs) for text in texts]


# ============================================================================
# DEMO Y EJEMPLOS
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Preprocesamiento de texto paso a paso
    """
    print("\n" + "🧹" * 35)
    print(" DEMO: TEXT PREPROCESSING")
    print("🧹" * 35)

    # Texto de ejemplo (similar a reviews reales)
    sample_text = """
    <p><b>I ABSOLUTELY LOVED this movie!!!</b> 🎬</p>
    The acting was superb and the storyline kept me engaged throughout.
    I would give it 10/10 stars! ⭐⭐⭐⭐⭐
    Check out the trailer at https://example.com/trailer
    """

    print(f"\n📝 TEXTO ORIGINAL:")
    print("-" * 70)
    print(sample_text)

    # Paso 1: Limpieza básica
    print(f"\n\n1️⃣  DESPUÉS DE LIMPIEZA BÁSICA:")
    print("-" * 70)
    cleaned = clean_text(sample_text)
    print(cleaned)

    # Paso 2: Tokenización
    print(f"\n\n2️⃣  DESPUÉS DE TOKENIZACIÓN:")
    print("-" * 70)
    tokens = tokenize(cleaned)
    print(f"Tokens ({len(tokens)}): {tokens}")

    # Paso 3: Remover stopwords
    print(f"\n\n3️⃣  DESPUÉS DE REMOVER STOPWORDS:")
    print("-" * 70)
    tokens_no_stop = remove_stopwords(tokens)
    print(f"Tokens ({len(tokens_no_stop)}): {tokens_no_stop}")
    print(f"Removidas: {set(tokens) - set(tokens_no_stop)}")

    # Paso 4: Stemming
    print(f"\n\n4️⃣  DESPUÉS DE STEMMING:")
    print("-" * 70)
    stemmed = stem_tokens(tokens_no_stop)
    print(f"Stems: {stemmed}")

    # Paso 5: Lemmatization
    print(f"\n\n5️⃣  DESPUÉS DE LEMMATIZATION:")
    print("-" * 70)
    lemmatized = lemmatize_tokens(tokens_no_stop)
    print(f"Lemmas: {lemmatized}")

    # Pipeline completo
    print(f"\n\n6️⃣  RESULTADO FINAL (Pipeline completo):")
    print("-" * 70)
    final_text = preprocess_text(sample_text)
    print(f"Texto original: {len(sample_text)} caracteres")
    print(f"Texto procesado: {len(final_text)} caracteres")
    print(f"\nTexto procesado:")
    print(f"'{final_text}'")

    # Comparación: Stemming vs Lemmatization
    print(f"\n\n7️⃣  COMPARACIÓN: STEMMING vs LEMMATIZATION")
    print("-" * 70)

    test_words = ["running", "runs", "ran", "movies", "better", "best", "loving"]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    print(f"\n{'Palabra':<15} {'Stemming':<15} {'Lemmatization':<15}")
    print("-" * 45)
    for word in test_words:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word)
        print(f"{word:<15} {stem:<15} {lemma:<15}")

    print("\n✅ Demo completada!")
