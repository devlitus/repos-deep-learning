"""
Visualizations - Análisis de Sentimientos
==========================================

Visualizaciones específicas para NLP y análisis de sentimientos.

NUEVAS VISUALIZACIONES (específicas de NLP):
--------------------------------------------
1. WORD CLOUD (Nube de palabras):
   Muestra palabras más frecuentes con tamaño proporcional a frecuencia

2. Distribución de longitudes de texto
3. Top palabras más frecuentes
4. Comparación de vocabulario positivo vs negativo
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter
from typing import List
import config

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. WORD CLOUD (Nube de Palabras)
# ============================================================================

def create_wordcloud(texts: List[str],
                    title: str = "Word Cloud",
                    color: str = 'viridis',
                    save_path = None):
    """
    Crea una nube de palabras.

    ☁️ WORD CLOUD:
    ==============
    Visualización donde el TAMAÑO de cada palabra representa su FRECUENCIA.

    Palabras más grandes = Aparecen más veces

    MUY ÚTIL para:
    - Ver qué palabras dominan en reviews positivas vs negativas
    - Identificar temas comunes
    - Entender el vocabulario del dataset

    Args:
        texts: Lista de textos
        title: Título del gráfico
        color: Mapa de colores ('viridis', 'plasma', 'Blues', 'Reds')
        save_path: Ruta para guardar imagen
    """
    print(f"\n☁️  Generando word cloud: {title}")

    # Combinar todos los textos
    combined_text = ' '.join(texts)

    # Crear wordcloud
    wordcloud = WordCloud(
        width=config.WORDCLOUD_WIDTH,
        height=config.WORDCLOUD_HEIGHT,
        background_color=config.WORDCLOUD_BACKGROUND,
        colormap=color,
        max_words=100,  # Máximo 100 palabras
        relative_scaling=0.5,
        min_font_size=10
    ).generate(combined_text)

    # Visualizar
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Guardado: {save_path.name}")

    plt.show()


def create_sentiment_wordclouds(positive_texts: List[str],
                                negative_texts: List[str],
                                save_dir = None):
    """
    Crea word clouds comparativos para sentimientos positivos y negativos.

    📊 COMPARACIÓN VISUAL:
    ======================
    Permite ver qué palabras son características de cada sentimiento.

    Ejemplo de insights:
    - Positivo: "excellent", "great", "loved", "perfect"
    - Negativo: "terrible", "awful", "waste", "boring"

    Args:
        positive_texts: Textos con sentimiento positivo
        negative_texts: Textos con sentimiento negativo
        save_dir: Directorio para guardar imágenes
    """
    print("\n" + "=" * 70)
    print("☁️  WORD CLOUDS COMPARATIVOS")
    print("=" * 70)

    # Word cloud positivo
    if save_dir:
        save_path_pos = save_dir / 'wordcloud_positive.png'
    else:
        save_path_pos = None

    create_wordcloud(
        positive_texts,
        title="🟢 Palabras en Reviews POSITIVAS",
        color='Greens',
        save_path=save_path_pos
    )

    # Word cloud negativo
    if save_dir:
        save_path_neg = save_dir / 'wordcloud_negative.png'
    else:
        save_path_neg = None

    create_wordcloud(
        negative_texts,
        title="🔴 Palabras en Reviews NEGATIVAS",
        color='Reds',
        save_path=save_path_neg
    )


# ============================================================================
# 2. DISTRIBUCIÓN DE LONGITUDES
# ============================================================================

def plot_text_length_distribution(texts: List[str],
                                  labels: np.ndarray = None,
                                  save_path = None):
    """
    Visualiza la distribución de longitudes de texto.

    📏 LONGITUD DE TEXTO:
    =====================
    Importante para:
    - Decidir MAX_SEQUENCE_LENGTH (para padding)
    - Entender el dataset
    - Detectar outliers (textos muy largos o muy cortos)

    Args:
        texts: Lista de textos
        labels: Etiquetas (opcional, para comparar pos vs neg)
        save_path: Ruta para guardar
    """
    print(f"\n📏 Analizando distribución de longitudes...")

    # Calcular longitudes
    lengths = [len(text.split()) for text in texts]

    plt.figure(figsize=(12, 6))

    if labels is not None:
        # Separar por sentimiento
        positive_lengths = [l for l, label in zip(lengths, labels) if label == 1]
        negative_lengths = [l for l, label in zip(lengths, labels) if label == 0]

        plt.hist(positive_lengths, bins=50, alpha=0.6, label='Positivo', color=config.COLOR_POSITIVE)
        plt.hist(negative_lengths, bins=50, alpha=0.6, label='Negativo', color=config.COLOR_NEGATIVE)
        plt.legend()
    else:
        plt.hist(lengths, bins=50, alpha=0.7, color='steelblue')

    plt.xlabel('Longitud (palabras)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Longitud de Textos', fontsize=14, fontweight='bold')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Media: {np.mean(lengths):.1f}')
    plt.axvline(np.median(lengths), color='orange', linestyle='--', label=f'Mediana: {np.median(lengths):.1f}')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Guardado: {save_path.name}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# 3. TOP PALABRAS MÁS FRECUENTES
# ============================================================================

def plot_top_words(texts: List[str],
                  top_n: int = 20,
                  title: str = "Top Palabras",
                  save_path = None):
    """
    Muestra las N palabras más frecuentes en un gráfico de barras.

    Args:
        texts: Lista de textos
        top_n: Número de palabras top
        title: Título del gráfico
        save_path: Ruta para guardar
    """
    print(f"\n📊 Analizando top {top_n} palabras más frecuentes...")

    # Contar palabras
    all_words = ' '.join(texts).split()
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(top_n)

    words, counts = zip(*top_words)

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(words)), counts, color='steelblue')
    plt.yticks(range(len(words)), words)
    plt.xlabel('Frecuencia', fontsize=12)
    plt.ylabel('Palabra', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Invertir para que el top esté arriba
    plt.grid(axis='x', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Guardado: {save_path.name}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. DISTRIBUCIÓN DE SENTIMIENTOS
# ============================================================================

def plot_sentiment_distribution(labels: np.ndarray,
                                save_path = None):
    """
    Visualiza la distribución de sentimientos (balance de clases).

    Args:
        labels: Array de etiquetas (0/1)
        save_path: Ruta para guardar
    """
    print(f"\n📊 Visualizando distribución de sentimientos...")

    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 6))
    colors = [config.COLOR_NEGATIVE, config.COLOR_POSITIVE]
    bars = plt.bar(['Negativo', 'Positivo'], counts, color=colors, alpha=0.7)

    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.ylabel('Número de reviews', fontsize=12)
    plt.title('Distribución de Sentimientos', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Guardado: {save_path.name}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# 5. MATRIZ DE CONFUSIÓN
# ============================================================================

def plot_confusion_matrix(cm: np.ndarray,
                         save_path = None):
    """
    Visualiza matriz de confusión.

    Args:
        cm: Matriz de confusión (2x2)
        save_path: Ruta para guardar
    """
    print(f"\n📊 Visualizando matriz de confusión...")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    plt.ylabel('Real', fontsize=12)
    plt.xlabel('Predicho', fontsize=12)
    plt.title('Matriz de Confusión', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Guardado: {save_path.name}")

    plt.tight_layout()
    plt.show()


# ============================================================================
# 6. HISTORIAL DE ENTRENAMIENTO (LSTM)
# ============================================================================

def plot_training_history(history,
                          save_path = None):
    """
    Visualiza historial de entrenamiento de modelo LSTM.

    Similar a prediccion-temperatura!

    Args:
        history: Objeto History de Keras
        save_path: Ruta para guardar
    """
    print(f"\n📈 Visualizando historial de entrenamiento...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Pérdida durante Entrenamiento', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Época', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Precisión durante Entrenamiento', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Guardado: {save_path.name}")

    plt.show()


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    """
    Demo: Visualizaciones con datos de ejemplo
    """
    print("\n" + "📊" * 35)
    print(" DEMO: NLP VISUALIZATIONS")
    print("📊" * 35)

    # Datos de ejemplo
    positive_texts = [
        "excellent movie loved it",
        "great performance amazing story",
        "wonderful film highly recommend",
        "fantastic acting brilliant"
    ] * 10  # Repetir para tener más datos

    negative_texts = [
        "terrible waste of time",
        "awful acting boring plot",
        "horrible movie disappointed",
        "worst film ever"
    ] * 10

    all_texts = positive_texts + negative_texts
    labels = np.array([1]*len(positive_texts) + [0]*len(negative_texts))

    # 1. Word clouds
    print("\n1️⃣  Generando word clouds...")
    create_wordcloud(positive_texts, "Palabras Positivas (Demo)", 'Greens')

    # 2. Distribución de longitudes
    print("\n2️⃣  Distribución de longitudes...")
    plot_text_length_distribution(all_texts, labels)

    # 3. Distribución de sentimientos
    print("\n3️⃣  Distribución de sentimientos...")
    plot_sentiment_distribution(labels)

    print("\n✅ Demo completada!")
