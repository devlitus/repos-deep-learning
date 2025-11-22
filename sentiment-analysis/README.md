# Sentiment Analysis (Análisis de Sentimientos)

Proyecto educativo de **Natural Language Processing (NLP)** para clasificar reviews de películas como positivas o negativas usando técnicas clásicas de Machine Learning y Deep Learning.

## 📚 Objetivos de Aprendizaje

Este proyecto te enseña conceptos nuevos de NLP que complementan tu experiencia previa con ML:

### Conceptos Nuevos (NLP):
- ✅ **Tokenización**: Convertir texto en palabras/tokens
- ✅ **Stopwords**: Remover palabras sin significado ("el", "la", "de")
- ✅ **Stemming/Lemmatization**: Reducir palabras a su raíz
- ✅ **TF-IDF**: Convertir texto a vectores numéricos ponderados
- ✅ **Word Embeddings**: Representación vectorial densa de palabras
- ✅ **Naive Bayes**: Modelo probabilístico ideal para texto
- ✅ **SVM para texto**: Clasificación con datos de alta dimensión
- ✅ **LSTM para secuencias de palabras**: Procesar orden de palabras

### Conexiones con Proyectos Anteriores:
- 🔗 **prediccion-temperatura**: LSTM, pero con secuencias de palabras en lugar de números
- 🔗 **fraude-detection**: Clasificación binaria, pero con features textuales
- 🔗 **predictor-titanic**: Mismo tipo de problema (clasificación), pero con texto

---

## 📊 Dataset

**IMDB Movie Reviews Dataset** (incluido en Keras):
- **50,000 reviews** de películas
- **25,000 entrenamiento**, **25,000 prueba**
- **Clasificación binaria**: 0 = negativo, 1 = positivo
- **Balanceado**: 50% positivo, 50% negativo

Ejemplo de reviews:
- Positivo: *"This movie is absolutely excellent! Best film I've ever seen."*
- Negativo: *"Terrible waste of time. Awful acting and boring plot."*

---

## 🏗️ Arquitectura del Proyecto

```
sentiment-analysis/
├── data/
│   ├── raw/                    # Dataset IMDB (descargado automáticamente)
│   └── processed/              # Datos procesados (opcional)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Carga y exploración del dataset IMDB
│   ├── text_preprocessing.py   # Limpieza, tokenización, stemming, lemmatization
│   ├── feature_extraction.py   # TF-IDF, Bag of Words, Embeddings
│   ├── model.py                # Modelos clásicos (Naive Bayes, SVM)
│   ├── deep_model.py           # Modelo LSTM con Bidirectional layers
│   ├── visualizations.py       # Word clouds, distribuciones, métricas
│   └── predictor.py            # Predicción en nuevos textos
├── models/                     # Modelos entrenados (.pkl, .keras)
├── reports/                    # Visualizaciones y métricas (.png, .json)
├── notebooks/                  # Jupyter notebooks (análisis exploratorio)
├── config.py                   # Configuración central (rutas, hiperparámetros)
├── main.py                     # Pipeline completo end-to-end
├── requirements.txt            # Dependencias de NLP
└── README.md                   # Este archivo
```

---

## 🛠️ Instalación

### 1. Clonar y navegar al proyecto

```bash
cd sentiment-analysis
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar recursos de NLTK

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

---

## 🚀 Uso

### Pipeline Completo (Recomendado)

Ejecuta el pipeline completo que:
1. Carga el dataset IMDB
2. Explora y visualiza los datos
3. Entrena 3 modelos (Naive Bayes, SVM, LSTM)
4. Evalúa y compara modelos
5. Guarda modelos entrenados
6. Hace predicciones de ejemplo

```bash
python main.py
```

**Tiempo estimado**: 15-30 minutos (dependiendo del hardware)

### Entrenar Solo LSTM

Si ya tienes los modelos clásicos y solo quieres entrenar el LSTM:

```bash
python main.py lstm
```

### Demo Rápido (Requiere Modelos Entrenados)

Prueba predicciones sin re-entrenar:

```bash
python main.py demo
```

### Usar en Código Personalizado

```python
from src.predictor import SentimentPredictorLSTM, SentimentPredictorClassic
import config

# Predictor con LSTM
predictor_lstm = SentimentPredictorLSTM(
    model_path=config.MODEL_LSTM,
    tokenizer_path=config.TOKENIZER_FILE
)

text = "This movie is absolutely amazing!"
prediction, confidence, label = predictor_lstm.predict_with_confidence(text)
print(f"{label} (confianza: {confidence:.2%})")
# Output: Positivo ✅ (confianza: 95.3%)
```

---

## 📈 Modelos Implementados

### 1. Naive Bayes (Multinomial)
**Características:**
- Modelo probabilístico basado en teorema de Bayes
- Muy rápido de entrenar
- Funciona excepcionalmente bien con texto
- Asume independencia entre palabras ("naive")

**Uso con:**
- TF-IDF features
- Bag of Words features

**Resultados esperados:**
- Accuracy: ~85-87%
- Training time: < 1 segundo

### 2. Support Vector Machine (SVM Lineal)
**Características:**
- Encuentra hiperplano óptimo de separación
- Excelente con datos de alta dimensión
- Robusto a overfitting
- Solo usa vectores de soporte (eficiente)

**Uso con:**
- TF-IDF features (recomendado)

**Resultados esperados:**
- Accuracy: ~87-89%
- Training time: ~10-30 segundos

**Ventaja adicional:**
- Interpretable: Puedes ver qué palabras son más importantes

### 3. LSTM (Long Short-Term Memory)
**Características:**
- Red neuronal recurrente
- Captura dependencias temporales (orden de palabras)
- Entiende contexto: "not good" ≠ "good"
- Usa word embeddings (representación densa)

**Arquitectura:**
```
Input (secuencia de palabras)
    ↓
Embedding Layer (100 dimensiones)
    ↓
Bidirectional LSTM (128 units)
    ↓
Bidirectional LSTM (64 units)
    ↓
Dense + Dropout (64 units)
    ↓
Output (Sigmoid: probabilidad 0-1)
```

**Resultados esperados:**
- Accuracy: ~88-90%
- Training time: 5-15 minutos (GPU recomendado)

---

## 📊 Resultados Típicos

| Modelo | Accuracy | Precision | Recall | F1-Score | Tiempo Entrenamiento |
|--------|----------|-----------|--------|----------|---------------------|
| Naive Bayes | 85-87% | 85-87% | 85-87% | 85-87% | < 1 seg |
| SVM | 87-89% | 87-89% | 87-89% | 87-89% | 10-30 seg |
| LSTM | 88-90% | 88-90% | 88-90% | 88-90% | 5-15 min |

---

## 🔍 Conceptos Clave Explicados

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)

**¿Qué es?**
Medida estadística que evalúa qué tan importante es una palabra en un documento.

**Fórmula:**
```
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

TF = frecuencia de palabra en documento
IDF = log(total_docs / docs_con_palabra)
```

**Ejemplo:**
- Palabra "movie" aparece en el 95% de reviews → IDF bajo → Poco discriminativa
- Palabra "masterpiece" aparece en el 1% de reviews → IDF alto → Muy discriminativa

### 2. Word Embeddings

**¿Qué son?**
Representación vectorial densa de palabras donde palabras similares tienen vectores similares.

**Ejemplo (3 dimensiones, real: 100-300):**
```
"excellent" → [0.8, 0.2, 0.1]
"great"     → [0.75, 0.25, 0.15]  ← Cercano a "excellent"
"terrible"  → [-0.7, 0.1, -0.2]   ← Opuesto a "excellent"
```

**Propiedades mágicas:**
- king - man + woman ≈ queen
- Paris - France + Italy ≈ Rome

### 3. LSTM para Texto

**¿Por qué LSTM?**
A diferencia de Bag of Words o TF-IDF, LSTM entiende el **orden** de las palabras:

```
Bag of Words:
"not good" = ["not", "good"] → Puede confundir
"good" = ["good"] → Mismo vocabulario

LSTM:
"not" → hidden_state_1
"good" + hidden_state_1 → hidden_state_2 (negativo!)
Entiende que "not" invierte el sentimiento
```

---

## 📁 Archivos de Configuración

### config.py

Todos los parámetros configurables están centralizados:

```python
# Preprocesamiento
REMOVE_STOPWORDS = True
USE_LEMMATIZATION = True
USE_STEMMING = False

# Feature extraction
TFIDF_MAX_FEATURES = 5000
EMBEDDING_DIM = 100

# LSTM
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
DROPOUT_RATE = 0.5
EPOCHS = 10
BATCH_SIZE = 64
```

---

## 🎨 Visualizaciones

El proyecto genera varias visualizaciones en `reports/`:

1. **Word Clouds**: Nubes de palabras positivas vs negativas
2. **Distribución de longitudes**: Histograma de longitudes de reviews
3. **Matriz de confusión**: Rendimiento del modelo
4. **Training history**: Loss y accuracy durante entrenamiento (LSTM)
5. **Top palabras**: Palabras más frecuentes y más importantes

---

## 🧪 Ejemplos de Predicción

```python
from src.predictor import SentimentPredictorLSTM
import config

# Cargar predictor
predictor = SentimentPredictorLSTM(
    model_path=config.MODEL_LSTM,
    tokenizer_path=config.TOKENIZER_FILE
)

# Ejemplos
texts = [
    "This movie is absolutely excellent!",
    "Terrible waste of time.",
    "It was okay, nothing special.",
    "I loved every minute of it!",
    "Disappointing. Expected better."
]

# Predecir
for text in texts:
    pred, conf, label = predictor.predict_with_confidence(text)
    print(f"{label} ({conf:.1%}) - {text}")
```

**Output esperado:**
```
Positivo ✅ (95.3%) - This movie is absolutely excellent!
Negativo ❌ (92.7%) - Terrible waste of time.
Positivo ✅ (56.2%) - It was okay, nothing special.
Positivo ✅ (98.1%) - I loved every minute of it!
Negativo ❌ (87.4%) - Disappointing. Expected better.
```

---

## 🔧 Experimentación

### Modificar Hiperparámetros

Edita `config.py`:

```python
# Probar con más épocas
EPOCHS = 20

# Aumentar tamaño de LSTM
LSTM_UNITS_1 = 256
LSTM_UNITS_2 = 128

# Usar más palabras del vocabulario
NUM_WORDS = 20000

# Secuencias más largas
MAX_SEQUENCE_LENGTH = 300
```

### Probar con Tus Propios Datos

```python
from src.text_preprocessing import preprocess_text
from src.predictor import SentimentPredictorLSTM
import config

predictor = SentimentPredictorLSTM(
    model_path=config.MODEL_LSTM,
    tokenizer_path=config.TOKENIZER_FILE
)

# Tus propias reviews
my_reviews = [
    "Tu review aquí",
    "Otra review"
]

for review in my_reviews:
    pred, conf, label = predictor.predict_with_confidence(review)
    print(f"{label} - {review}")
```

---

## 📚 Recursos de Aprendizaje

### Conceptos NLP:
- [NLTK Documentation](https://www.nltk.org/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Word Embeddings](https://www.tensorflow.org/text/guide/word_embeddings)

### Modelos:
- [Naive Bayes for Text](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [SVM for Text Classification](https://scikit-learn.org/stable/modules/svm.html)
- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🐛 Troubleshooting

### Error: "NLTK resources not found"

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Error: "Model file not found"

Ejecuta primero el training:
```bash
python main.py
```

### Entrenamiento muy lento

Para LSTM, considera:
1. Usar menos épocas: `EPOCHS = 5` en config.py
2. Batch size más grande: `BATCH_SIZE = 128`
3. Entrenar con subset: Modificar `load_imdb_data()` para usar menos datos

---

## 🎯 Próximos Pasos

1. ✅ **Crear Web App con Streamlit**
   - Interfaz interactiva para predicciones
   - Visualización en tiempo real

2. ✅ **Probar otros datasets**
   - Reviews de productos de Amazon
   - Tweets con sentimientos
   - Reviews de restaurantes de Yelp

3. ✅ **Experimentar con modelos avanzados**
   - Transformers (BERT, RoBERTa)
   - Embeddings pre-entrenados (GloVe, Word2Vec)
   - Ensembles (combinar múltiples modelos)

4. ✅ **Análisis multiclase**
   - Clasificación de 5 estrellas (1-5)
   - Detección de emociones (alegría, tristeza, ira, etc.)

---

## 📝 Notas Importantes

1. **Preprocesamiento**: Para modelos clásicos (TF-IDF), el preprocesamiento agresivo (stopwords, lemmatization) ayuda. Para LSTM, menos preprocesamiento puede ser mejor.

2. **Overfitting**: LSTM es propenso a overfitting. Usa Dropout (0.3-0.5) y EarlyStopping.

3. **Tiempo de entrenamiento**: LSTM requiere más tiempo. Considera usar GPU o reducir epochs para experimentación.

4. **Vocabulario**: Más palabras (NUM_WORDS) = más features = mejor captura de información pero más memoria y tiempo.

---

## 📄 Licencia

Proyecto educativo - Libre para uso y modificación.

---

## ✨ Créditos

- **Dataset**: [IMDB Movie Reviews](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
- **Inspiración**: Proyectos previos (prediccion-temperatura, fraude-detection, predictor-titanic)
- **Librerías**: scikit-learn, TensorFlow/Keras, NLTK, spaCy

---

**¡Feliz aprendizaje de NLP! 🚀**
