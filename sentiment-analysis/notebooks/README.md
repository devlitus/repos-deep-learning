# 📓 Notebooks Educativos - Análisis de Sentimientos

## 🎯 Descripción

Estos notebooks te permiten aprender **Análisis de Sentimientos con NLP** de forma **interactiva**, ejecutando celda por celda y viendo los resultados inmediatamente.

---

## 📚 Lista de Notebooks

### **Notebook 1: Introducción y Dataset IMDB** ⭐ **EMPIEZA AQUÍ**
📁 `01_introduccion_dataset_imdb.ipynb`

**Aprenderás:**
- ✅ Qué es el Análisis de Sentimientos
- ✅ Cargar el dataset IMDB (50,000 reviews)
- ✅ Explorar estadísticas del dataset
- ✅ Entender qué es la tokenización
- ✅ Decodificar reviews (números → texto)
- ✅ Ver ejemplos de reviews positivas y negativas

⏱️ **Tiempo**: 15 minutos

---

### **Notebook 2: Tokenización y Preprocesamiento**
📁 `02_tokenizacion_preprocesamiento.ipynb`

**Aprenderás:**
- ✅ Limpieza de texto paso a paso (HTML, URLs, puntuación)
- ✅ Stopwords: qué son y por qué removerlas
- ✅ Stemming vs Lemmatization (comparación detallada)
- ✅ Pipeline completo de preprocesamiento
- ✅ Ejercicios interactivos

⏱️ **Tiempo**: 20 minutos

---

### **Notebook 3: Feature Extraction (TF-IDF)** 🔢
📁 `03_feature_extraction_tfidf.ipynb`

**Aprenderás:**
- ✅ Por qué convertir texto → números
- ✅ Bag of Words (bolsa de palabras)
- ✅ TF-IDF explicado con ejemplos visuales
- ✅ Crear vectorizador TF-IDF
- ✅ Ver palabras más importantes

⏱️ **Tiempo**: 25 minutos

---

### **Notebook 4: Modelos Clásicos (ML)** 🤖
📁 `04_modelos_clasicos_ML.ipynb`

**Aprenderás:**
- ✅ Naive Bayes (modelo probabilístico)
- ✅ SVM (Support Vector Machine)
- ✅ Entrenar y evaluar modelos
- ✅ Matriz de confusión
- ✅ Feature importance (palabras más importantes)
- ✅ Comparar modelos

⏱️ **Tiempo**: 30 minutos

---

### **Notebook 5: Word Embeddings y LSTM** 🧠
📁 `05_word_embeddings_lstm.ipynb`

**Aprenderás:**
- ✅ Qué son Word Embeddings
- ✅ Diferencia con TF-IDF
- ✅ Construir modelo LSTM paso a paso
- ✅ Entrenar red neuronal
- ✅ Visualizar training history
- ✅ Por qué LSTM entiende "not good"

⏱️ **Tiempo**: 35 minutos

---

### **Notebook 6: Predicciones Interactivas** 🔮
📁 `06_predicciones_nuevas_reviews.ipynb`

**Aprenderás:**
- ✅ Cargar modelos entrenados
- ✅ Predecir tus propias reviews
- ✅ Comparar SVM vs LSTM
- ✅ Analizar casos difíciles
- ✅ Ejercicios prácticos

⏱️ **Tiempo**: 20 minutos

---

## 🚀 Cómo Usar los Notebooks

### **Opción 1: Jupyter Notebook (Clásico)**

```bash
# 1. Navegar a la carpeta
cd sentiment-analysis/notebooks

# 2. Iniciar Jupyter
jupyter notebook

# 3. Abrir el primer notebook (01_introduccion_dataset_imdb.ipynb)
```

### **Opción 2: JupyterLab (Moderno)**

```bash
# 1. Navegar a la carpeta
cd sentiment-analysis/notebooks

# 2. Iniciar JupyterLab
jupyter lab

# 3. Abrir el primer notebook
```

### **Opción 3: VS Code**

```bash
# 1. Abrir VS Code
code .

# 2. Abrir cualquier notebook (.ipynb)
# 3. VS Code detectará automáticamente el kernel de Python
```

---

## 📝 Orden Recomendado

**Para aprender desde cero:**

```
1. Notebook 01 → Introducción y Dataset
   ↓
2. Notebook 02 → Preprocesamiento
   ↓
3. Notebook 03 → TF-IDF
   ↓
4. Notebook 04 → Modelos Clásicos
   ↓
5. Notebook 05 → LSTM
   ↓
6. Notebook 06 → Predicciones
```

**Si ya sabes NLP básico:**

```
Puedes saltar directamente a:
- Notebook 04: Modelos Clásicos
- Notebook 05: LSTM
- Notebook 06: Predicciones
```

---

## ⚙️ Requisitos

### Instalar dependencias:

```bash
cd sentiment-analysis
pip install -r requirements.txt
```

### Descargar recursos de NLTK:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

---

## 💡 Consejos para Aprender Mejor

### **1. Ejecuta Celda por Celda**
No ejecutes todo de golpe. Lee la explicación, ejecuta la celda, entiende el resultado.

### **2. Modifica el Código**
Cambia valores, experimenta, rompe cosas. Así se aprende.

### **3. Toma Notas**
Usa celdas Markdown para agregar tus propias notas.

### **4. Haz los Ejercicios**
Cada notebook tiene ejercicios interactivos. ¡Házlos!

### **5. No te Saltes Notebooks**
Los conceptos se construyen uno sobre otro.

---

## 🎯 Ejercicios Adicionales

Al final de cada notebook encontrarás:

- 📝 **Ejercicios prácticos** (cambia parámetros y observa)
- 🎮 **Ejercicios interactivos** (prueba con tus propios datos)
- 🤔 **Preguntas de reflexión** (para verificar que entendiste)

---

## 🐛 Troubleshooting

### **Error: "ModuleNotFoundError: No module named 'src'"**

**Solución:**
```python
import sys
sys.path.append('..')  # Esta línea debe estar al inicio de cada notebook
```

### **Error: "NLTK resources not found"**

**Solución:**
```bash
python -c "import nltk; nltk.download('all')"
```

### **Error: "Kernel died"**

**Solución:**
- Memoria insuficiente → Reinicia el kernel
- Reduce el tamaño del dataset en las celdas de ejemplo

---

## 📊 Resultados Esperados

Después de completar los notebooks sabrás:

✅ Cómo funciona el Análisis de Sentimientos
✅ Preprocesar texto para NLP
✅ Convertir texto a números (TF-IDF, embeddings)
✅ Entrenar modelos clásicos (Naive Bayes, SVM)
✅ Entrenar redes neuronales (LSTM)
✅ Evaluar y comparar modelos
✅ Predecir sentimientos en nuevas reviews

---

## 🎓 Después de los Notebooks

Una vez completados todos los notebooks, puedes:

1. **Ejecutar el pipeline completo**:
   ```bash
   cd ..
   python main.py
   ```

2. **Crear tu propia web app con Streamlit** (proyecto futuro)

3. **Experimentar con otros datasets**:
   - Reviews de productos de Amazon
   - Tweets
   - Reviews de restaurantes

---

## 📚 Recursos Adicionales

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Text Processing](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [TensorFlow Text Classification](https://www.tensorflow.org/tutorials/keras/text_classification)

---

**¡Feliz aprendizaje!** 🚀📓
