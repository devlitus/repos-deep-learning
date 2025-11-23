# Sentiment Analysis Web App

Aplicación web interactiva para análisis de sentimientos usando LSTM y Streamlit.

## 🚀 Inicio Rápido

### Requisitos Previos

1. **Entrenar el modelo** (si no lo has hecho):
```bash
cd ..  # Ir al directorio sentiment-analysis
python main.py
```

2. **Instalar Streamlit** (si no está instalado):
```bash
pip install streamlit
```

### Ejecutar la Aplicación

#### Opción 1: Scripts de ejecución

**Linux/Mac:**
```bash
chmod +x run.sh  # Solo la primera vez
./run.sh
```

**Windows:**
```bash
run.bat
```

#### Opción 2: Comando directo

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en: **http://localhost:8501**

## 📱 Páginas de la Aplicación

### 🏠 Página Principal (app.py)
- **Predicción en tiempo real** de sentimientos
- Input de texto libre o ejemplos predefinidos
- Visualización de confianza y probabilidades
- Análisis del texto preprocesado
- Estadísticas de palabras

### 📊 Métricas (pages/1_Metricas.py)
- Rendimiento del modelo (Accuracy, Precision, Recall, AUC-ROC)
- Histórico de entrenamiento (Loss y Accuracy)
- Comparación con modelos clásicos (SVM, Naive Bayes)
- Ejemplos de predicciones
- Distribución de confianza

### ℹ️ About (pages/2_About.py)
- Descripción del proyecto
- Stack tecnológico
- Información del dataset IMDB
- Arquitectura del modelo LSTM
- Guía de uso y referencias

## 🛠️ Características

- ✅ **Predicciones instantáneas** con modelo LSTM entrenado
- ✅ **Interfaz intuitiva** con ejemplos predefinidos
- ✅ **Visualizaciones interactivas** de métricas y resultados
- ✅ **Análisis de confianza** para cada predicción
- ✅ **Preprocesamiento automático** de texto
- ✅ **Responsive design** compatible con móviles

## 📂 Estructura

```
web/
├── app.py                      # Página principal de predicción
├── pages/                      # Páginas adicionales
│   ├── 1_Metricas.py          # Métricas y visualizaciones
│   └── 2_About.py             # Información del proyecto
├── utils/                      # Utilidades (opcional)
├── run.sh                      # Script de ejecución Linux/Mac
├── run.bat                     # Script de ejecución Windows
└── README.md                   # Este archivo
```

## 🎯 Cómo Usar

1. **Escribe una reseña** de película en el cuadro de texto
2. O **selecciona un ejemplo** del sidebar
3. Haz clic en **"🔍 Analizar"**
4. Observa:
   - Sentimiento predicho (Positivo/Negativo)
   - Nivel de confianza
   - Probabilidades detalladas
   - Análisis del texto

### Ejemplos de Reseñas

**Positiva:**
```
This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.
```

**Negativa:**
```
Terrible movie. Waste of time and money. The acting was awful and the plot made no sense.
```

**Mixta:**
```
The movie had some good moments but overall it was disappointing. Great visuals but weak story.
```

## 🧠 Modelo

- **Arquitectura:** LSTM (Long Short-Term Memory)
- **Dataset:** IMDB (50,000 reseñas de películas)
- **Accuracy:** ~87%
- **Vocabulario:** 10,000 palabras más frecuentes
- **Longitud máxima:** 300 palabras

## 📊 Métricas de Rendimiento

| Métrica    | Valor |
|------------|-------|
| Accuracy   | 87.1% |
| Precision  | 86.8% |
| Recall     | 87.1% |
| AUC-ROC    | 93.3% |

## 🔧 Tecnologías

- **Framework Web:** Streamlit
- **Deep Learning:** TensorFlow/Keras
- **NLP:** NLTK
- **Visualización:** Matplotlib, Seaborn
- **Data:** Pandas, NumPy

## 💡 Tips

- Las reseñas más largas y detalladas obtienen mejores predicciones
- El modelo está entrenado en inglés (IMDB dataset)
- Sé específico sobre lo que te gustó o disgustó
- El modelo puede tener dificultades con sarcasmo o ironía

## ⚙️ Configuración Avanzada

### Cambiar Puerto

```bash
streamlit run app.py --server.port 8502
```

### Desactivar Modo Watch

```bash
streamlit run app.py --server.fileWatcherType none
```

### Configuración Personalizada

Crea un archivo `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
enableCORS = false
```

## 🐛 Solución de Problemas

### Error: "Modelo no encontrado"
```bash
# Ejecutar pipeline de entrenamiento
cd ..
python main.py
```

### Error: "Streamlit no está instalado"
```bash
pip install streamlit
```

### Error: "ModuleNotFoundError: No module named 'config'"
```bash
# Asegúrate de estar en el directorio web/
cd web
streamlit run app.py
```

### La app no se abre automáticamente
Abre manualmente: http://localhost:8501

## 📝 Notas

- El modelo debe estar entrenado antes de usar la app
- Los reportes visuales se generan al ejecutar `main.py`
- La app carga el modelo una sola vez (caché de Streamlit)
- Las predicciones son instantáneas después de la carga inicial

## 🤝 Contribuciones

Mejoras sugeridas:
- [ ] Soporte multilenguaje (español, francés)
- [ ] Análisis de sentimientos multi-clase
- [ ] Explicabilidad (LIME/SHAP)
- [ ] Comparación lado a lado de modelos
- [ ] Exportar predicciones a CSV
- [ ] API REST endpoint

## 📚 Recursos

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorFlow/Keras](https://www.tensorflow.org/)
- [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Understanding LSTMs](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

**Desarrollado con ❤️ usando TensorFlow, Keras y Streamlit**
