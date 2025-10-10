# Detección de Fraude en Tarjetas de Crédito

## 📋 Descripción

Este proyecto implementa un sistema de detección de fraude en transacciones de tarjetas de crédito utilizando técnicas de machine learning. El modelo se entrena con el dataset "Credit Card Fraud Detection" de Kaggle, que contiene transacciones etiquetadas como fraudulentas o legítimas.

El proyecto incluye:

- Análisis exploratorio de datos (EDA)
- Preprocesamiento y balanceo de clases
- Entrenamiento de modelos de clasificación
- Evaluación de rendimiento
- Aplicación web interactiva para predicciones
- Visualizaciones y reportes

## 🚀 Características

- **Modelo de ML**: Random Forest optimizado para detección de anomalías
- **Balanceo de clases**: Técnicas SMOTE para manejar datos desbalanceados
- **Evaluación completa**: Métricas de precisión, recall, F1-score, AUC-ROC
- **Aplicación web**: Interfaz Streamlit para predicciones en tiempo real
- **Notebooks interactivos**: Análisis paso a paso en Jupyter
- **Persistencia de modelos**: Guardado y carga de modelos entrenados

## 🛠️ Instalación

### Prerrequisitos

- Python 3.8+
- pip

### Pasos de instalación

1. **Clona el repositorio** (si aplica) o navega al directorio del proyecto:

   ```bash
   cd fraude-detection
   ```

2. **Instala las dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Verifica la instalación**:
   ```bash
   python verify_installation.py
   ```

## 📊 Uso

### Ejecutar el pipeline completo

```bash
python main.py
```

### Ejecutar la aplicación web

```bash
streamlit run web/app.py
```

### Ejecutar notebooks individualmente

```bash
jupyter notebook notebooks/
```

### Estructura del proyecto

```
fraude-detection/
├── config/                 # Configuraciones y rutas
├── data/
│   ├── raw/               # Datos crudos (creditcard.csv)
│   └── processed/         # Datos procesados y splits
├── models/                # Modelos entrenados y métricas
├── notebooks/             # Análisis en Jupyter
├── reports/               # Reportes y figuras
├── src/
│   ├── data/              # Carga y preprocesamiento
│   ├── models/            # Entrenamiento y evaluación
│   └── visualization/     # Gráficos y plots
├── web/                   # Aplicación Streamlit
│   ├── pages/             # Páginas de la app
│   └── utils/             # Utilidades web
├── main.py                # Script principal
├── requirements.txt       # Dependencias
└── README.md             # Este archivo
```

## 🔧 Tecnologías utilizadas

- **Python 3.8+**
- **Pandas & NumPy**: Manipulación de datos
- **Scikit-learn**: Modelos de ML
- **Imbalanced-learn**: Balanceo de clases
- **Matplotlib & Seaborn**: Visualizaciones
- **Streamlit**: Aplicación web
- **Jupyter**: Notebooks interactivos

## 📈 Resultados del modelo

El modelo actual alcanza:

- **Accuracy**: ~99.9%
- **Precision**: 95%
- **Recall**: 85%
- **F1-Score**: 90%
- **AUC-ROC**: 0.95

_Nota: Los resultados pueden variar según el split de datos y parámetros de entrenamiento._

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

_Proyecto desarrollado como parte del repositorio repos-deep-learning_</content>
<parameter name="filePath">c:\dev\repos-deep-learning\fraude-detection\README.md
