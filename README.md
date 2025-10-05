# 🏠 Predictor de Precios de Casas

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub repo](https://img.shields.io/badge/GitHub-repos--deep--learning-181717?logo=github)](https://github.com/devlitus/repos-deep-learning)

Un proyecto de machine learning para predecir precios de viviendas usando regresión lineal múltiple. Este sistema analiza características como tamaño, ubicación y antigüedad para estimar el valor de mercado de propiedades inmobiliarias.

## 🎯 Características del Proyecto

- **Modelo de Machine Learning**: Regresión lineal múltiple con scikit-learn
- **Análisis de Datos**: Exploración y visualización de patrones inmobiliarios
- **Pipeline Automatizado**: Desde carga de datos hasta predicciones finales
- **Visualizaciones**: Gráficos de análisis de características y precisión del modelo
- **Predicciones en Tiempo Real**: Sistema para evaluar nuevas propiedades

## 📁 Estructura del Proyecto

```
predictor-house/
├── data/
│   ├── raw/                    # Datos originales
│   │   └── casas.csv
│   └── processed/              # Datos procesados
├── src/                        # Código fuente principal
│   ├── data_loader.py         # Carga y limpieza de datos
│   ├── model.py               # Entrenamiento y evaluación
│   ├── predictor.py           # Sistema de predicciones
│   └── visualizations.py     # Gráficos y análisis visual
├── models/                     # Modelos entrenados
│   └── modelo_casas.pkl
├── reports/                    # Reportes y visualizaciones
│   └── figures/
│       ├── feature_analysis.png
│       └── predictions_vs_actual.png
├── tests/                      # Tests unitarios
├── main.py                     # Script principal
├── config.py                   # Configuraciones del proyecto
├── requirements.txt            # Dependencias
└── README.md                   # Documentación
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Instalación

1. **Clonar el repositorio**

   ```bash
   git clone <url-del-repositorio>
   cd repos-deep-learning/predictor-house
   ```

2. **Crear entorno virtual (recomendado)**

   ```bash
   python -m venv venv
   # En Windows
   venv\Scripts\activate
   # En Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Uso del Sistema

### Ejecutar el Pipeline Completo

```bash
python main.py
```

Este comando ejecutará todo el proceso:

1. ✅ Carga y exploración de datos
2. ✅ Preparación y limpieza
3. ✅ Análisis visual de características
4. ✅ División de datos (entrenamiento/prueba)
5. ✅ Entrenamiento del modelo
6. ✅ Evaluación y métricas
7. ✅ Generación de visualizaciones
8. ✅ Guardado del modelo
9. ✅ Predicciones de ejemplo

### Predicciones Personalizadas

```python
from src.predictor import predict_new_houses
import pickle

# Cargar modelo entrenado
with open('models/modelo_casas.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Predecir nuevas casas
# Formato: [tamaño_m2, habitaciones, baños, edad_años, distancia_centro_km]
casas_nuevas = [
    [130, 3, 2, 10, 5.0],  # Casa de 130m², 3 hab, 2 baños, 10 años, 5km del centro
    [200, 5, 4, 1, 2.0],   # Casa de 200m², 5 hab, 4 baños, 1 año, 2km del centro
]

predict_new_houses(modelo, casas_nuevas)
```

## 📊 Características del Dataset

El modelo utiliza las siguientes variables para realizar predicciones:

| Variable              | Descripción                               | Tipo     | Rango             |
| --------------------- | ----------------------------------------- | -------- | ----------------- |
| `tamano_m2`           | Superficie en metros cuadrados            | Numérico | 70-200 m²         |
| `habitaciones`        | Número de habitaciones                    | Entero   | 1-5               |
| `banos`               | Número de baños                           | Entero   | 1-4               |
| `edad_anos`           | Antigüedad de la propiedad                | Entero   | 1-30 años         |
| `distancia_centro_km` | Distancia al centro de la ciudad          | Decimal  | 1.8-18.0 km       |
| `precio`              | Precio de la vivienda (variable objetivo) | Numérico | $145,000-$620,000 |

## 🔬 Rendimiento del Modelo

El modelo de regresión lineal múltiple proporciona:

- **R² Score**: Coeficiente de determinación para evaluar la precisión
- **MAE**: Error absoluto medio en dólares
- **RMSE**: Raíz del error cuadrático medio
- **Visualizaciones**: Gráficos de predicciones vs valores reales

## 📈 Visualizaciones Incluidas

1. **Análisis de Características** (`feature_analysis.png`)

   - Distribución de variables
   - Correlaciones entre características
   - Patrones de precios por zona

2. **Precisión del Modelo** (`predictions_vs_actual.png`)
   - Predicciones vs valores reales
   - Línea de regresión ideal
   - Distribución de errores

## 🛠️ Tecnologías Utilizadas

- **Python 3.8+**: Lenguaje de programación principal
- **pandas 2.3.2**: Manipulación y análisis de datos
- **numpy 2.3.3**: Computación numérica
- **scikit-learn 1.7.2**: Algoritmos de machine learning
- **matplotlib 3.10.6**: Visualización de datos
- **jupyter 1.1.1**: Notebooks para exploración interactiva

## 🧪 Testing

Ejecutar tests unitarios:

```bash
python -m pytest tests/
```

## 📝 Configuración

Las configuraciones principales se encuentran en `config.py`:

- **Rutas de archivos**: Datos, modelos, reportes
- **Parámetros del modelo**: Tamaño de test, semilla aleatoria
- **Variables del dataset**: Características y variable objetivo

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para contribuir:

1. Fork del repositorio
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit de cambios (`git commit -am 'Añadir nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 👨‍💻 Autor

**Deep Learning Repository**

- Proyecto educativo de predicción de precios inmobiliarios
- Implementación de pipeline de machine learning completo

---

### 🔮 Próximas Mejoras

- [ ] Implementar validación cruzada
- [ ] Añadir más algoritmos de ML (Random Forest, XGBoost)
- [ ] API REST para predicciones en tiempo real
- [ ] Dashboard interactivo con Streamlit
- [ ] Análisis de importancia de características
- [ ] Detección y manejo de outliers
- [ ] Integración con datos inmobiliarios en tiempo real

---

**¿Tienes preguntas?** Abre un issue en el repositorio o contacta al equipo de desarrollo.
