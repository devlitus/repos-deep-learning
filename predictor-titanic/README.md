# 🚢 Predictor de Supervivencia del Titanic

Proyecto educativo de Machine Learning para clasificar si un pasajero del Titanic sobrevivió o no. Este proyecto sigue el **patrón arquitectónico estándar** definido en `.github/copilot-instructions.md`.

**Última actualización:** 7 de octubre de 2025

## 📋 Resumen rápido

- **Problema:** Clasificación binaria (target: `survived`)
- **Algoritmo:** Random Forest Classifier
- **Dataset:** `seaborn.load_dataset('titanic')` (891 pasajeros)
- **Accuracy:** 81.01%
- **Features principales:** Sexo (36%), Precio del ticket (21%), Edad (17%)

## 🎯 Estructura del Proyecto

```
predictor-titanic/
├─ main.py                  # 🚀 PUNTO DE ENTRADA - Pipeline completo
├─ config.py                # ⚙️ Configuración centralizada
├─ requirements.txt
├─ data/
├─ models/
│  └─ titanic_random_forest.pkl
├─ src/
│  ├─ data_loader.py        # Carga y exploración de datos
│  ├─ data_preprocessing.py # Limpieza y feature engineering
│  ├─ model.py              # Entrenamiento y evaluación
│  ├─ predictor.py          # Predicciones con modelo guardado
│  ├─ visualization.py      # Gráficos exploratorios
│  └─ app.py                # App interactiva (Streamlit)
└─ reports/
```

## 🚀 Instalación y Ejecución

### Instalación

```powershell
# Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar Pipeline Completo

```powershell
cd predictor-titanic
python main.py
```

Este comando ejecuta todo el flujo:

1. ✅ Carga de datos
2. ✅ Exploración inicial
3. ✅ Preprocesamiento
4. ✅ Entrenamiento del modelo
5. ✅ Evaluación con métricas
6. ✅ Guardado del modelo

### Hacer Predicciones

```powershell
python src/predictor.py
```

### Visualizaciones

```powershell
python src/visualization.py
```

### App Interactiva (Streamlit)

```powershell
streamlit run src/app.py
```

## 📊 Flujo de Trabajo Estándar

El proyecto sigue el patrón definido en las instrucciones:

```
load_data() → explore_data() → prepare_data() →
split_data() → train_model() → evaluate_model() →
save_model() → predict()
```

Cada módulo es **ejecutable independientemente** para testing:

```powershell
python src/data_loader.py        # Carga y explora datos
python src/data_preprocessing.py # Preprocesa y muestra resultados
python src/model.py              # Entrena y evalúa modelo
python src/predictor.py          # Hace predicciones de ejemplo
```

## 🧹 Preprocesamiento de Datos

Implementado en `data_preprocessing.py`:

1. **Manejo de valores faltantes:**

   - `age`: Rellenado con mediana (28 años)
   - `embarked`: Rellenado con moda ('S' - Southampton)

2. **Conversión de variables categóricas:**

   - `sex`: male → 0, female → 1
   - `embarked`: C → 0, Q → 1, S → 2

3. **Feature Engineering:**

   - `family_size`: sibsp + parch + 1
   - `is_alone`: 1 si viaja solo, 0 si no

4. **Features finales (9 variables):**
   - `pclass`, `sex`, `age`, `sibsp`, `parch`
   - `fare`, `embarked`, `family_size`, `is_alone`

## 🎯 Resultados del Modelo

### Métricas en Test Set

| Métrica       | Valor  |
| ------------- | ------ |
| **Accuracy**  | 81.01% |
| **Precision** | 79.66% |
| **Recall**    | 68.12% |
| **F1-Score**  | 0.7344 |

### Importancia de Variables

| Feature         | Importancia |
| --------------- | ----------- |
| **sex**         | 35.9%       |
| **fare**        | 20.5%       |
| **age**         | 17.3%       |
| **pclass**      | 11.9%       |
| **family_size** | 4.7%        |

## ⚙️ Configuración

Todos los parámetros están centralizados en `config.py`:

```python
# Hiperparámetros del modelo
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}

# División train/test
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

## 📦 Dependencias

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- streamlit (para app interactiva)
- joblib (para serialización del modelo)

## 🔍 Contexto Histórico

El RMS Titanic se hundió el 15 de abril de 1912. De los 2,224 pasajeros y tripulación, solo sobrevivieron 710 personas (32%).

### Factores de supervivencia:

- **Mujeres:** 74% de supervivencia
- **Hombres:** 19% de supervivencia
- **1ra Clase:** 63% de supervivencia
- **3ra Clase:** 24% de supervivencia

## 📝 Notas de Implementación

- ✅ Rutas absolutas configuradas en `config.py`
- ✅ Sin chained assignment (compatible con pandas 3.0)
- ✅ Uso de `joblib` para persistencia del modelo
- ✅ Output formateado para claridad educativa
- ✅ Reproducibilidad garantizada con `random_state=42`

## 🎓 Propósito Educativo

Este proyecto demuestra:

- Pipeline completo de ML de principio a fin
- Preprocesamiento robusto de datos
- Feature engineering básico
- Evaluación exhaustiva con múltiples métricas
- Persistencia y reutilización de modelos
- Arquitectura modular y mantenible

- Relleno de faltantes: `age` (mediana por `sex`+`pclass`), `embarked` (moda), `fare` (mediana).
- Feature engineering: `family_size = sibsp + parch + 1`; `is_alone`.
- Codificación: variables categóricas (`sex`, `embarked`) via dummies/one-hot.

Nota: evitar chained assignment con `inplace=True`; usar asignaciones directas para compatibilidad con pandas 2.x/3.x.

## Entrenamiento y persistencia

El entrenamiento usa RandomForestClassifier. Métricas calculadas: accuracy, precision, recall, f1 y matriz de confusión.
El modelo se guarda con `pickle` en `models/titanic_random_forest.pkl`.

Ejemplo de guardado:

```python
import pickle
with open('models/titanic_random_forest.pkl', 'wb') as f:
    pickle.dump(modelo, f)
```

## Errores comunes y soluciones

- No module named 'sklearn' → `pip install scikit-learn`
- No module named 'seaborn' → `pip install seaborn`
- Errores de import desde `src/`: ejecutar desde la raíz del proyecto o ajustar `PYTHONPATH`.

## Pendientes / mejoras sugeridas

- Completar `config.py` con rutas y parámetros (usar rutas absolutas basadas en `BASE_DIR`).
- Añadir `main.py` que orqueste el pipeline completo.
- Agregar `predictor.py` para cargar modelo y predecir nuevos casos.
- Implementar tests unitarios en `test/`.

---
