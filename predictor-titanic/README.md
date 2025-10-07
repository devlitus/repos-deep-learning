# 🚢 Predictor de Supervivencia del Titanic

Proyecto de **Machine Learning de clasificación binaria** para predecir la supervivencia de pasajeros del Titanic utilizando Random Forest Classifier.

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de clasificación para predecir si un pasajero del Titanic sobrevivió o no, basándose en características como clase de ticket, edad, sexo, tarifa pagada, entre otras.

**Tipo de problema**: Clasificación Binaria  
**Algoritmo**: Random Forest Classifier  
**Dataset**: Titanic dataset de Seaborn  
**Objetivo**: Maximizar la precisión en la predicción de supervivencia

## 🎯 Características del Modelo

### Variables de Entrada (Features)

- **pclass**: Clase del ticket (1ra, 2da, 3ra clase)
- **sex**: Sexo del pasajero
- **age**: Edad del pasajero
- **fare**: Tarifa pagada
- **embarked**: Puerto de embarque (C = Cherbourg, Q = Queenstown, S = Southampton)
- **family_size**: Tamaño de la familia (calculado: sibsp + parch + 1)

# Predictor de Supervivencia del Titanic

Proyecto educativo de Machine Learning para clasificar si un pasajero del Titanic sobrevivió o no. Este README resume cómo instalar, ejecutar y entender el pipeline presente en `src/`.

Última actualización: 2025-10-07

## Resumen rápido

- Problema: Clasificación binaria (target: `survived`)
- Algoritmo: Random Forest (implementado en `train_model.py`)
- Dataset: `seaborn.load_dataset('titanic')` (no se requiere CSV local)

## Estructura principal

```
predictor-titanic/
├─ data/
├─ models/                  # Modelos guardados (.pkl)
├─ notebooks/
├─ reports/
├─ src/
│  ├─ data_loader.py
│  ├─ data_preprocessing.py
│  ├─ train_model.py
│  ├─ visualization.py
│  └─ alanizis_year.py
├─ config.py
├─ requirements.txt
└─ README.md
```

## Instalación rápida

Desde PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Dependencias clave: pandas, numpy, matplotlib, seaborn, scikit-learn.

## Ejecutar el pipeline (pasos comunes)

Recomendado: ejecutar desde la raíz del repo o desde `predictor-titanic/src`.

1. Exploración rápida:

```powershell
cd predictor-titanic\src
python data_loader.py
```

2. Preprocesamiento:

```powershell
python data_preprocessing.py
```

3. Visualizaciones:

```powershell
python visualization.py
```

4. Entrenar y evaluar (guarda modelo en `models/`):

```powershell
python train_model.py
```

## Resumen del preprocesamiento

Implementado en `data_preprocessing.py`:

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
