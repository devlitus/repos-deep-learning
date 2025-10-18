# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an **educational ML repository** containing 5 independent machine learning projects with complete end-to-end pipelines:

1. **predictor-house**: Regression (linear & Random Forest) for house price prediction
2. **predictor-titanic**: Binary classification for Titanic survival prediction
3. **fraude-detection**: Imbalanced classification with SMOTE for fraud detection (most advanced)
4. **amazone**: Recommendation system using collaborative filtering
5. **prediccion-temperatura**: Time series prediction with LSTM neural networks

All projects follow a consistent modular architecture.

## Project Architecture Pattern

Every project follows this identical structure:

```
[project-name]/
├── data/
│   ├── raw/                    # Original datasets (never modify)
│   └── processed/              # Cleaned/preprocessed data
├── src/                        # Core modules
│   ├── data_loader.py          # Load & explore data
│   ├── model.py / train.py     # Train & evaluate models
│   ├── predictor.py            # Make predictions with trained models
│   ├── visualizations.py       # EDA & plotting
│   └── [data/, models/]        # Submodules for complex projects
├── models/                     # Serialized models (.pkl, .keras, joblib)
├── reports/                    # Generated visualizations & metrics (.png, .csv, .json)
├── notebooks/                  # Jupyter notebooks (fraude-detection has 4)
├── web/                        # Streamlit apps (optional)
├── config.py                   # Absolute paths & configuration
├── main.py                     # Complete pipeline entry point
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Common Development Tasks

### Running Projects

```bash
# Navigate to project directory first
cd fraude-detection

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py

# Run Streamlit app (if available)
streamlit run web/app.py

# Launch Jupyter notebooks
jupyter notebook notebooks/
```

### Key Files to Read First

When working on a project:
1. **`main.py`** - Understand the complete pipeline flow
2. **`config.py`** - See all absolute paths and feature definitions
3. **`src/data_loader.py`** - How data is loaded and explored
4. **`src/model.py`** - Model training and evaluation logic

## Critical Conventions

### Configuration & Paths

**ALWAYS use absolute paths via config.py:**

```python
# config.py pattern
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'model_name.pkl')

# In other files: Use config paths, NOT hardcoded strings
from config import MODEL_FILE, DATA_RAW_DIR
df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'data.csv'))
```

### Pandas 3.0 Compatibility

**Never use chained assignment with `inplace=True`:**

```python
# ✅ CORRECT - Use direct assignment
df['age'] = df['age'].fillna(value)

# ❌ WRONG - Causes FutureWarning
df['age'].fillna(value, inplace=True)
```

Always strip column names when loading CSVs:
```python
df.columns = df.columns.str.strip()
```

### Model Persistence

Projects use different serialization methods:
- **predictor-house**: `pickle`
- **predictor-titanic, fraude-detection**: `joblib`
- **prediccion-temperatura**: `keras` (.keras format)

Example pattern:
```python
import joblib
import os

def save_model(model, filepath):
    joblib.dump(model, filepath)  # Use absolute path
    print(f"�� Modelo guardado: {os.path.basename(filepath)}")  # Show only filename

def load_model(filepath):
    model = joblib.load(filepath)
    print(f"✅ Modelo cargado: {os.path.basename(filepath)}")
    return model
```

### Output Formatting

Use structured formatting for clarity (important for educational purposes):

```python
print("=" * 60)
print("🧹 INICIANDO PREPROCESAMIENTO DE DATOS")
print("=" * 60)
print("\n📋 PASO 1: Manejo de valores faltantes")
print("-" * 60)
print(f"✅ Edad: Rellenados {count} valores con mediana ({median:.1f} años)")
```

## Datasets & Features

### predictor-house
- **Features**: `tamano_m2`, `habitaciones`, `banos`, `edad_anos`, `distancia_centro_km`
- **Target**: `precio`
- **Source**: Local CSV + Kaggle House Prices dataset

### predictor-titanic
- **Features**: `pclass`, `sex`, `age`, `fare`, `embarked`, `family_size`, `is_alone`
- **Target**: `survived`
- **Source**: `sns.load_dataset('titanic')`

### fraude-detection
- **Features**: `Time`, `V1-V28` (PCA-transformed), `Amount`
- **Target**: `Class` (0=legit, 1=fraud)
- **Challenge**: Highly imbalanced (0.17% fraud)
- **Solution**: SMOTE for synthetic oversampling

### prediccion-temperatura
- **Target**: Daily minimum temperature
- **Dataset**: 10 years Melbourne temperatures (1981-1990)
- **Model**: 3-layer LSTM neural network
- **Key files**: `src/model.py`, reports with metrics & visualizations

## Technology Stack

### Core ML Libraries
- **scikit-learn**: Classical ML (regression, Random Forest, metrics)
- **pandas >= 2.2.0**: Data manipulation
- **numpy >= 1.26.0**: Numerical computing
- **imbalanced-learn >= 0.12.0**: SMOTE & balancing techniques

### Deep Learning
- **TensorFlow/Keras**: LSTM for time series (prediccion-temperatura)

### Visualization
- **matplotlib >= 3.8.0**: Core plotting
- **seaborn >= 0.13.0**: Statistical visualizations
- **plotly**: Interactive visualizations in Streamlit

### Web & Notebooks
- **Streamlit**: Multi-page web apps (fraude-detection/web/)
- **Jupyter**: Interactive notebooks

### Model Storage
- **pickle**: Basic serialization (predictor-house)
- **joblib >= 1.3.0**: Better for sklearn models
- **keras format**: Neural network models

## Feature Engineering Patterns

Always create derived features BEFORE feature selection:

```python
# Create family_size and is_alone BEFORE selecting columns
df['family_size'] = df['sibsp'] + df['parch'] + 1
df['is_alone'] = (df['family_size'] == 1).astype(int)

# Then select features
features = ['pclass', 'sex', 'age', 'family_size', 'is_alone']
X = df[features]
```

## Standard Pipeline Flow

The `main.py` in each project follows this order:

1. Load data → 2. Explore data → 3. Prepare/preprocess data → 4. Visualize features
5. Split train/test/validation → 6. Train model → 7. Evaluate metrics → 8. Visualize predictions
9. Save model → 10. Make predictions on new data

Use `random_state=42` for all random operations to ensure reproducibility.

## Important Files Not to Modify

- `data/raw/` - Original datasets (work on copies)
- `.github/copilot-instructions.md` - AI agent guidelines (230+ lines of additional patterns)
- `.gitignore` - Excludes .venv, __pycache__, .pyc

## Git Status

- **Current branch**: main
- **Main branch for PRs**: master
- **Recent changes**: LSTM model improvements (prediccion-temperatura)

## Common Errors to Avoid

1. **Hardcoded relative paths** → Always use `config.py` with `BASE_DIR`
2. **Chained assignment with inplace** → Use direct assignment for pandas 3.0
3. **Wrong working directory** → `cd` to project dir before running
4. **Modifying raw data** → Always `df.copy()` before changes
5. **Missing column strip** → Always do `df.columns = df.columns.str.strip()`
6. **Model path issues** → Use `os.path.basename()` for display, full path for loading

## Exploring the Codebase

For new projects in this repo:
- Start with **`main.py`** to understand the pipeline
- Check **`config.py`** for all configuration and paths
- Review **`.github/copilot-instructions.md`** for detailed patterns (Spanish language)
- Look at **`src/`** modules for implementation details
- Run **`python main.py`** to see the complete pipeline in action
