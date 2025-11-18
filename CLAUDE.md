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
cd [project-name]

# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python main.py

# Run Streamlit app (if available)
streamlit run web/app.py

# Launch Jupyter notebooks
jupyter notebook notebooks/
```

### Project-Specific Commands

**fraude-detection:**
```bash
cd fraude-detection

# Verify installation
python verify_installation.py

# Run web app (multi-page)
streamlit run web/app.py

# Build and test
python main.py
```

**prediccion-temperatura:**
```bash
cd prediccion-temperatura

# Train the model (with detailed logging)
python src/train.py

# Make predictions on new data
python src/predict.py

# Run Streamlit web app (includes visualizations)
streamlit run web/app.py
```

**amazone:**
```bash
cd amazone

# Download MovieLens dataset (first time setup)
python download_movielens.py

# Run main recommendation pipeline
python main.py
```

### Key Files to Read First

When working on a project:
1. **`main.py`** - Understand the complete pipeline flow
2. **`config.py`** or **`config/config.py`** - See all absolute paths and feature definitions
3. **`src/data_loader.py`** - How data is loaded and explored
4. **`src/model.py` or `src/train.py`** - Model training and evaluation logic
5. **Project-specific `README.md`** - Dataset details and performance metrics

## Critical Conventions

### Configuration & Paths

**ALWAYS use absolute paths via config.py:**

```python
# config.py pattern (most projects - using os.path.join)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_FILE = os.path.join(BASE_DIR, 'models', 'model_name.pkl')

# fraude-detection pattern (modern - using pathlib.Path)
from pathlib import Path
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / 'data' / 'raw'
MODEL_FILE = BASE_DIR / 'models' / 'model_name.pkl'

# In other files: Use config paths, NOT hardcoded strings
from config import MODEL_FILE, DATA_RAW_DIR
df = pd.read_csv(os.path.join(DATA_RAW_DIR, 'data.csv'))
```

**Config should contain:**
- All directory paths (BASE_DIR, DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR)
- Feature column names (FEATURES, TARGET, CATEGORICAL_COLS, NUMERICAL_COLS)
- Hyperparameters (RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE)
- Model-specific parameters (for LSTM: sequence_length, prediction_steps, etc.)

### Standard Module Functions

Each module provides consistent functions:

**`src/data_loader.py`:**
```python
def load_data(filepath) -> pd.DataFrame
def explore_data(df) -> None  # Print statistics, missing values, etc.
def prepare_data(df) -> pd.DataFrame  # Preprocessing, cleaning, feature engineering
```

**`src/model.py` or `src/train.py`:**
```python
def split_data(X, y) -> tuple  # (X_train, X_test, y_train, y_test, ...)
def train_model(X_train, y_train) -> Model  # Train and return model
def evaluate_model(model, X_test, y_test) -> dict  # Return metrics dict
```

**`src/predictor.py`:**
```python
def make_prediction(model, new_data) -> predictions  # Load model and predict
def predict_single(model, single_sample) -> prediction  # Single sample prediction
```

**`src/visualizations.py`:**
```python
def visualize_eda(df) -> None  # EDA plots
def visualize_feature_distributions(X) -> None
def visualize_predictions(y_true, y_pred) -> None
```

### Pandas 3.0 Compatibility

**Never use chained assignment with `inplace=True`:**

```python
# ✅ CORRECT - Use direct assignment
df['age'] = df['age'].fillna(value)

# ❌ WRONG - Causes FutureWarning
df['age'].fillna(value, inplace=True)
```

**Always strip column names when loading CSVs:**
```python
df.columns = df.columns.str.strip()
```

### Model Persistence

Projects use different serialization methods:
- **predictor-house**: `pickle` (`.pkl`)
- **predictor-titanic**: `pickle` (`.pkl`)
- **fraude-detection**: `joblib` (`.pkl`)
- **prediccion-temperatura**: `keras` (`.keras` format for LSTM)
- **amazone**: No persistent model (collaborative filtering computed on-demand)

**Example pattern:**
```python
import joblib
import os

def save_model(model, filepath):
    joblib.dump(model, filepath)  # Use absolute path
    print(f"✅ Modelo guardado: {os.path.basename(filepath)}")  # Show only filename

def load_model(filepath):
    model = joblib.load(filepath)
    print(f"✅ Modelo cargado: {os.path.basename(filepath)}")
    return model

# For Keras/TensorFlow (prediction-temperatura)
model.save(filepath)  # Saves as .keras format
model = keras.models.load_model(filepath)
```

### TensorFlow/Keras Specific (prediction-temperatura)

**Time series data handling:**
```python
from sklearn.preprocessing import MinMaxScaler

# Normalize time series data
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(data.reshape(-1, 1))

# Save scaler for later predictions (use save_scaler.py utility)
joblib.dump(scaler, scaler_path)
```

**LSTM model structure:**
```python
model = Sequential([
    LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
    LSTM(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
```

**Model callbacks for training:**
```python
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]
model.fit(X_train, y_train, callbacks=callbacks, validation_data=(X_val, y_val), epochs=100)
```

### Web Applications (Streamlit)

Projects with web apps: **fraude-detection**, **prediccion-temperatura**, **amazone**

**fraude-detection (multi-page):**
```
web/
├── app.py                    # Main entry point with sidebar navigation
├── pages/
│   ├── 1_Dashboard.py        # Overview metrics and visualizations
│   ├── 2_Prediction.py       # Model prediction interface
│   ├── 3_DataExplorer.py     # Interactive data exploration
│   ├── 4_Analytics.py        # Advanced analysis
│   └── 5_About.py            # Project information
├── utils/                    # Helper functions
└── styles/                   # CSS customization
```

**prediccion-temperatura:**
- Single-page app (web/app.py)
- See `WEB_APP_GUIDE.md` for detailed documentation
- Interactive forecasting and visualization

**Running web apps:**
```bash
streamlit run web/app.py

# fraude-detection also provides shell scripts
cd fraude-detection/web
./run.sh        # Linux/Mac
run.bat         # Windows
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
- **Solution**: SMOTE for synthetic oversampling in training

### prediccion-temperatura
- **Target**: Daily minimum temperature (Melbourne, 1981-1990)
- **Dataset**: 3,650 daily observations
- **Model**: 3-layer LSTM neural network
- **Preprocessing**: MinMaxScaler normalization, sequence creation
- **Key files**: `src/train.py`, `src/model.py`, `src/predict.py`

### amazone
- **Features**: User IDs, Movie IDs, Ratings
- **Dataset**: MovieLens 100K (100,000 ratings from 1,000 users on 1,700 movies)
- **Approach**: Multiple collaborative filtering methods (user-based, item-based, SVD)
- **Download**: Run `download_movielens.py` before first use

## Technology Stack

### Core ML Libraries
- **scikit-learn >= 1.3.0**: Classical ML (regression, Random Forest, metrics, preprocessing)
- **pandas >= 2.2.0**: Data manipulation (with 3.0 compatibility)
- **numpy >= 1.26.0**: Numerical computing
- **imbalanced-learn >= 0.12.0**: SMOTE & balancing techniques (fraude-detection)

### Deep Learning
- **TensorFlow >= 2.0**: Neural networks
- **Keras**: LSTM for time series (prediccion-temperatura)

### Visualization
- **matplotlib >= 3.8.0**: Core plotting
- **seaborn >= 0.13.0**: Statistical visualizations
- **plotly**: Interactive visualizations in Streamlit

### Web & Notebooks
- **streamlit >= 1.0**: Web applications for model interaction
- **jupyter**: Interactive notebooks for development

### Model Storage & Utilities
- **pickle**: Basic serialization (predictor-house, predictor-titanic)
- **joblib >= 1.3.0**: Better for sklearn models (all projects)
- **keras format**: Neural network models (.keras extension)
- **scipy**: Sparse matrix handling (amazone)

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

For categorical encoding:
```python
# Use pd.get_dummies or LabelEncoder, apply BEFORE train/test split
X = pd.get_dummies(X, columns=['categorical_col'], drop_first=True)
```

## Standard Pipeline Flow

The `main.py` in each project follows this order:

1. Load data → 2. Explore data → 3. Prepare/preprocess data → 4. Visualize features
5. Split train/test/validation → 6. Train model → 7. Evaluate metrics → 8. Visualize predictions
9. Save model → 10. Make predictions on new data

**Use `random_state=42` for all random operations** to ensure reproducibility:
```python
from sklearn.model_selection import train_test_split
train_test_split(..., random_state=42)

from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier(..., random_state=42)
```

## Important Files Not to Modify

- `data/raw/` - Original datasets (work on copies)
- `.github/copilot-instructions.md` - AI agent guidelines (230+ lines of additional patterns)
- `.gitignore` - Excludes .venv, __pycache__, .pyc, model files
- Repository README.md - Master documentation across all projects

## Utility Scripts

**fraude-detection/verify_installation.py:**
```bash
python verify_installation.py  # Checks all dependencies are installed
```

**amazone/download_movielens.py:**
```bash
python download_movielens.py   # Downloads MovieLens 100K dataset
```

**prediccion-temperatura/save_scaler.py:**
```bash
python save_scaler.py  # Persists MinMaxScaler for future predictions
```

## Common Errors to Avoid

1. **Hardcoded relative paths** → Always use `config.py` with `BASE_DIR`
2. **Chained assignment with inplace** → Use direct assignment for pandas 3.0 compatibility
3. **Wrong working directory** → Always `cd` to project dir before running `python main.py`
4. **Modifying raw data** → Always `df.copy()` before making changes
5. **Missing column strip** → Always do `df.columns = df.columns.str.strip()` after CSV load
6. **Model path issues** → Use `os.path.basename()` for display messages, full path for loading
7. **Not setting random_state** → Must use `random_state=42` everywhere for reproducibility
8. **Forgetting to save scaler** → For time series models, save scaler with model for inference
9. **Training on full dataset** → Always split data before training (never fit scaler on test data)
10. **Different serialization methods** → Use correct format per project (pickle vs joblib vs keras)

## Git Status

- **Current branch**: main (up to date)
- **Main branch for PRs**: main
- **Recent changes**: LSTM model improvements (prediccion-temperatura)
- **Untracked**: `amazone/notebooks/amazon.ipynb`

## Exploring the Codebase

For new projects in this repo:
- Start with **`main.py`** to understand the pipeline
- Check **`config.py`** (or **`config/config.py`** for fraude-detection) for configuration
- Review **`src/data_loader.py`** to understand data handling
- Look at **`src/model.py`** or **`src/train.py`** for model implementation
- Review **`.github/copilot-instructions.md`** for detailed extended patterns (Spanish language)
- Run **`python main.py`** to see the complete pipeline in action
- For web apps, check **`web/app.py`** and **`WEB_APP_GUIDE.md`** (if available)
