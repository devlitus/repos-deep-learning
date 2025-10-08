# Fraud Detection Web Application

Aplicación web refactorizada para la detección de fraude con Machine Learning.

## 📁 Estructura del Proyecto

```
web/
├── app.py                  # Aplicación principal (punto de entrada)
├── app_backup.py          # Backup de la versión original
│
├── pages/                  # Módulo de páginas
│   ├── __init__.py
│   ├── dashboard.py       # Dashboard con métricas
│   ├── prediction.py      # Predicción en tiempo real
│   ├── analytics.py       # Análisis del modelo
│   ├── data_explorer.py   # Exploración de datos
│   └── about.py           # Información del proyecto
│
├── styles/                 # Módulo de estilos CSS
│   ├── __init__.py
│   └── custom_css.py      # Estilos personalizados
│
└── utils/                  # Módulo de utilidades
    ├── __init__.py
    └── data_loader.py     # Carga de datos y modelos
```

## 🚀 Cómo Ejecutar

### Opción 1: Script directo
```bash
cd fraude-detection/web
streamlit run app.py
```

### Opción 2: Scripts de inicio
**Windows:**
```bash
cd fraude-detection/web
run.bat
```

**Linux/Mac:**
```bash
cd fraude-detection/web
chmod +x run.sh
./run.sh
```

## 📦 Módulos

### Pages (`pages/`)
Contiene todas las páginas de la aplicación:
- **dashboard.py**: Métricas principales y visualizaciones
- **prediction.py**: Predicción de fraude en tiempo real
- **analytics.py**: Análisis de rendimiento del modelo
- **data_explorer.py**: Exploración y visualización de datos
- **about.py**: Información sobre el proyecto

### Styles (`styles/`)
Estilos CSS personalizados para la aplicación:
- **custom_css.py**: Todos los estilos CSS (tema oscuro, cards, botones, etc.)

### Utils (`utils/`)
Funciones auxiliares:
- **data_loader.py**: Funciones para cargar datos y modelos con caché

## 🎨 Ventajas de la Refactorización

1. **Modularidad**: Cada página es un módulo independiente
2. **Mantenibilidad**: Más fácil de mantener y actualizar
3. **Reutilización**: Componentes reutilizables
4. **Legibilidad**: Código más limpio y organizado
5. **Escalabilidad**: Fácil agregar nuevas páginas o funcionalidades

## 🔧 Modificar la Aplicación

### Agregar una nueva página
1. Crear archivo en `pages/nueva_pagina.py`
2. Definir función `show_nueva_pagina()`
3. Importar en `pages/__init__.py`
4. Agregar en el radio selector de `app.py`

### Modificar estilos
Editar `styles/custom_css.py` con los nuevos estilos CSS

### Agregar utilidades
Crear nuevas funciones en `utils/` según necesidad
