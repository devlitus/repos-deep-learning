# 🌐 Guía de Aplicación Web - Predictor de Temperatura

## Información General

Se ha creado una **aplicación web interactiva** usando Streamlit para hacer predicciones de temperatura de forma fácil y visual.

**Ubicación:** `prediction-temperature/web/app.py`

---

## ¿Qué es lo que se creó?

### Archivo Principal
- **`web/app.py`** - Aplicación Streamlit completa con interface interactiva

### Funcionalidades

#### 1. **Entrada de Datos** (3 métodos)
- **Valores Individuales**: Sliders interactivos para 60 días
- **Texto Pegado**: Ingresa temperaturas separadas por espacios/comas
- **Datos de Demostración**: Genera valores realistas automáticamente

#### 2. **Predicción**
- Predice temperatura para el día 61
- Muestra rango de confianza (±2.23°C basado en RMSE)
- Visualización inmediata

#### 3. **Visualizaciones**
- Gráfica de últimos 60 días
- Comparación de predicción vs promedio
- Estadísticas (mín, máx, promedio, desviación estándar)
- Análisis de cambio de temperatura

#### 4. **Información Educativa**
- Explicación del modelo LSTM
- ¿Por qué 60 días?
- Casos de uso prácticos
- Métricas de desempeño

---

## Cómo Ejecutar

### Requisitos Previos
```bash
# Asegúrate de tener instalado:
- Python 3.8+
- Modelo entrenado: models/lstm_temperatura.keras
- Scaler: models/scaler.pkl
```

### Paso 1: Instalar Dependencias
```bash
cd prediction-temperature
pip install -r requirements.txt
```

### Paso 2: Generar el Scaler (si no existe)
```bash
python save_scaler.py
```

Verificar que ambos archivos existan:
```bash
ls models/
# Debe mostrar:
# - lstm_temperatura.keras (1.7 MB)
# - scaler.pkl (522 bytes)
```

### Paso 3: Ejecutar la Aplicación
```bash
streamlit run web/app.py
```

### Resultado
La aplicación se abrirá en tu navegador en `http://localhost:8501`

---

## Funcionalidades de la Interfaz

### Tab 1: Entrada Manual
```
┌─────────────────────────────────────────┐
│  Elige método de entrada:               │
│  - Valores individuales (60 sliders)    │
│  - Texto pegado (copy-paste)            │
│  - Datos de demostración                │
│                                          │
│  Visualiza:                             │
│  - Gráfica de temperaturas              │
│  - Estadísticas (min, max, promedio)    │
│                                          │
│  [Botón Predecir]                       │
│                                          │
│  Resultado:                             │
│  - Temperatura predicha                 │
│  - Rango de confianza                   │
│  - Comparación con promedio             │
└─────────────────────────────────────────┘
```

### Tab 2: Datos de Demostración
```
┌─────────────────────────────────────────┐
│  Automáticamente genera datos realistas │
│  basados en patrones estacionales       │
│                                          │
│  [Botón Predecir con Demostración]     │
│                                          │
│  Resultado instantáneo con análisis     │
└─────────────────────────────────────────┘
```

### Tab 3: Información
```
┌─────────────────────────────────────────┐
│  Información educativa:                 │
│  - ¿Qué es LSTM?                        │
│  - ¿Por qué 60 días?                    │
│  - Rendimiento del modelo               │
│  - Dataset usado                        │
│  - Pipeline completo                    │
│  - Casos de uso prácticos               │
└─────────────────────────────────────────┘
```

### Sidebar
```
┌──────────────────────────────┐
│  INFORMACIÓN DEL MODELO      │
│                              │
│  ✓ Arquitectura LSTM        │
│  ✓ Métricas (RMSE, R²)      │
│  ✓ Dataset (10 años)        │
│  ✓ Cómo usar (pasos)        │
└──────────────────────────────┘
```

---

## Ejemplo de Uso

### Escenario: Predecir temperatura para mañana

**Paso 1: Ingresar datos**
```
Últimos 60 días de temperaturas:
15°C, 16°C, 17°C, ..., 22°C
```

**Paso 2: Hacer clic en "Predecir Temperatura del Día 61"**

**Paso 3: Resultado**
```
🌡️ Temperatura Predicha:        18.5°C
📊 Rango (±RMSE 2.23°C):        16.3°C a 20.7°C
🎯 Confianza del Modelo:        70.6%

Comparación:
- Promedio últimos 60 días:     16.2°C
- Predicción día 61:             18.5°C
- Cambio esperado:               +2.3°C
```

---

## Estructura de Archivos

```
prediction-temperature/
├── web/
│   └── app.py                    ← APLICACIÓN WEB (NUEVO)
│
├── models/
│   ├── lstm_temperatura.keras    (existe)
│   └── scaler.pkl               (generado por save_scaler.py)
│
├── save_scaler.py               ← SCRIPT PARA GENERAR SCALER (NUEVO)
│
├── requirements.txt             (actualizado)
│
└── README.md                    (actualizado)
```

---

## Archivos Modificados/Creados

### NUEVOS
✅ `web/app.py` - Aplicación Streamlit completa
✅ `save_scaler.py` - Script para generar scaler.pkl
✅ `WEB_APP_GUIDE.md` - Esta guía

### ACTUALIZADOS
✅ `requirements.txt` - Agregadas: streamlit, plotly
✅ `README.md` - Agregada sección sobre web app

---

## Troubleshooting

### Error: "No se pudo cargar el modelo"
```bash
# Verifica que el modelo existe
ls models/lstm_temperatura.keras

# Si no existe, ejecuta:
cd ..
python train.py
cd web
```

### Error: "ModuleNotFoundError: No module named 'streamlit'"
```bash
# Instala las dependencias
pip install -r requirements.txt
```

### Error: "UnicodeEncodeError" (Windows)
```bash
# La aplicación está diseñada para evitar esto, pero si ocurre:
set PYTHONIOENCODING=utf-8
streamlit run web/app.py
```

### Puerto 8501 ya está en uso
```bash
# Usar otro puerto
streamlit run web/app.py --server.port 8502
```

---

## Características de Usabilidad

### ✅ Interfaz Intuitiva
- No requiere conocimiento técnico
- Entrada clara de datos
- Resultados visuales inmediatos

### ✅ Múltiples Formas de Entrada
- Sliders (fácil)
- Texto (rápido)
- Datos de demostración (exploración)

### ✅ Visualizaciones
- Gráficas interactivas
- Estadísticas en tiempo real
- Comparaciones útiles

### ✅ Información Educativa
- Explica qué es LSTM
- Muestra métricas del modelo
- Describe casos de uso

### ✅ Rendimiento
- Predicciones instantáneas
- Interfaz responsiva
- Cache de recursos

---

## Casos de Uso

### 1. Agricultura
```
Predecir temperatura → Planificar riego
Detectar riesgo de helada → Proteger cultivos
```

### 2. Energía
```
Predecir temperatura → Estimar demanda calefacción
Optimizar generación de energía
```

### 3. Eventos
```
Predecir temperatura → Planificar actividades
Preparar contingencias
```

### 4. Educación
```
Entender cómo funciona LSTM
Aprender sobre series temporales
Experimentar con Machine Learning
```

---

## Limitaciones Conocidas

### ⚠️ Precisión
- RMSE: 2.23°C (±2-3 grados de error)
- No es apta para decisiones críticas
- Diseñada para estimaciones general

### ⚠️ Datos
- Solo funciona con 60 temperaturas
- Necesita formato consistente
- Basada en patrón Melbourne 1981-1990

### ⚠️ Modelo
- Entrenado con datos históricos
- Cambios climáticos no se capturan
- Solo predice 1 día adelante

---

## Mejoras Futuras Posibles

### Corto Plazo
- [ ] Exportar predicciones a CSV
- [ ] Historial de predicciones
- [ ] Comparación con datos reales

### Mediano Plazo
- [ ] Múltiples ciudades
- [ ] Predicción de 7+ días
- [ ] Agregar más características (humedad, presión)

### Largo Plazo
- [ ] API REST para integración
- [ ] Base de datos de predicciones
- [ ] Dashboard comparativo de modelos

---

## Comandos Rápidos

```bash
# Entrar al directorio
cd c:\dev\repos-deep-learning\prediction-temperature

# Instalar dependencias
pip install -r requirements.txt

# Generar scaler (si es necesario)
python save_scaler.py

# Ejecutar aplicación
streamlit run web/app.py

# Ejecutar con puerto diferente
streamlit run web/app.py --server.port 8502

# Ver logs
streamlit run web/app.py --logger.level=debug
```

---

## Contacto y Soporte

Si tienes problemas:
1. Verifica que Python 3.8+ está instalado
2. Asegúrate de tener todas las dependencias
3. Comprueba que modelo y scaler existen en `models/`
4. Revisa el README.md del proyecto

---

**¡La aplicación está lista para usar!**

Ejecuta: `streamlit run web/app.py`
