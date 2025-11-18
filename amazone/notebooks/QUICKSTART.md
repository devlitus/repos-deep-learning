# 🚀 Guía Rápida - Notebook de Sistemas de Recomendación

## Inicio en 3 Pasos

### 1. Descargar Dataset
```bash
cd amazone
python download_movielens.py
```
✅ Descarga MovieLens 100K (943 usuarios, 1,682 películas, 100K ratings)

### 2. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 3. Abrir Notebook
```bash
cd amazone/notebooks
jupyter notebook amazone_sistema_recomendacion.ipynb
```
O en VS Code: `Ctrl+Shift+P` → "Jupyter: Open Notebook"

---

## 📖 Estructura del Notebook

| Parte | Tema | Tiempo | Conceptos Clave |
|-------|------|--------|-----------------|
| 1 | Carga y Exploración | 5 min | Dataset, estadísticas |
| 2 | Sparsidad | 10 min | Problema central: 93.7% vacío |
| 3 | User-Based CF | 15 min | Usuarios similares |
| 4 | Item-Based CF | 10 min | Películas similares |
| 5 | SVD / Matrix Fact. | 20 min | Factores latentes |
| 6 | Sistema Híbrido | 10 min | Combinación de métodos |

**Total: ~70 minutos** (depende de tu ritmo)

---

## 🎯 Qué Apenderás

### Teoría
- ✅ Qué es un sistema de recomendación
- ✅ Por qué son difíciles (sparsidad, cold start)
- ✅ Cómo funcionan 3 métodos diferentes
- ✅ Cómo combinarlos para mejores resultados

### Práctica
- ✅ Cargar y explorar datos con Pandas
- ✅ Calcular similitud entre usuarios/películas
- ✅ Implementar predicción de ratings
- ✅ Usar SVD de scipy
- ✅ Crear un sistema híbrido

### Visualización
- ✅ 4 gráficos de análisis exploratorio
- ✅ Matriz de similaridad heatmap
- ✅ Distribuciones y varianza
- ✅ Comparación de métodos

---

## 💻 Usar el Notebook

### Ejecución Secuencial
```
✅ Ejecuta celdas de arriba a abajo (Shift+Enter)
⚠️ No saltes celdas - tienen dependencias
💡 Lee los comentarios en cada celda
```

### Modificar y Experimentar
```python
# Cambiar usuario de prueba
test_user_id = 5  # Prueba con otro usuario

# Cambiar número de usuarios similares
k_users = 30  # En lugar de 20

# Cambiar número de factores en SVD
k = 100  # En lugar de 50 (más lento pero más preciso)
```

### Visualizaciones
Se guardan automáticamente en `amazone/reports/`:
```
📊 exploratory_analysis.png
📊 sparsity_analysis.png
📊 user_based_cf.png
📊 svd_analysis.png
📊 comparison_methods.png
```

---

## 🎬 Usuario de Ejemplo

El notebook usa **Usuario ID 1** de forma predeterminada:

**Sus películas favoritas (ratings altos):**
- Star Wars
- Terminator 2
- Alien
- Jurassic Park

**Recomendaciones de cada método:**
- **User-Based**: Películas que usuarios similares vieron
- **Item-Based**: Películas similares a las que le gustaron
- **SVD**: Predicciones basadas en patrones latentes

---

## 🤔 Preguntas Frecuentes

**P: ¿Cuánto tarda en ejecutarse?**
R: Depende de tu computadora. Las partes más lentas (similitud entre usuarios) tardan 30-60 segundos.

**P: ¿Puedo usar en Colab?**
R: Sí, pero cambiar rutas de datos. Ve a "Próximos Pasos" en README_NOTEBOOK.md

**P: ¿Necesito saber Pandas/NumPy de antemano?**
R: No necesario. El notebook está diseñado para ser autodidáctico. Si tienes dudas, Google es tu amigo 😊

**P: ¿Puedo modificar el código?**
R: ¡Absolutamente! Experimenta, rompe cosas, aprende. Los mejores aprendizajes vienen de probar.

**P: ¿Hay un vídeo?**
R: No aún, pero hay un README completo en README_NOTEBOOK.md

---

## 📊 Ejemplo de Salida

```
🎬 Top 10 Recomendaciones para Usuario 1:
==================================================

1. Saving Private Ryan
   Rating predicho: 4.87 ⭐
   Razón: Te gustó 'Forrest Gump' (5⭐)

2. The Sixth Sense
   Rating predicho: 4.65 ⭐
   Razón: Te gustó 'The Shawshank Redemption' (5⭐)

3. Titanic
   Rating predicho: 4.52 ⭐
   Razón: Te gustó 'Braveheart' (4⭐)

... y 7 más
```

---

## 🔧 Personalización

### Para Profundizar
- Aumenta `k` en SVD (de 50 a 100) para más precisión
- Prueba con múltiples usuarios (`for user_id in [1, 5, 10, 50]`)
- Implementa validación cruzada

### Para Hacer Más Rápido
- Reduce tamaño de muestra (primeros 500 usuarios)
- Reduce `k` en similitud (de 20 a 10)
- Usa matriz dispersa en lugar de densa

### Para Aprender Más
- Lee el código en `src/*.py`
- Modifica fórmulas de predicción
- Implementa tus propias métricas

---

## 🚀 Próximos Pasos

Después del notebook:

1. **Ejecuta el pipeline completo:**
   ```bash
   cd amazone && python main.py
   ```

2. **Intenta la app web:**
   ```bash
   cd amazone && streamlit run web/app.py
   ```

3. **Explora el código fuente:**
   - `src/user_based_collaborative_filtering.py`
   - `src/matrix_factorization_svd.py`
   - `web/app.py` (aplicación interactiva)

4. **Implementa mejoras:**
   - Normalización normalizada
   - Regularización en SVD
   - Contenido-basado CF
   - Cold start handling

---

## ✨ Tips Finales

- **Lee los comentarios**: Explican el "por qué", no solo el "qué"
- **Juega con parámetros**: Cambiar k, factores, usuarios - ver qué pasa
- **Visualiza resultados**: Las gráficas ayudan a entender el comportamiento
- **Toma notas**: Escribe lo que aprendes, lo solidifica

---

**¡Diviértete aprendiendo sistemas de recomendación!** 🎬✨

Para más detalles: Ver `README_NOTEBOOK.md`
