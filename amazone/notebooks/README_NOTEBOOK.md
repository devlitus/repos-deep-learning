# 🎬 Sistema de Recomendación de Películas - Notebook Educativo

## Descripción

Este es un **notebook educativo completo** que explora cómo construir un **sistema de recomendación de películas** usando el dataset **MovieLens 100K** y múltiples técnicas de **Collaborative Filtering**.

El notebook integra todo el código del proyecto `src/` de forma pedagógica, con explicaciones detalladas en cada paso.

## 📋 Contenido del Notebook

### Parte 1: Carga y Exploración de Datos
- Importación de librerías y configuración
- Carga del dataset MovieLens 100K (943 usuarios, 1,682 películas, 100K ratings)
- Exploración básica: estadísticas, distribuciones
- Análisis de usuarios y películas más activas
- Visualización de patrones en los datos

**Conceptos clave:**
- Estructura de datos de ratings
- Distribución de preferencias de usuarios
- Películas populares vs. películas menos conocidas

### Parte 2: Análisis de Sparsidad
- ¿Qué es sparsidad? (93.7% de la matriz está vacía)
- Cálculo matemático de sparsidad
- Visualización de la matriz usuario-película
- Análisis de usuarios con pocos datos
- Análisis de películas con pocos ratings

**Conceptos clave:**
- Creación de matriz usuario-película (pivot)
- El **problema principal**: predecir celdas vacías
- Impacto de sparsity en calidad de recomendaciones

### Parte 3: User-Based Collaborative Filtering
- **Concepto**: "Si dos usuarios tienen gustos similares, les gustarán las mismas películas"
- Cálculo de **similitud del coseno** entre todos los usuarios
- Algoritmo: encontrar usuarios similares → ver qué vieron → predecir ratings
- **Fórmula de predicción**: promedio ponderado por similitud
- Prueba con usuario de ejemplo
- Visualización de matriz de similitudes

**Conceptos clave:**
- Similitud del coseno (0 = nada similar, 1 = idéntico)
- Promedio ponderado: $\hat{r}_{u,i} = \frac{\sum sim(u,v) \cdot r_{v,i}}{\sum sim(u,v)}$
- Limitaciones: sparsity, cold start, escalabilidad

### Parte 4: Item-Based Collaborative Filtering
- **Concepto**: "Si un usuario le gustó una película, le gustará una similar"
- Cálculo de similitud entre **películas** (no usuarios)
- Ventajas sobre User-Based: más estable, mayor similitud, más explicable
- Implementación: transposición de matriz → similitud → predicción
- Comparación con User-Based

**Conceptos clave:**
- Las películas no cambian, los usuarios sí → más estable
- "Porque te gustó X, te recomendamos Y" (explicabilidad)
- Mejor para sistemas con muchos usuarios

### Parte 5: Matrix Factorization (SVD)
- **Concepto**: Descomposición de matriz en factores latentes
- **Fórmula**: $R \approx U \times \Sigma \times V^T$
  - U = usuarios × factores latentes (943 × 50)
  - Σ = importancia de factores
  - V^T = factores latentes × películas (50 × 1,682)
- Normalización de matriz
- Aplicación de Singular Value Decomposition
- Reconstrucción de predicciones
- Visualización de varianza explicada y errores

**Conceptos clave:**
- Factores latentes: características no visibles capturadas automáticamente
- Generalmente mejor precisión que Collaborative Filtering
- Trade-off: precisión vs. interpretabilidad

### Parte 6: Sistema Híbrido
- **Combinación**: SVD (precisión) + Item-Based (explicabilidad)
- Recomendaciones con razones: "porque te gustó..."
- Ventajas de combinar métodos
- Comparación visual de los tres métodos

**Conceptos clave:**
- Sinergia de métodos
- Balanceo entre precisión y explicabilidad
- Mejor experiencia de usuario

### Conclusiones y Próximos Pasos
- Tabla comparativa de métodos
- Problemas principales en producción:
  - **Cold Start Problem**: usuarios/películas nuevas
  - **Sparsity**: datos faltantes
  - **Serendipity**: descubrimiento vs. predicción
- Métricas de evaluación (Precision@K, Recall@K, RMSE, NDCG)
- Pasos para producción

## 🚀 Cómo Usar

### Prerequisitos

1. **Tener el dataset descargado:**
   ```bash
   cd amazone
   python download_movielens.py
   ```

2. **Instalar dependencias:**
   ```bash
   cd amazone
   pip install -r requirements.txt
   ```

### Ejecutar el Notebook

**Opción 1: Jupyter Lab/Notebook**
```bash
cd amazone/notebooks
jupyter notebook amazone_sistema_recomendacion.ipynb
```

**Opción 2: VS Code con extensión de Jupyter**
- Abrir el archivo `.ipynb` en VS Code
- VS Code detectará automáticamente el kernel de Python

**Opción 3: Google Colab**
- Subir el notebook a Google Colab
- Modificar la ruta del dataset si es necesario

### Estructura de Ejecución

El notebook está diseñado para ser ejecutado **secuencialmente**:

1. **Celda 1-5**: Importaciones y configuración
2. **Celda 6-20**: Carga y exploración de datos
3. **Celda 21-30**: Análisis de sparsidad
4. **Celda 31-50**: User-Based CF
5. **Celda 51-65**: Item-Based CF
6. **Celda 66-90**: SVD / Matrix Factorization
7. **Celda 91-100**: Sistema Híbrido
8. **Celda 101-110**: Conclusiones

**⚠️ Nota**: No ejecutes celdas fuera de orden, ya que dependen de variables definidas anteriormente.

## 📊 Archivos Generados

El notebook genera visualizaciones en `amazone/reports/`:
- `exploratory_analysis.png` - Análisis inicial de datos
- `sparsity_analysis.png` - Visualización de sparsidad
- `user_based_cf.png` - Matriz de similitud entre usuarios
- `svd_analysis.png` - Análisis de descomposición SVD
- `comparison_methods.png` - Comparación de métodos

## 🎓 Conceptos Educativos Cubiertos

### Teoría de Sistemas de Recomendación
- Content-Based vs. Collaborative Filtering
- Métodos basados en similitud (user-based, item-based)
- Métodos basados en factorización de matriz
- Sistemas híbridos

### Matemática
- Similitud del coseno
- Promedio ponderado
- Singular Value Decomposition (SVD)
- Normalización de datos

### Python y Librerías
- Pandas: manipulación de datos
- NumPy: operaciones numéricas
- Scikit-learn: `cosine_similarity`
- Scipy: `svds` para descomposición
- Matplotlib & Seaborn: visualización

### Machine Learning
- Validación de datos
- Métricas de evaluación (RMSE, MAE)
- Hiperparámetros (k, número de factores)
- Normalización de datos

## 💡 Preguntas para Reflexionar

Mientras trabajas con el notebook, considera:

1. **Sparsity**: ¿Por qué la matrix tiene 93.7% de sparsity? ¿Cómo afecta esto a las recomendaciones?

2. **User-Based vs Item-Based**: ¿Cuándo usarías cada uno? ¿Qué tan diferente son las recomendaciones?

3. **SVD**: ¿Qué factores latentes está descubriendo el modelo? ¿Son interpretables?

4. **Cold Start**: ¿Cómo recomendarías películas a un usuario nuevo sin ratings?

5. **Precisión vs Explicabilidad**: ¿Vale la pena sacrificar precisión por explicabilidad?

6. **Evaluación**: ¿Cómo medirías si mi sistema de recomendación es "bueno"?

## 🔗 Relación con el Proyecto

Este notebook está integrado con el proyecto amazone completo:

```
amazone/
├── src/
│   ├── exploratory_analysis.py           ← Parte 1 del notebook
│   ├── sparsity_analysis.py              ← Parte 2 del notebook
│   ├── user_based_collaborative_filtering.py  ← Parte 3
│   ├── item_based_collaborative_filtering.py  ← Parte 4
│   ├── matrix_factorization_svd.py       ← Parte 5
│   ├── hybrid_recommender_system.py      ← Parte 6
│   └── exploratory_analysis.py
├── notebooks/
│   └── amazone_sistema_recomendacion.ipynb  ← Este notebook
└── web/
    └── app.py                            ← Aplicación Streamlit interactiva
```

## 🚀 Próximos Pasos

Después de trabajar con este notebook, puedes:

1. **Ejecutar el pipeline completo:**
   ```bash
   cd amazone
   python main.py
   ```

2. **Explorar la aplicación web interactiva:**
   ```bash
   cd amazone
   streamlit run web/app.py
   ```

3. **Experimenta modificando parámetros:**
   - Cambia `k` en User-Based CF (número de usuarios similares)
   - Cambia número de factores en SVD
   - Prueba con diferentes usuarios

4. **Implementa tus propias mejoras:**
   - Añade validación cruzada
   - Implementa otras métricas (Precision@K, NDCG)
   - Prueba con regularización en SVD
   - Crea un ensemble de métodos

## 📚 Recursos Adicionales

**Lectura recomendada:**
- "Collective Intelligence" - O'Reilly
- "Recommender Systems Handbook" - Springer
- Papers en ACM RecSys

**Datasets:**
- MovieLens (10K, 100K, 1M, 10M, 25M)
- Netflix Prize Dataset
- Last.fm dataset

## ❓ Troubleshooting

### "ModuleNotFoundError: No module named 'sklearn'"
```bash
pip install scikit-learn
```

### "FileNotFoundError: data/raw/ml-100k/u.data"
Descarga el dataset primero:
```bash
python download_movielens.py
```

### Notebook tarda mucho en ejecutarse
- La celda de User-Based CF (matriz 943×943) puede tardar 30-60 segundos
- La celda de SVD puede tardar 10-20 segundos
- Esto es normal, paciencia 😊

## 📝 Notas de Implementación

- **Simplificación en User-Based**: Rellenamos NaN con 0 para calcular similitud. En producción se usa mejor alternativas.
- **Normalización en SVD**: Restamos la media de usuario antes de SVD para mejorar predicciones.
- **Clipping de predicciones**: Las predicciones se limitan a [1, 5] para mantener rango válido de ratings.

## 👨‍🏫 Autor y Contexto

Este notebook es parte del repositorio educativo `repos-deep-learning` con múltiples proyectos de ML.

**Propósito**: Enseñanza y aprendizaje de sistemas de recomendación desde cero, pasando por diferentes técnicas hasta sistemas híbridos avanzados.

## 📄 Licencia

Los notebooks y código en este repositorio son para fines educativos.

---

**¡Disfruta aprendiendo sobre sistemas de recomendación!** 🎬✨
