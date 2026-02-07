# 🧹 Guía de Limpieza del Proyecto

## Archivos que PUEDES ELIMINAR (Opcional)

Si solo quieres aprender **redes neuronales** y no te interesan los métodos tradicionales, puedes eliminar estos archivos de forma segura:

### 📁 Directorio `src/` - Modelos Tradicionales

```bash
# Eliminar todos los archivos de modelos tradicionales
rm src/user_based_collaborative_filtering.py
rm src/item_based_collaborative_filtering.py
rm src/matrix_factorization_svd.py
rm src/hybrid_recommender_system.py
rm src/deep_hybrid_recommender.py
rm src/exploratory_analysis.py
rm src/sparsity_analysis.py
rm src/data_loader.py
```

**⚠️ Mantener**: `src/__init__.py` (necesario para Python)

### 📄 Scripts de Entrenamiento Antiguo

```bash
rm main.py                    # Entrena modelos tradicionales
rm train_deep_hybrid.py       # Sistema híbrido complejo
rm verify_deep_hybrid.py      # Verificación de deep hybrid
rm download_fashion.py        # Ya tienes el dataset
```

### 📚 Documentación Antigua

```bash
rm DEEP_HYBRID_GUIDE.md       # Guía del sistema híbrido complejo
rm README_OLD.md              # Backup del README antiguo
```

### 📁 Directorio `web/` - App Web

```bash
rmdir /s web    # Windows
rm -rf web/     # Linux/Mac
```

La app web usa los modelos tradicionales que ya no necesitas.

### 📁 Directorio `models/` - Limpiar Modelos Viejos

```bash
# Mantener solo el modelo NCF
cd models
rm user_similarity.pkl
rm item_similarity.pkl
rm svd_model.pkl
rm hybrid_model.pkl
```

**✅ Mantener**: `ncf_model.pth` (tu modelo de deep learning)

---

## ¿Qué MANTENER?

### ✅ Archivos Esenciales para Aprender

```
recommendation-fashion/
├── notebooks/
│   └── 01_neural_network_tutorial.ipynb    ⭐ PRINCIPAL
├── data/raw/
│   └── fashion_reviews.json                ⭐ DATASET
├── models/
│   └── ncf_model.pth                       ⭐ MODELO ENTRENADO
├── reports/figures/                        ⭐ GRÁFICOS
├── config.py                               ⭐ CONFIGURACIÓN
├── train_ncf_only.py                       ⭐ ENTRENAMIENTO
├── GUIA_APRENDIZAJE.md                     ⭐ CONCEPTOS
├── README.md                               ⭐ INICIO
└── requirements.txt                        ⭐ DEPENDENCIAS
```

---

## Comando Completo de Limpieza (Windows PowerShell)

```powershell
# Navegar al proyecto
cd d:\work\repos-deep-learning\recommendation-fashion

# Eliminar modelos tradicionales de src/
Remove-Item src/user_based_collaborative_filtering.py
Remove-Item src/item_based_collaborative_filtering.py
Remove-Item src/matrix_factorization_svd.py
Remove-Item src/hybrid_recommender_system.py
Remove-Item src/deep_hybrid_recommender.py
Remove-Item src/exploratory_analysis.py
Remove-Item src/sparsity_analysis.py
Remove-Item src/data_loader.py

# Eliminar scripts antiguos
Remove-Item main.py
Remove-Item train_deep_hybrid.py
Remove-Item verify_deep_hybrid.py
Remove-Item download_fashion.py

# Eliminar documentación antigua
Remove-Item DEEP_HYBRID_GUIDE.md
Remove-Item README_OLD.md -ErrorAction SilentlyContinue

# Eliminar web app
Remove-Item -Recurse -Force web

# Limpiar modelos antiguos
Remove-Item models/user_similarity.pkl -ErrorAction SilentlyContinue
Remove-Item models/item_similarity.pkl -ErrorAction SilentlyContinue
Remove-Item models/svd_model.pkl -ErrorAction SilentlyContinue
Remove-Item models/hybrid_model.pkl -ErrorAction SilentlyContinue

Write-Host "✅ Limpieza completada!" -ForegroundColor Green
```

---

## Comando Completo de Limpieza (Linux/Mac)

```bash
# Navegar al proyecto
cd recommendation-fashion

# Eliminar modelos tradicionales de src/
rm -f src/user_based_collaborative_filtering.py
rm -f src/item_based_collaborative_filtering.py
rm -f src/matrix_factorization_svd.py
rm -f src/hybrid_recommender_system.py
rm -f src/deep_hybrid_recommender.py
rm -f src/exploratory_analysis.py
rm -f src/sparsity_analysis.py
rm -f src/data_loader.py

# Eliminar scripts antiguos
rm -f main.py
rm -f train_deep_hybrid.py
rm -f verify_deep_hybrid.py
rm -f download_fashion.py

# Eliminar documentación antigua
rm -f DEEP_HYBRID_GUIDE.md
rm -f README_OLD.md

# Eliminar web app
rm -rf web/

# Limpiar modelos antiguos
rm -f models/user_similarity.pkl
rm -f models/item_similarity.pkl
rm -f models/svd_model.pkl
rm -f models/hybrid_model.pkl

echo "✅ Limpieza completada!"
```

---

## Estructura DESPUÉS de la Limpieza

```
recommendation-fashion/
├── notebooks/
│   ├── 01_neural_network_tutorial.ipynb    ⭐ Tutorial principal
│   └── __init__.py
├── data/
│   ├── raw/
│   │   └── fashion_reviews.json
│   └── processed/                          (vacío)
├── src/
│   └── __init__.py                         (vacío - solo para compatibilidad)
├── models/
│   ├── ncf_model.pth                       ✅ Tu modelo
│   └── ncf_metrics.json                    ✅ Métricas
├── reports/
│   └── figures/
│       └── ncf_training_history.png        ✅ Gráfico
├── config.py                               ✅ Configuración
├── train_ncf_only.py                       ✅ Script de entrenamiento
├── GUIA_APRENDIZAJE.md                     ✅ Guía educativa
├── README.md                               ✅ Documentación
└── requirements.txt                        ✅ Dependencias
```

**🎯 Ganancia**: Proyecto ~70% más pequeño y enfocado solo en deep learning.

---

## ⚠️ Advertencias

### NO Elimines Estos Archivos:

- ❌ `config.py` - Necesario para rutas y configuración
- ❌ `requirements.txt` - Necesario para instalar dependencias
- ❌ `data/raw/fashion_reviews.json` - Tu dataset
- ❌ `notebooks/01_neural_network_tutorial.ipynb` - Tu tutorial principal
- ❌ `train_ncf_only.py` - Script de entrenamiento

### Backup Recomendado:

Antes de eliminar, haz un backup:

```powershell
# Windows
Compress-Archive -Path recommendation-fashion -DestinationPath recommendation-fashion-backup.zip

# Linux/Mac
tar -czf recommendation-fashion-backup.tar.gz recommendation-fashion/
```

---

## ✅ Checklist de Limpieza

- [ ] Hice backup del proyecto completo
- [ ] Eliminé archivos de `src/` (modelos tradicionales)
- [ ] Eliminé scripts antiguos (`main.py`, `train_deep_hybrid.py`, etc.)
- [ ] Eliminé documentación antigua (`DEEP_HYBRID_GUIDE.md`)
- [ ] Eliminé directorio `web/`
- [ ] Limpié modelos antiguos de `models/`
- [ ] Verifiqué que el tutorial notebook sigue funcionando
- [ ] Verifiqué que `train_ncf_only.py` sigue funcionando

---

## 🧪 Verificar que Todo Funciona Después de Limpiar

```bash
# 1. Verificar estructura
ls -la

# 2. Probar script de entrenamiento
python train_ncf_only.py

# 3. Abrir notebook
jupyter notebook notebooks/01_neural_network_tutorial.ipynb
```

Si todo funciona correctamente, ¡la limpieza fue exitosa! 🎉

---

## 💡 ¿Por Qué Limpiar?

1. **Claridad**: Menos archivos = menos confusión
2. **Foco**: Solo lo necesario para aprender redes neuronales
3. **Espacio**: Proyecto ~70% más pequeño
4. **Mantenimiento**: Más fácil de entender y modificar

---

## ⏪ ¿Cómo Revertir la Limpieza?

Si eliminaste archivos y quieres recuperarlos:

```bash
# Restaurar desde backup
Unzip recommendation-fashion-backup.zip

# O desde Git (si tienes control de versiones)
git checkout -- .
```

---

**Recomendación**: Si eres principiante, haz la limpieza para tener un proyecto simple y claro. Si quieres experimentar con métodos tradicionales también, mantén los archivos.
