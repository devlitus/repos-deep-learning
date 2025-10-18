"""
═══════════════════════════════════════════════════════════════
MÓDULO: VISUALIZACIÓN
═══════════════════════════════════════════════════════════════

Propósito: Crear gráficas para analizar datos y resultados

Funciones:
1. plot_temperature_history() → Gráfica de temperaturas históricas
2. plot_training_history() → Progreso del entrenamiento
3. plot_predictions() → Predicciones vs Realidad
4. plot_prediction_scatter() → Gráfica de dispersión
5. plot_errors() → Análisis de errores
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import FIGSIZE


def plot_temperature_history(df, save_path='reports/temperatura_historica.png'):
    """
    Grafica las temperaturas históricas completas (10 años)
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ MUESTRA ESTA GRÁFICA?
    ═══════════════════════════════════════════════════════════
    
    - Eje X: Días (0 a 3650)
    - Eje Y: Temperatura (°C)
    - Línea: Temperatura mínima diaria en Melbourne
    
    ═══════════════════════════════════════════════════════════
    ¿PARA QUÉ SIRVE?
    ═══════════════════════════════════════════════════════════
    
    1. Ver PATRONES ESTACIONALES:
       - ¿Se ve el ciclo invierno-verano?
       - ¿Cada cuánto se repite?
    
    2. Detectar TENDENCIAS:
       - ¿Las temperaturas están subiendo/bajando?
       - ¿Hay cambio climático visible?
    
    3. Identificar OUTLIERS:
       - ¿Hay días muy raros?
       - ¿Olas de calor o frío extremo?
    
    4. Entender la VARIABILIDAD:
       - ¿Cuánto varían las temperaturas?
       - ¿Es predecible o muy caótico?
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        df: DataFrame con columnas 'Date' y 'Temp'
        save_path: Ruta donde guardar la imagen
    """
    
    # Crear carpeta reports si no existe
    os.makedirs('reports', exist_ok=True)
    
    # ═══════════════════════════════════════════════════════════
    # CONFIGURAR LA FIGURA
    # ═══════════════════════════════════════════════════════════
    
    plt.figure(figsize=FIGSIZE)  # FIGSIZE = (15, 5) desde config
    
    """
    ¿Qué es figsize?
    
    figsize=(ancho, alto) en pulgadas
    
    FIGSIZE = (15, 5):
    - 15 pulgadas de ancho → Gráfica horizontal alargada
    - 5 pulgadas de alto → No muy alta
    
    ¿Por qué esta proporción?
    - Series temporales se ven mejor en formato ancho
    - Permite ver más detalle en el eje X (tiempo)
    """
    
    # ═══════════════════════════════════════════════════════════
    # GRAFICAR LOS DATOS
    # ═══════════════════════════════════════════════════════════
    
    plt.plot(
        df['Temp'],           # Datos a graficar
        color='blue',         # Color de la línea
        linewidth=0.8,        # Grosor de la línea
        alpha=0.7             # Transparencia (0=transparente, 1=opaco)
    )
    
    """
    Parámetros de plt.plot():
    
    color='blue':
    - Color de la línea
    - Opciones: 'red', 'green', 'blue', '#FF5733' (hex)
    
    linewidth=0.8:
    - Grosor de la línea
    - 0.5 = muy delgada
    - 1.0 = normal
    - 2.0 = gruesa
    - Usamos 0.8 porque tenemos muchos datos (3650 puntos)
    
    alpha=0.7:
    - Transparencia (opacidad)
    - 0.0 = invisible
    - 1.0 = completamente opaco
    - 0.7 = Ligeramente transparente (se ve mejor)
    """
    
    # ═══════════════════════════════════════════════════════════
    # PERSONALIZAR LA GRÁFICA
    # ═══════════════════════════════════════════════════════════
    
    plt.title(
        'Temperaturas Mínimas Diarias en Melbourne (1981-1990)',
        fontsize=14,
        fontweight='bold'
    )
    
    """
    Título de la gráfica:
    
    fontsize=14:
    - Tamaño de letra (puntos)
    - 10-12 = pequeño
    - 14 = mediano (recomendado para títulos)
    - 16-18 = grande
    
    fontweight='bold':
    - Negrita
    - Hace el título más visible
    """
    
    plt.xlabel('Días', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    
    """
    Etiquetas de ejes:
    
    xlabel = Eje X (horizontal)
    ylabel = Eje Y (vertical)
    
    Siempre incluye unidades:
    ✅ "Temperatura (°C)"
    ❌ "Temperatura"
    """
    
    plt.grid(True, alpha=0.3)
    
    """
    ¿Qué hace plt.grid()?
    
    Agrega líneas de cuadrícula al fondo
    
    True = Mostrar grid
    alpha=0.3 = Grid muy transparente (sutil, no molesta)
    
    ¿Por qué usar grid?
    ✅ Facilita leer valores aproximados
    ✅ Hace la gráfica más profesional
    ✅ Ayuda a ver tendencias
    """
    
    plt.tight_layout()
    
    """
    ¿Qué hace tight_layout()?
    
    Ajusta automáticamente los márgenes para que nada se corte
    
    Sin tight_layout():
    - Títulos pueden cortarse
    - Etiquetas pueden salirse
    
    Con tight_layout():
    - Todo cabe perfectamente ✅
    
    SIEMPRE úsalo antes de guardar
    """
    
    # ═══════════════════════════════════════════════════════════
    # GUARDAR Y MOSTRAR
    # ═══════════════════════════════════════════════════════════
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    """
    Guardar la gráfica:
    
    dpi=300:
    - Dots Per Inch (puntos por pulgada)
    - Resolución de la imagen
    - 72 = pantalla (baja calidad)
    - 150 = buena para web
    - 300 = calidad impresión (alta) ✅
    - 600 = calidad profesional (muy pesado)
    
    bbox_inches='tight':
    - Recorta espacios blancos innecesarios
    - Imagen final más compacta
    """
    
    plt.show()
    
    """
    ¿Qué hace plt.show()?
    
    Muestra la gráfica en pantalla
    
    Nota: En scripts, no es necesario
    Pero es útil para ver la gráfica inmediatamente
    """
    
    print(f"✅ Gráfica guardada en {save_path}")


def plot_training_history(history, save_path='reports/entrenamiento.png'):
    """
    Grafica el progreso del entrenamiento (pérdida vs épocas)
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ MUESTRA ESTA GRÁFICA?
    ═══════════════════════════════════════════════════════════
    
    - Eje X: Épocas (1, 2, 3, ..., 50)
    - Eje Y: Pérdida (MSE)
    - 2 líneas:
      * Azul: Pérdida en ENTRENAMIENTO
      * Naranja: Pérdida en VALIDACIÓN
    
    ═══════════════════════════════════════════════════════════
    ¿PARA QUÉ SIRVE?
    ═══════════════════════════════════════════════════════════
    
    1. Ver si el modelo APRENDE:
       - ¿Las líneas bajan? → Sí aprende ✅
       - ¿Están planas? → No aprende ❌
    
    2. Detectar OVERFITTING:
       - Train baja, Val sube → Overfitting ❌
       - Ambas bajan juntas → Todo bien ✅
    
    3. Decidir CUÁNDO PARAR:
       - ¿Cuándo dejó de mejorar?
       - ¿Se paró en el momento correcto?
    
    4. Ver CONVERGENCIA:
       - ¿Llegó al mínimo?
       - ¿Necesita más épocas?
    
    ═══════════════════════════════════════════════════════════
    PATRONES IMPORTANTES A BUSCAR
    ═══════════════════════════════════════════════════════════
    
    PATRÓN 1 - Aprendizaje Normal (✅):
    Train: \\\\\____
    Val:   \\\\\____
    Ambas bajan juntas y se estabilizan
    
    PATRÓN 2 - Overfitting (❌):
    Train: \\\\\\\\\___
    Val:   \\\___/‾‾‾‾
    Train sigue bajando, Val sube
    
    PATRÓN 3 - Underfitting (❌):
    Train: \\\\\\\\\\\
    Val:   \\\\\\\\\\\
    Ambas aún bajan, necesita más épocas
    
    PATRÓN 4 - No aprende (❌):
    Train: ~~~~~~~~~~
    Val:   ~~~~~~~~~~
    Planas, no mejora
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        history: Objeto History de model.fit()
        save_path: Ruta donde guardar la imagen
    """
    
    plt.figure(figsize=(12, 5))
    
    """
    figsize=(12, 5):
    Un poco más ancho que alto
    Perfecto para ver evolución en el tiempo
    """
    
    # ═══════════════════════════════════════════════════════════
    # GRAFICAR AMBAS LÍNEAS
    # ═══════════════════════════════════════════════════════════
    
    plt.plot(
        history.history['loss'],
        label='Pérdida Entrenamiento',
        linewidth=2,
        color='blue'
    )
    
    plt.plot(
        history.history['val_loss'],
        label='Pérdida Validación',
        linewidth=2,
        color='orange'
    )
    
    """
    history.history es un diccionario con:
    
    'loss': [0.5, 0.3, 0.2, 0.15, ...] → Pérdida en train
    'val_loss': [0.6, 0.35, 0.25, 0.2, ...] → Pérdida en val
    
    label='...':
    - Texto que aparecerá en la leyenda
    - Debe ser descriptivo
    
    linewidth=2:
    - Líneas un poco más gruesas
    - Se ven mejor cuando hay 2 líneas
    """
    
    # ═══════════════════════════════════════════════════════════
    # PERSONALIZAR
    # ═══════════════════════════════════════════════════════════
    
    plt.title('Progreso del Entrenamiento', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida (MSE)', fontsize=12)
    
    plt.legend(fontsize=11)
    
    """
    ¿Qué hace plt.legend()?
    
    Crea la leyenda que explica qué es cada línea
    
    Toma los 'label' de cada plot y crea una cajita:
    ┌─────────────────────────┐
    │ — Pérdida Entrenamiento │ (azul)
    │ — Pérdida Validación    │ (naranja)
    └─────────────────────────┘
    
    fontsize=11:
    - Tamaño de letra de la leyenda
    - Un poco más pequeño que el título (14)
    """
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en {save_path}")


def plot_predictions(y_test, predictions, title='Predicciones vs Realidad',
                    save_path='reports/predicciones.png'):
    """
    Compara predicciones del modelo con valores reales
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ MUESTRA ESTA GRÁFICA?
    ═══════════════════════════════════════════════════════════
    
    - Eje X: Días de prueba
    - Eje Y: Temperatura (°C)
    - 2 líneas superpuestas:
      * Azul: Temperatura REAL
      * Roja: Temperatura PREDICHA
    
    ═══════════════════════════════════════════════════════════
    ¿PARA QUÉ SIRVE?
    ═══════════════════════════════════════════════════════════
    
    1. Ver QUÉ TAN BIEN SIGUE el modelo:
       - ¿Las líneas se superponen? → Bueno ✅
       - ¿Están separadas? → Malo ❌
    
    2. Detectar RETRASO (lag):
       - ¿La predicción va 1 día atrás? → Problema
    
    3. Ver ERRORES EN PICOS:
       - ¿Predice bien los máximos/mínimos?
       - ¿Suaviza demasiado?
    
    4. Identificar PATRONES DE ERROR:
       - ¿Siempre predice bajo en verano?
       - ¿Siempre predice alto en invierno?
    
    ═══════════════════════════════════════════════════════════
    QUÉ BUSCAR EN LA GRÁFICA
    ═══════════════════════════════════════════════════════════
    
    BUENO (✅):
    Real:  ~~~\/\~/~~~
    Pred:  ~~~\/\~/~~~
    → Las líneas casi no se distinguen
    
    MALO - Suavizado excesivo (❌):
    Real:  ~~~/\~~/\~~
    Pred:  ~~~\__/~~~
    → Predicción ignora cambios bruscos
    
    MALO - Con retraso (❌):
    Real:  ~/\~~/\~~
    Pred:   ~\/~~/\~
    → Predicción va 1 paso atrás
    
    MALO - Sesgo (❌):
    Real:  ~~~~~~~~~~~  (promedio: 20°C)
    Pred:  __________ (promedio: 18°C)
    → Siempre predice más bajo
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        y_test: Valores reales
        predictions: Valores predichos
        title: Título de la gráfica
        save_path: Ruta donde guardar
    """
    
    plt.figure(figsize=FIGSIZE)
    
    # ═══════════════════════════════════════════════════════════
    # GRAFICAR AMBAS LÍNEAS
    # ═══════════════════════════════════════════════════════════
    
    plt.plot(
        y_test,
        color='blue',
        label='Temperatura Real',
        linewidth=2,
        alpha=0.7
    )
    
    plt.plot(
        predictions,
        color='red',
        label='Predicción',
        linewidth=2,
        alpha=0.7
    )
    
    """
    Colores elegidos:
    
    Azul (real): Color frío, asociado con "datos reales"
    Rojo (predicción): Color cálido, destaca más
    
    alpha=0.7:
    - Semi-transparente
    - Donde se superponen se ve morado
    - Permite ver ambas líneas incluso si coinciden
    """
    
    # ═══════════════════════════════════════════════════════════
    # PERSONALIZAR
    # ═══════════════════════════════════════════════════════════
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Días', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en {save_path}")


def plot_prediction_scatter(y_test, predictions, save_path='reports/scatter.png'):
    """
    Gráfica de dispersión: cada punto es (real, predicción)
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ MUESTRA ESTA GRÁFICA?
    ═══════════════════════════════════════════════════════════
    
    - Eje X: Temperatura REAL
    - Eje Y: Temperatura PREDICHA
    - Cada punto: Un día
    - Línea diagonal roja: Predicción perfecta
    
    ═══════════════════════════════════════════════════════════
    ¿PARA QUÉ SIRVE?
    ═══════════════════════════════════════════════════════════
    
    1. Ver PRECISIÓN GLOBAL:
       - ¿Puntos cerca de la diagonal? → Bueno ✅
       - ¿Puntos dispersos? → Malo ❌
    
    2. Detectar SESGO:
       - ¿Puntos arriba de diagonal? → Predice ALTO
       - ¿Puntos abajo de diagonal? → Predice BAJO
    
    3. Ver HETEROSCEDASTICIDAD:
       - ¿Error aumenta con la temperatura?
       - ¿Error constante en todo el rango?
    
    4. Identificar OUTLIERS:
       - ¿Hay puntos muy alejados?
       - ¿Días problemáticos?
    
    ═══════════════════════════════════════════════════════════
    INTERPRETACIÓN VISUAL
    ═══════════════════════════════════════════════════════════
    
    PERFECTO (✅):
         P
         |  ●●●●●
         |  ●●●●●  ← Todos en la línea
         |  ●●●●●
         └────── R
    
    SESGO ALTO (❌):
         P
         |    ●●●●
         |    ●●●●  ← Predice siempre más alto
         |  ●●●●
         └────── R
    
    SESGO BAJO (❌):
         P
         |  ●●●●
         |    ●●●●  ← Predice siempre más bajo
         |      ●●●●
         └────── R
    
    DISPERSIÓN (❌):
         P
         |  ●  ●
         |    ●  ●  ← Puntos muy dispersos
         | ●    ●
         └────── R
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        y_test: Valores reales
        predictions: Valores predichos
        save_path: Ruta donde guardar
    """
    
    plt.figure(figsize=(10, 10))
    
    """
    figsize=(10, 10):
    - Cuadrada (mismo ancho y alto)
    - Para scatter plots se ven mejor cuadrados
    - Permite comparar ejes X e Y visualmente
    """
    
    # ═══════════════════════════════════════════════════════════
    # GRAFICAR PUNTOS
    # ═══════════════════════════════════════════════════════════
    
    plt.scatter(
        y_test,
        predictions,
        alpha=0.5,
        s=20,
        color='blue',
        edgecolors='black',
        linewidth=0.5
    )
    
    """
    plt.scatter() parámetros:
    
    alpha=0.5:
    - Transparencia
    - Si puntos se superponen, se ve más oscuro
    - Permite ver densidad
    
    s=20:
    - Size (tamaño de cada punto)
    - 10 = pequeño
    - 20 = mediano ✅
    - 50 = grande
    
    edgecolors='black':
    - Borde negro alrededor de cada punto
    - Hace puntos más visibles
    
    linewidth=0.5:
    - Grosor del borde
    - Línea delgada, sutil
    """
    
    # ═══════════════════════════════════════════════════════════
    # LÍNEA DIAGONAL (Predicción Perfecta)
    # ═══════════════════════════════════════════════════════════
    
    # Calcular rango para la línea diagonal
    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())
    
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        'r--',
        linewidth=2,
        label='Predicción Perfecta'
    )
    
    """
    Línea diagonal:
    
    [min_val, max_val]: Coordenadas X
    [min_val, max_val]: Coordenadas Y (mismas)
    → Línea de (min,min) a (max,max)
    
    'r--':
    - 'r' = red (rojo)
    - '--' = dashed (discontinua)
    
    Esta línea representa y=x (predicción = realidad)
    Si todos los puntos estuvieran aquí → modelo perfecto
    """
    
    # ═══════════════════════════════════════════════════════════
    # PERSONALIZAR
    # ═══════════════════════════════════════════════════════════
    
    plt.title('Predicción vs Realidad (Dispersión)', fontsize=14, fontweight='bold')
    plt.xlabel('Temperatura Real (°C)', fontsize=12)
    plt.ylabel('Temperatura Predicha (°C)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.axis('equal')
    
    """
    ¿Qué hace plt.axis('equal')?
    
    Hace que las escalas de X e Y sean iguales
    
    Sin axis('equal'):
    - 1cm en X podría = 5°C
    - 1cm en Y podría = 10°C
    - Diagonal no se ve a 45°
    
    Con axis('equal'):
    - 1cm = mismo valor en ambos ejes
    - Diagonal perfecta a 45°
    - Más fácil interpretar ✅
    """
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en {save_path}")


def plot_errors(y_test, predictions, save_path='reports/errores.png'):
    """
    Analiza los errores de predicción (2 gráficas en 1)
    
    ═══════════════════════════════════════════════════════════
    ¿QUÉ MUESTRA ESTA GRÁFICA?
    ═══════════════════════════════════════════════════════════
    
    GRÁFICA 1 (Izquierda): Errores en el tiempo
    - Eje X: Días
    - Eje Y: Error (real - predicción)
    - Línea horizontal en 0 = Sin error
    
    GRÁFICA 2 (Derecha): Histograma de errores
    - Eje X: Error (°C)
    - Eje Y: Frecuencia (cantidad de días)
    - Muestra distribución de errores
    
    ═══════════════════════════════════════════════════════════
    ¿PARA QUÉ SIRVE?
    ═══════════════════════════════════════════════════════════
    
    1. Ver PATRONES TEMPORALES en errores:
       - ¿Errores más grandes en verano/invierno?
       - ¿Hay épocas problemáticas?
    
    2. Detectar SESGO SISTEMÁTICO:
       - ¿Errores mayormente positivos? → Predice bajo
       - ¿Errores mayormente negativos? → Predice alto
       - ¿Centrados en 0? → Sin sesgo ✅
    
    3. Ver DISTRIBUCIÓN de errores:
       - ¿Forma de campana? → Normal ✅
       - ¿Asimétrica? → Problema
       - ¿Con colas largas? → Outliers
    
    4. Calcular ESTADÍSTICAS:
       - Error promedio
       - Error absoluto promedio
       - Desviación estándar
    
    ═══════════════════════════════════════════════════════════
    INTERPRETACIONES
    ═══════════════════════════════════════════════════════════
    
    ERRORES EN EL TIEMPO:
    
    BUENO (✅):
    Error
      2  ╭─╮  ╭─╮
      0 ─┼─┼──┼─┼──  ← Oscila alrededor de 0
     -2  ╰─╯  ╰─╯
    
    MALO - Sesgo (❌):
    Error
      2  ─────────  ← Siempre positivo (predice bajo)
      0
     -2
    
    MALO - Patrón temporal (❌):
    Error
      2      ╱╲    ← Error crece con el tiempo
      0  ╱╲╱  ╲╱
     -2
    
    HISTOGRAMA:
    
    BUENO (✅):
         │    ●●●
         │   ●●●●●    ← Campana centrada en 0
         │  ●●●●●●●
         │ ●●●●●●●●●
         └──────────
         -2  0   2
    
    MALO - Sesgo (❌):
         │      ●●●
         │     ●●●●●  ← Centrado en 1, no en 0
         │    ●●●●●●
         │   ●●●●●●●
         └──────────
          0  1  2  3
    
    ═══════════════════════════════════════════════════════════
    
    Args:
        y_test: Valores reales
        predictions: Valores predichos
        save_path: Ruta donde guardar
    """
    
    # Calcular errores
    errors = y_test - predictions.flatten()
    
    """
    Error = Real - Predicción
    
    Si error > 0: Predicción fue BAJA (predijo menos)
    Si error < 0: Predicción fue ALTA (predijo más)
    Si error = 0: Predicción PERFECTA
    
    .flatten() convierte [[1], [2]] en [1, 2]
    """
    
    # ═══════════════════════════════════════════════════════════
    # CREAR FIGURA CON 2 SUBPLOTS
    # ═══════════════════════════════════════════════════════════
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    """
    plt.subplots(1, 2):
    
    Crea 1 fila, 2 columnas de gráficas
    
    Resultado:
    ┌──────────┬──────────┐
    │  axes[0] │  axes[1] │
    └──────────┴──────────┘
    
    axes[0] = Gráfica izquierda
    axes[1] = Gráfica derecha
    
    figsize=(15, 5):
    - Ancho total: 15 pulgadas
    - Alto: 5 pulgadas
    - Cada subplot: ~7.5 pulgadas de ancho
    """
    
    # ═══════════════════════════════════════════════════════════
    # SUBPLOT 1: Errores en el Tiempo
    # ═══════════════════════════════════════════════════════════
    
    axes[0].plot(errors, color='red', alpha=0.6, linewidth=1)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    """
    axhline(y=0):
    - Dibuja línea HORIZONTAL en y=0
    - Representa "sin error"
    - Referencia visual importante
    
    Errores arriba de 0: Predicción baja
    Errores abajo de 0: Predicción alta
    """
    
    axes[0].set_title('Errores de Predicción en el Tiempo', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Días', fontsize=11)
    axes[0].set_ylabel('Error (°C)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    """
    axes[0].set_title() vs plt.title():
    
    Cuando tienes subplots, usas:
    - axes[0].set_title() para cada subplot
    - No plt.title() (eso es para figura completa)
    """
    
    # ═══════════════════════════════════════════════════════════
    # SUBPLOT 2: Histograma de Errores
    # ═══════════════════════════════════════════════════════════
    
    axes[1].hist(
        errors,
        bins=50,
        color='red',
        alpha=0.6,
        edgecolor='black'
    )
    
    """
    plt.hist() parámetros:
    
    bins=50:
    - Cantidad de "barras" en el histograma
    - Más bins = más detalle
    - 30-50 es buen rango para la mayoría de casos
    
    edgecolor='black':
    - Borde negro en cada barra
    - Separa barras visualmente
    - Hace histograma más legible
    
    ¿Qué es un histograma?
    Cuenta cuántos valores caen en cada rango:
    
    Errores: [-1, -0.5, 0, 0.2, 0.3, 1, 1.2, ...]
    
    Histogram:
         │
      5  │     ██
      4  │   ████
      3  │ ██████
      2  │████████
      1  │████████
         └────────
        -2  0  2
    
    Altura = Cuántos errores en ese rango
    """
    
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    """
    axvline(x=0):
    - Línea VERTICAL en x=0
    - Marca el "sin error"
    
    Si el pico del histograma está en 0:
    → La mayoría de errores son pequeños ✅
    
    Si el pico está en 2:
    → Hay sesgo (predice 2°C bajo) ❌
    """
    
    axes[1].set_title('Distribución de Errores', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Error (°C)', fontsize=11)
    axes[1].set_ylabel('Frecuencia', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # ═══════════════════════════════════════════════════════════
    # GUARDAR Y CALCULAR ESTADÍSTICAS
    # ═══════════════════════════════════════════════════════════
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Gráfica guardada en {save_path}")
    
    # Imprimir estadísticas de errores
    print(f"\n📊 Estadísticas de errores:")
    print(f"   Error promedio         : {np.mean(errors):.3f}°C")
    print(f"   Error absoluto promedio: {np.mean(np.abs(errors)):.3f}°C")
    print(f"   Desviación estándar    : {np.std(errors):.3f}°C")
    print(f"   Error mínimo           : {np.min(errors):.3f}°C")
    print(f"   Error máximo           : {np.max(errors):.3f}°C")
    
    """
    Estadísticas útiles:
    
    Error promedio:
    - Cercano a 0 → Sin sesgo ✅
    - Positivo → Predice bajo
    - Negativo → Predice alto
    
    Error absoluto promedio:
    - Magnitud típica del error
    - Siempre positivo
    
    Desviación estándar:
    - Variabilidad de los errores
    - Baja → Errores consistentes ✅
    - Alta → Errores impredecibles ❌
    
    Min/Max:
    - Errores extremos
    - Identifica outliers
    """


# ═══════════════════════════════════════════════════════════
# BLOQUE DE PRUEBA
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 PROBANDO MÓDULO DE VISUALIZACIÓN")
    print("="*70 + "\n")
    
    print("💡 Este módulo crea gráficas cuando se ejecuta train.py")
    print("   No hay prueba standalone porque necesita datos reales\n")
    
    print("📊 Funciones disponibles:")
    print("   1. plot_temperature_history() → Temperaturas históricas")
    print("   2. plot_training_history()    → Progreso entrenamiento")
    print("   3. plot_predictions()          → Predicciones vs Realidad")
    print("   4. plot_prediction_scatter()   → Gráfica de dispersión")
    print("   5. plot_errors()               → Análisis de errores")
    
    print("\n✅ Módulo cargado correctamente")
    print("="*70)