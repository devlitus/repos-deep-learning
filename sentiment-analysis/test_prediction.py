"""
Test de Predicciones - LSTM Sentiment Analysis
===============================================

Script de prueba que REPLICA EXACTAMENTE el pipeline de entrenamiento
para verificar el rendimiento real del modelo.

PIPELINE CORRECTO:
1. Preprocesar texto (clean_text o preprocess_text)
2. Tokenizar con el tokenizer guardado
3. Padding='pre' (igual que en entrenamiento)
4. Predecir con modelo LSTM
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al PATH
BASE_DIR = Path(__file__).parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Importar módulos del proyecto
import config
from src import text_preprocessing

# ============================================================================
# CARGAR MODELO Y TOKENIZER
# ============================================================================

print("=" * 70)
print("🔮 TEST DE PREDICCIONES - LSTM SENTIMENT ANALYSIS")
print("=" * 70)

print("\n📂 Cargando modelo y tokenizer...")
MODEL_PATH = config.MODEL_LSTM
TOKENIZER_PATH = config.TOKENIZER_FILE

if not MODEL_PATH.exists():
    print(f"❌ Error: Modelo no encontrado en {MODEL_PATH}")
    print("💡 Ejecuta 'python main.py' para entrenar el modelo")
    sys.exit(1)

if not TOKENIZER_PATH.exists():
    print(f"❌ Error: Tokenizer no encontrado en {TOKENIZER_PATH}")
    sys.exit(1)

model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

print(f"✅ Modelo cargado: {MODEL_PATH.name}")
print(f"✅ Tokenizer cargado: {TOKENIZER_PATH.name}")
print(f"📏 Max sequence length: {config.MAX_SEQUENCE_LENGTH}")

# ============================================================================
# FUNCIÓN DE PREDICCIÓN (REPLICA EL PIPELINE DE ENTRENAMIENTO)
# ============================================================================

def predict_with_pipeline(text: str, verbose: bool = True):
    """
    Predice el sentimiento usando el MISMO pipeline que el entrenamiento.

    Pipeline:
    1. Preprocesar texto (eliminar HTML, URLs, puntuación, etc.)
    2. Tokenizar (convertir palabras a índices)
    3. Padding='pre' (rellenar al inicio)
    4. Predecir con modelo

    Returns:
        dict con 'sentiment', 'confidence', 'probability', 'details'
    """
    if verbose:
        print(f"\n{'─' * 70}")
        print(f"📝 Texto original:")
        print(f"   '{text}'")

    # 1. PREPROCESAR (IGUAL QUE EN ENTRENAMIENTO)
    # Usa clean_text o preprocess_text según config
    preprocessed = text_preprocessing.clean_text(
        text,
        remove_html=True,
        remove_url=True,
        remove_punct=config.REMOVE_PUNCTUATION,
        remove_num=config.REMOVE_NUMBERS,
        lowercase=True
    )

    if verbose:
        print(f"\n🧹 Texto preprocesado:")
        print(f"   '{preprocessed}'")
        print(f"   Palabras: {len(text.split())} → {len(preprocessed.split())}")

    # 2. TOKENIZAR
    sequence = tokenizer.texts_to_sequences([preprocessed])

    if verbose:
        print(f"\n🔢 Secuencia tokenizada:")
        print(f"   Índices: {sequence[0][:20]}{'...' if len(sequence[0]) > 20 else ''}")
        print(f"   Longitud: {len(sequence[0])} palabras")

        # Mostrar palabras reconocidas
        words = preprocessed.split()
        print(f"\n📖 Palabras en vocabulario:")
        for word in words[:10]:  # Primeras 10 palabras
            idx = tokenizer.word_index.get(word)
            if idx:
                print(f"   ✓ '{word}' → índice {idx}")
            else:
                print(f"   ✗ '{word}' → NO EN VOCABULARIO")

    # 3. PADDING (IGUAL QUE EN ENTRENAMIENTO: 'pre')
    padded = pad_sequences(
        sequence,
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding='pre',  # ← CRÍTICO: mismo que en entrenamiento
        truncating='post'
    )

    if verbose:
        print(f"\n⬜ Padding aplicado:")
        print(f"   Método: 'pre' (rellena al inicio)")
        print(f"   Shape: {padded.shape}")
        print(f"   Ceros añadidos: {config.MAX_SEQUENCE_LENGTH - len(sequence[0])}")

    # 4. PREDECIR
    probability = float(model.predict(padded, verbose=0)[0][0])

    # Interpretar resultado
    if probability > 0.5:
        sentiment = "POSITIVO"
        confidence = probability
    else:
        sentiment = "NEGATIVO"
        confidence = 1 - probability

    if verbose:
        print(f"\n🎯 PREDICCIÓN:")
        print(f"   Probabilidad raw: {probability:.4f}")
        print(f"   Sentimiento: {sentiment}")
        print(f"   Confianza: {confidence:.2%}")

    return {
        'text': text,
        'preprocessed': preprocessed,
        'sentiment': sentiment,
        'confidence': confidence,
        'probability': probability,
        'sequence_length': len(sequence[0])
    }

# ============================================================================
# CASOS DE PRUEBA
# ============================================================================

print("\n" + "=" * 70)
print("🧪 EJECUTANDO CASOS DE PRUEBA")
print("=" * 70)

test_cases = [
    {
        'text': "This movie is excellent! Best film ever!",
        'expected': "POSITIVO",
        'description': "Positivo claro con palabras fuertes"
    },
    {
        'text': "Terrible waste of time. The plot was boring and confusing.",
        'expected': "NEGATIVO",
        'description': "Negativo claro con múltiples palabras negativas"
    },
    {
        'text': "I absolutely loved this movie!",
        'expected': "POSITIVO",
        'description': "Positivo simple y corto"
    },
    {
        'text': "Worst movie I've ever seen. Awful.",
        'expected': "NEGATIVO",
        'description': "Negativo fuerte y corto"
    },
    {
        'text': "The movie was okay. Nothing special but not terrible either.",
        'expected': "NEUTRAL/INCIERTO",
        'description': "Sentimiento mixto/neutral"
    },
    {
        'text': "Incredible performance by the lead actor. The cinematography was stunning and the soundtrack perfect.",
        'expected': "POSITIVO",
        'description': "Positivo largo con detalles específicos"
    },
    {
        'text': "I fell asleep halfway through. Boring, predictable, and poorly acted.",
        'expected': "NEGATIVO",
        'description': "Negativo con razones específicas"
    },
    {
        'text': "Masterpiece!",
        'expected': "POSITIVO",
        'description': "Positivo muy corto (1 palabra)"
    },
    {
        'text': "Garbage.",
        'expected': "NEGATIVO",
        'description': "Negativo muy corto (1 palabra)"
    },
    {
        'text': "Not bad, but could have been better. Some good scenes though.",
        'expected': "NEUTRAL/POSITIVO",
        'description': "Sentimiento ligeramente positivo con reservas"
    }
]

# ============================================================================
# EJECUTAR PRUEBAS
# ============================================================================

results = []
correct = 0
total = 0

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'=' * 70}")
    print(f"TEST {i}/{len(test_cases)}: {test_case['description']}")
    print(f"{'=' * 70}")

    result = predict_with_pipeline(test_case['text'], verbose=True)

    # Verificar si es correcto
    expected = test_case['expected']
    predicted = result['sentiment']

    # Manejo especial para casos neutrales/inciertos
    if "NEUTRAL" in expected or "INCIERTO" in expected:
        is_correct = result['confidence'] < 0.7  # Baja confianza esperada
        status = "✓" if is_correct else "?"
    else:
        is_correct = predicted == expected
        status = "✅" if is_correct else "❌"

    if is_correct:
        correct += 1
    total += 1

    print(f"\n{status} ESPERADO: {expected}")
    print(f"{status} OBTENIDO: {predicted} (confianza: {result['confidence']:.2%})")

    results.append({
        'test': i,
        'text': test_case['text'],
        'expected': expected,
        'predicted': predicted,
        'confidence': result['confidence'],
        'correct': is_correct
    })

# ============================================================================
# RESUMEN DE RESULTADOS
# ============================================================================

print("\n" + "=" * 70)
print("📊 RESUMEN DE RESULTADOS")
print("=" * 70)

accuracy = (correct / total) * 100

print(f"\n🎯 Accuracy en casos de prueba: {correct}/{total} ({accuracy:.1f}%)")
print(f"\n📋 Detalle por caso:")

for r in results:
    status = "✅" if r['correct'] else "❌"
    print(f"{status} Test {r['test']}: {r['predicted']} (confianza: {r['confidence']:.1%})")
    if not r['correct']:
        print(f"   Esperado: {r['expected']}")
        print(f"   Texto: {r['text'][:60]}...")

# ============================================================================
# ANÁLISIS DE ERRORES
# ============================================================================

errors = [r for r in results if not r['correct']]

if errors:
    print(f"\n⚠️  ANÁLISIS DE ERRORES ({len(errors)} casos):")
    print("=" * 70)

    for err in errors:
        print(f"\n❌ Test {err['test']}:")
        print(f"   Texto: {err['text']}")
        print(f"   Esperado: {err['expected']}")
        print(f"   Predicho: {err['predicted']} (confianza: {err['confidence']:.1%})")

        # Sugerir posibles causas
        if err['confidence'] < 0.65:
            print(f"   💡 Baja confianza - caso ambiguo o límite")
        if len(err['text'].split()) < 5:
            print(f"   💡 Texto muy corto - modelo entrenado con textos más largos")
else:
    print(f"\n🎉 ¡Todos los casos de prueba pasaron correctamente!")

# ============================================================================
# RECOMENDACIONES
# ============================================================================

print("\n" + "=" * 70)
print("💡 RECOMENDACIONES")
print("=" * 70)

if accuracy >= 90:
    print("\n✅ El modelo funciona excelentemente en estos casos")
    print("   Rendimiento esperado con el 87% de accuracy en dataset completo")
elif accuracy >= 70:
    print("\n⚠️  El modelo tiene rendimiento aceptable pero mejorable")
    print("   Posibles mejoras:")
    print("   - Aumentar épocas de entrenamiento")
    print("   - Ajustar learning rate")
    print("   - Agregar más capas LSTM o Bidirectional")
else:
    print("\n❌ El modelo tiene bajo rendimiento en estos casos")
    print("   Posibles causas:")
    print("   1. Modelo no entrenado suficientemente")
    print("   2. Casos de prueba muy diferentes del dataset IMDB")
    print("   3. Necesita re-arquitectura (BERT, más capas, etc.)")
    print("\n   Recomendaciones:")
    print("   - Verificar que el modelo se entrenó correctamente")
    print("   - Revisar el histórico de entrenamiento")
    print("   - Considerar usar transfer learning (BERT/RoBERTa)")

print("\n" + "=" * 70)
print("✅ TEST COMPLETADO")
print("=" * 70)
