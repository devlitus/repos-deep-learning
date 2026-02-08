"""
Módulo de visualización de resultados NCF
Incluye curvas de entrenamiento, predicciones y comparación de experimentos
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# Motor de Diagnóstico y Recomendaciones
# ============================================================

# Umbrales de diagnóstico
_THRESHOLDS = {
    'overfitting_mild': 0.10,
    'overfitting_severe': 0.20,
    'underfitting': 0.85,
    'good_rmse': 0.70,
    'convergence_ratio': 0.50,
    'instability_std': 0.02,
    'early_stop_ratio': 0.40,
    'high_lr': 0.005,
    'low_lr': 0.0003,
    'divergence_threshold': 0.10,  # Si val_rmse sube >10% en últimas epochs
    'stagnation_ratio': 0.40,  # Si no mejora por >40% de las epochs
    'test_val_discrepancy': 0.10,  # Si test_rmse > val_rmse por >10%
    'train_val_separation': 0.05,  # Mejora en train vs mejora en val
    'params_per_sample': 10,  # Mínimo de muestras por parámetro
}


def _diagnose_experiment(exp):
    """
    Analiza un experimento y retorna lista de diagnósticos.
    Cada diagnóstico: {'id': str, 'severity': str, 'title': str, 'description': str}
    severity: 'critical' | 'warning' | 'info' | 'success'
    """
    diagnostics = []
    history = exp['history']
    train_rmse_final = history['train_rmse'][-1]
    val_rmse_final = history['val_rmse'][-1]
    gap = val_rmse_final - train_rmse_final
    test_rmse = exp['final_test_rmse']
    best_epoch = exp['best_epoch']
    total_epochs = exp['total_epochs']
    max_epochs = exp['max_epochs']
    best_val_rmse = min(history['val_rmse']) if history['val_rmse'] else val_rmse_final

    # 1. Overfitting
    if gap >= _THRESHOLDS['overfitting_severe']:
        diagnostics.append({
            'id': 'overfitting_severe',
            'severity': 'critical',
            'title': 'Overfitting severo',
            'description': (
                f"La diferencia entre entrenamiento y validación es **{gap:.3f}** "
                f"(Train: {train_rmse_final:.4f} vs Val: {val_rmse_final:.4f}). "
                "El modelo está memorizando los datos en vez de aprender patrones generales."
            )
        })
    elif gap >= _THRESHOLDS['overfitting_mild']:
        diagnostics.append({
            'id': 'overfitting_mild',
            'severity': 'warning',
            'title': 'Overfitting moderado',
            'description': (
                f"Hay una diferencia de **{gap:.3f}** entre entrenamiento y validación. "
                "El modelo muestra signos de memorización, pero es controlable."
            )
        })

    # 2. Underfitting
    if test_rmse >= _THRESHOLDS['underfitting']:
        diagnostics.append({
            'id': 'underfitting',
            'severity': 'critical',
            'title': 'Underfitting',
            'description': (
                f"El Test RMSE de **{test_rmse:.4f}** es alto. "
                "El modelo no tiene suficiente capacidad para capturar los patrones del dataset."
            )
        })

    # 3. Convergencia prematura
    if total_epochs > 3 and best_epoch < total_epochs * _THRESHOLDS['convergence_ratio']:
        diagnostics.append({
            'id': 'early_convergence',
            'severity': 'warning',
            'title': 'Convergencia prematura',
            'description': (
                f"El mejor resultado se alcanzó en la epoch **{best_epoch}** de {total_epochs}. "
                "El modelo encontró su mejor punto muy pronto y luego no mejoró."
            )
        })

    # 4. Inestabilidad
    if len(history['val_rmse']) > 3:
        val_diffs = np.diff(history['val_rmse'])
        val_std = np.std(val_diffs)
        # Solo reportar inestabilidad si es significativa Y el modelo no convergió bien
        # Si el gap es bajo, las oscilaciones son normales (ajuste fino)
        if val_std > _THRESHOLDS['instability_std'] and (gap >= 0.15 or test_rmse >= 0.80):
            diagnostics.append({
                'id': 'instability',
                'severity': 'warning',
                'title': 'Inestabilidad en el entrenamiento',
                'description': (
                    f"La validación oscila significativamente entre epochs "
                    f"(varianza de cambios: {val_std:.4f}). "
                    "Esto indica que el aprendizaje es errático."
                )
            })

    # 5. Early stopping muy temprano (no aplicar si fue detención manual)
    stopped_manually = exp.get('stopped_manually', False)
    if exp['stopped_early'] and not stopped_manually and total_epochs < max_epochs * _THRESHOLDS['early_stop_ratio']:
        diagnostics.append({
            'id': 'early_stop_premature',
            'severity': 'info',
            'title': 'Early stopping muy temprano',
            'description': (
                f"El entrenamiento paró en la epoch {total_epochs} de {max_epochs} "
                f"({total_epochs/max_epochs*100:.0f}%). "
                "El modelo podría beneficiarse de más rondas con ajustes."
            )
        })

    # 6. Agotó todas las epochs sin early stopping (no aplicar si fue detención manual)
    if not exp['stopped_early'] and not stopped_manually and total_epochs == max_epochs:
        # Check if val_rmse was still decreasing significantly
        if len(history['val_rmse']) >= 5:
            last_5 = history['val_rmse'][-5:]
            improvement = last_5[0] - last_5[-1]  # Mejora en las últimas 5 epochs
            avg_val = np.mean(last_5)

            # Solo recomendar más epochs si:
            # 1. Hay mejora significativa (>1% del RMSE promedio)
            # 2. El gap no está ya en zona excelente (<0.10)
            # 3. La mejora no es residual
            if improvement > avg_val * 0.01 and gap >= 0.10:
                diagnostics.append({
                    'id': 'could_train_more',
                    'severity': 'info',
                    'title': 'Podría seguir mejorando',
                    'description': (
                        f"Se usaron las {max_epochs} epochs completas con mejora de {improvement:.4f} "
                        f"en las últimas 5 epochs. Subir el número máximo de epochs podría mejorar el resultado."
                    )
                })

    # 7. Modelo excelente
    if test_rmse < _THRESHOLDS['good_rmse'] and gap < _THRESHOLDS['overfitting_mild']:
        diagnostics.append({
            'id': 'excellent',
            'severity': 'success',
            'title': '¡Excelente resultado!',
            'description': (
                f"Test RMSE de **{test_rmse:.4f}** con un gap de solo **{gap:.3f}**. "
                "El modelo generaliza muy bien. Considera guardarlo como referencia."
            )
        })

    # 8. LR Analysis
    lr = exp['learning_rate']
    if lr >= _THRESHOLDS['high_lr']:
        diagnostics.append({
            'id': 'high_lr',
            'severity': 'info',
            'title': 'Learning rate alto',
            'description': (
                f"El LR de **{lr}** es relativamente alto. "
                "Puede causar oscilaciones y dificultar la convergencia fina."
            )
        })
    elif lr <= _THRESHOLDS['low_lr']:
        diagnostics.append({
            'id': 'low_lr',
            'severity': 'info',
            'title': 'Learning rate bajo',
            'description': (
                f"El LR de **{lr}** es bajo. "
                "El modelo tarda más en converger — asegúrate de tener suficientes epochs."
            )
        })

    # 9. Divergencia (modelo empeora en últimas epochs)
    if len(history['val_rmse']) >= 5:
        last_5_val = history['val_rmse'][-5:]
        first_of_last_5 = last_5_val[0]
        final_val = last_5_val[-1]
        if final_val > first_of_last_5 * (1 + _THRESHOLDS['divergence_threshold']):
            increase = ((final_val - first_of_last_5) / first_of_last_5) * 100
            diagnostics.append({
                'id': 'divergence',
                'severity': 'critical',
                'title': '🔻 Divergencia detectada',
                'description': (
                    f"El modelo está empeorando: Val RMSE subió **{increase:.1f}%** "
                    f"en las últimas 5 epochs ({first_of_last_5:.4f} → {final_val:.4f}). "
                    "Esto indica gradientes explosivos o learning rate demasiado alto."
                )
            })

    # 10. Estancamiento prolongado (no mejora por muchas epochs)
    if total_epochs >= 8:
        # Buscar el mejor val_rmse y ver cuántas epochs después siguió entrenando
        best_val_idx = np.argmin(history['val_rmse'])
        epochs_after_best = total_epochs - best_val_idx - 1
        if epochs_after_best > total_epochs * _THRESHOLDS['stagnation_ratio']:
            wasted_epochs = epochs_after_best
            diagnostics.append({
                'id': 'stagnation',
                'severity': 'warning',
                'title': '📉 Estancamiento prolongado',
                'description': (
                    f"El mejor resultado fue en la epoch {best_val_idx + 1}, pero continuó "
                    f"entrenando **{wasted_epochs} epochs más** sin mejora. "
                    f"Esto desperdicia tiempo de entrenamiento."
                )
            })

    # 11. Discrepancia Test vs Validation (test mucho peor que validation)
    if best_val_rmse > 0:
        test_val_diff = (test_rmse - best_val_rmse) / best_val_rmse
        if test_val_diff > _THRESHOLDS['test_val_discrepancy']:
            diagnostics.append({
                'id': 'test_val_discrepancy',
                'severity': 'warning',
                'title': '⚖️ Discrepancia Test-Validation',
                'description': (
                    f"Test RMSE (**{test_rmse:.4f}**) es **{test_val_diff*100:.1f}%** peor "
                    f"que el mejor Val RMSE (**{best_val_rmse:.4f}**). "
                    "Posible overfitting al validation set o distribución diferente de datos."
                )
            })

    # 12. Mejora solo en train (overfitting en progreso)
    if len(history['train_rmse']) >= 5 and len(history['val_rmse']) >= 5:
        last_5_train = history['train_rmse'][-5:]
        last_5_val = history['val_rmse'][-5:]
        train_improvement = (last_5_train[0] - last_5_train[-1]) / last_5_train[0]
        val_improvement = max(0, (last_5_val[0] - last_5_val[-1]) / last_5_val[0])

        if train_improvement > 0.05 and val_improvement < 0.01:
            diagnostics.append({
                'id': 'train_only_improvement',
                'severity': 'warning',
                'title': '🚀 Mejora solo en Train',
                'description': (
                    f"En las últimas 5 epochs: Train mejoró **{train_improvement*100:.1f}%** "
                    f"pero Val solo **{val_improvement*100:.1f}%**. "
                    "El modelo está empezando a memorizar en vez de generalizar."
                )
            })

    # 13. Capacidad excesiva del modelo
    total_params = exp.get('total_params', 0)
    if total_params > 0 and 'n_train_samples' in exp:
        n_samples = exp['n_train_samples']
        ratio = n_samples / total_params
        if ratio < _THRESHOLDS['params_per_sample']:
            diagnostics.append({
                'id': 'model_too_large',
                'severity': 'info',
                'title': '🧠 Capacidad excesiva del modelo',
                'description': (
                    f"El modelo tiene **{total_params:,}** parámetros para solo "
                    f"**{n_samples:,}** muestras de entrenamiento (ratio: {ratio:.1f}). "
                    "Un modelo muy grande para pocos datos aumenta el riesgo de overfitting. "
                    f"Idealmente deberías tener al menos {_THRESHOLDS['params_per_sample']} muestras por parámetro."
                )
            })

    return diagnostics


def _generate_recommendations(exp, diagnostics):
    """
    Genera recomendaciones concretas basadas en los diagnósticos.
    Cada recomendación: {
        'param': str, 'current': str, 'suggested': str,
        'direction': str, 'reason': str, 'priority': int
    }
    """
    recs = []
    diag_ids = {d['id'] for d in diagnostics}

    dropout = exp['dropout']
    lr = exp['learning_rate']
    weight_decay = exp['weight_decay']
    batch_norm = exp['batch_norm']
    embedding_dim = exp['embedding_dim']
    hidden_layers = exp['hidden_layers']
    batch_size = exp['batch_size']
    patience = exp['patience']
    max_epochs = exp['max_epochs']
    optimizer = exp['optimizer']

    # --- Overfitting severo ---
    if 'overfitting_severe' in diag_ids:
        new_dropout = min(dropout + 0.15, 0.6)
        if new_dropout != dropout:
            recs.append({
                'param': 'Dropout', 'current': f'{dropout}',
                'suggested': f'{new_dropout:.2f}', 'direction': '⬆️',
                'reason': 'Aumentar olvido aleatorio para forzar generalización',
                'priority': 1
            })

        new_wd = weight_decay * 5 if weight_decay > 0 else 5e-4
        recs.append({
            'param': 'Weight Decay', 'current': f'{weight_decay:.0e}',
            'suggested': f'{new_wd:.0e}', 'direction': '⬆️',
            'reason': 'Mayor penalización contra pesos grandes',
            'priority': 1
        })

        if not batch_norm:
            recs.append({
                'param': 'Batch Norm', 'current': '❌ Desactivado',
                'suggested': '✅ Activar', 'direction': '✅',
                'reason': 'Estabiliza el entrenamiento y reduce overfitting',
                'priority': 1
            })

        new_emb = max(embedding_dim // 2, 16)
        if new_emb != embedding_dim:
            recs.append({
                'param': 'Embedding Dim', 'current': str(embedding_dim),
                'suggested': str(new_emb), 'direction': '⬇️',
                'reason': 'Reducir capacidad del modelo para evitar memorización',
                'priority': 2
            })

        if isinstance(hidden_layers, str):
            layers_list = [int(x) for x in hidden_layers.strip('[]').split(',')]
        else:
            layers_list = hidden_layers
        if len(layers_list) > 2:
            new_layers = layers_list[1:]  # Quitar capa más grande
            recs.append({
                'param': 'Capas Ocultas', 'current': str(layers_list),
                'suggested': str(new_layers), 'direction': '⬇️',
                'reason': 'Red más pequeña con menos riesgo de memorización',
                'priority': 2
            })

        if patience > 3:
            recs.append({
                'param': 'Paciencia', 'current': str(patience),
                'suggested': '2-3', 'direction': '⬇️',
                'reason': 'Parar antes para limitar el overfitting',
                'priority': 2
            })

    # --- Overfitting moderado ---
    elif 'overfitting_mild' in diag_ids:
        new_dropout = min(dropout + 0.05, 0.5)
        if new_dropout != dropout:
            recs.append({
                'param': 'Dropout', 'current': f'{dropout}',
                'suggested': f'{new_dropout:.2f}', 'direction': '⬆️',
                'reason': 'Ligero aumento de regularización',
                'priority': 2
            })

        new_wd = weight_decay * 2 if weight_decay > 0 else 1e-5
        recs.append({
            'param': 'Weight Decay', 'current': f'{weight_decay:.0e}',
            'suggested': f'{new_wd:.0e}', 'direction': '⬆️',
            'reason': 'Incrementar penalización moderadamente',
            'priority': 2
        })

        new_bs = min(batch_size * 2, 1024)
        if new_bs != batch_size:
            recs.append({
                'param': 'Batch Size', 'current': str(batch_size),
                'suggested': str(new_bs), 'direction': '⬆️',
                'reason': 'Más datos por paso = gradientes más estables',
                'priority': 3
            })

        if not batch_norm:
            recs.append({
                'param': 'Batch Norm', 'current': '❌ Desactivado',
                'suggested': '✅ Activar', 'direction': '✅',
                'reason': 'Mejora estabilidad y generalización',
                'priority': 2
            })

    # --- Underfitting ---
    if 'underfitting' in diag_ids:
        new_emb = min(embedding_dim * 2, 1024)
        if new_emb != embedding_dim:
            recs.append({
                'param': 'Embedding Dim', 'current': str(embedding_dim),
                'suggested': str(new_emb), 'direction': '⬆️',
                'reason': 'Más capacidad para representar usuarios/productos',
                'priority': 1
            })

        new_dropout = max(dropout - 0.1, 0.0)
        if new_dropout != dropout:
            recs.append({
                'param': 'Dropout', 'current': f'{dropout}',
                'suggested': f'{new_dropout:.2f}', 'direction': '⬇️',
                'reason': 'Menos regularización para que el modelo pueda aprender más',
                'priority': 1
            })

        if weight_decay > 1e-5:
            new_wd = weight_decay / 5
            recs.append({
                'param': 'Weight Decay', 'current': f'{weight_decay:.0e}',
                'suggested': f'{new_wd:.0e}', 'direction': '⬇️',
                'reason': 'Reducir penalización para dar más libertad al modelo',
                'priority': 2
            })

        new_epochs = min(max_epochs + 15, 80)
        if new_epochs > max_epochs:
            recs.append({
                'param': 'Epochs Máximas', 'current': str(max_epochs),
                'suggested': str(new_epochs), 'direction': '⬆️',
                'reason': 'Más rondas para aprender patrones complejos',
                'priority': 2
            })

    # --- Convergencia prematura ---
    if 'early_convergence' in diag_ids:
        # Solo reducir LR si no está ya muy bajo
        if lr > _THRESHOLDS['low_lr']:
            new_lr = max(lr / 2, _THRESHOLDS['low_lr'])
            recs.append({
                'param': 'Learning Rate', 'current': f'{lr}',
                'suggested': f'{new_lr:.4f}', 'direction': '⬇️',
                'reason': 'Pasos más pequeños para convergencia más gradual',
                'priority': 1
            })

        if not exp.get('lr_scheduler', False):
            recs.append({
                'param': 'LR Scheduler', 'current': '❌ Desactivado',
                'suggested': '✅ Activar', 'direction': '✅',
                'reason': 'Reducción automática de velocidad cuando se estanca',
                'priority': 1
            })

        new_patience = min(patience + 2, 10)
        if new_patience > patience:
            recs.append({
                'param': 'Paciencia', 'current': str(patience),
                'suggested': str(new_patience), 'direction': '⬆️',
                'reason': 'Más tiempo para explorar mejoras lentas',
                'priority': 2
            })

        if 'SGD' in optimizer:
            recs.append({
                'param': 'Optimizador', 'current': optimizer,
                'suggested': 'AdamW', 'direction': '🔄',
                'reason': 'AdamW adapta mejor el learning rate por parámetro',
                'priority': 2
            })

    # --- Inestabilidad ---
    if 'instability' in diag_ids:
        # Solo bajar LR si no está ya muy bajo
        if lr > _THRESHOLDS['low_lr']:
            new_lr = max(lr / 2, _THRESHOLDS['low_lr'])
            recs.append({
                'param': 'Learning Rate', 'current': f'{lr}',
                'suggested': f'{new_lr:.4f}', 'direction': '⬇️',
                'reason': 'Pasos más pequeños para entrenamiento más estable',
                'priority': 1
            })

        new_bs = min(batch_size * 2, 1024)
        if new_bs != batch_size:
            recs.append({
                'param': 'Batch Size', 'current': str(batch_size),
                'suggested': str(new_bs), 'direction': '⬆️',
                'reason': 'Gradientes más estables con más datos por paso',
                'priority': 1
            })

        gc = exp.get('gradient_clip', 5.0)
        if gc > 2.0:
            recs.append({
                'param': 'Gradient Clip', 'current': f'{gc}',
                'suggested': '2.0', 'direction': '⬇️',
                'reason': 'Limitar más los saltos bruscos del gradiente',
                'priority': 2
            })

        if not batch_norm:
            recs.append({
                'param': 'Batch Norm', 'current': '❌ Desactivado',
                'suggested': '✅ Activar', 'direction': '✅',
                'reason': 'Normalización para estabilizar el flujo de datos',
                'priority': 1
            })

    # --- Early stop prematuro ---
    if 'early_stop_premature' in diag_ids:
        new_patience = min(patience + 2, 8)
        # Evitar duplicado si ya se recomendó
        if not any(r['param'] == 'Paciencia' for r in recs):
            recs.append({
                'param': 'Paciencia', 'current': str(patience),
                'suggested': str(new_patience), 'direction': '⬆️',
                'reason': 'Dar más tiempo antes de parar el entrenamiento',
                'priority': 2
            })

        # Solo ajustar LR si no está ya muy bajo y no se recomendó antes
        if not any(r['param'] == 'Learning Rate' for r in recs) and lr > _THRESHOLDS['low_lr']:
            new_lr = max(lr * 0.7, _THRESHOLDS['low_lr'])
            recs.append({
                'param': 'Learning Rate', 'current': f'{lr}',
                'suggested': f'{new_lr:.4f}', 'direction': '⬇️',
                'reason': 'Convergencia más suave para aprovechar más epochs',
                'priority': 2
            })

        if not exp.get('lr_scheduler', False):
            if not any(r['param'] == 'LR Scheduler' for r in recs):
                recs.append({
                    'param': 'LR Scheduler', 'current': '❌ Desactivado',
                    'suggested': '✅ Activar', 'direction': '✅',
                    'reason': 'Auto-ajuste de velocidad cuando se estanca',
                    'priority': 2
                })

    # --- Podría seguir entrenando ---
    if 'could_train_more' in diag_ids:
        new_epochs = min(max_epochs + 10, 80)
        recs.append({
            'param': 'Epochs Máximas', 'current': str(max_epochs),
            'suggested': str(new_epochs), 'direction': '⬆️',
            'reason': 'El modelo aún mejoraba — darle más rondas',
            'priority': 2
        })

    # --- LR alto ---
    if 'high_lr' in diag_ids and not any(r['param'] == 'Learning Rate' for r in recs):
        recs.append({
            'param': 'Learning Rate', 'current': f'{lr}',
            'suggested': '0.001', 'direction': '⬇️',
            'reason': 'Un LR más conservador suele dar mejores resultados',
            'priority': 2
        })

    # --- LR bajo ---
    if 'low_lr' in diag_ids and not any(r['param'] == 'Learning Rate' for r in recs):
        recs.append({
            'param': 'Learning Rate', 'current': f'{lr}',
            'suggested': '0.001', 'direction': '⬆️',
            'reason': 'Subir LR si tienes pocas epochs disponibles',
            'priority': 3
        })

    # --- Divergencia ---
    if 'divergence' in diag_ids:
        # Reducir LR drásticamente (dividir por 5-10)
        if not any(r['param'] == 'Learning Rate' for r in recs):
            new_lr = max(lr / 10, 1e-5)
            recs.append({
                'param': 'Learning Rate', 'current': f'{lr}',
                'suggested': f'{new_lr:.5f}', 'direction': '⬇️',
                'reason': 'Reducción drástica para detener el empeoramiento',
                'priority': 1
            })

        gc = exp.get('gradient_clip', 5.0)
        if gc > 1.0:
            recs.append({
                'param': 'Gradient Clip', 'current': f'{gc}',
                'suggested': '1.0', 'direction': '⬇️',
                'reason': 'Limitar gradientes explosivos más estrictamente',
                'priority': 1
            })

        # Considerar parar y revisar arquitectura
        recs.append({
            'param': '🛑 Acción', 'current': 'Continuar entrenamiento',
            'suggested': 'Detener y revisar configuración', 'direction': '⚠️',
            'reason': 'El modelo está divergiendo — revisar antes de continuar',
            'priority': 1
        })

    # --- Estancamiento ---
    if 'stagnation' in diag_ids:
        if patience > 2:
            recs.append({
                'param': 'Paciencia', 'current': str(patience),
                'suggested': '2-3', 'direction': '⬇️',
                'reason': 'Parar antes cuando no hay mejoras significativas',
                'priority': 2
            })

        if not exp.get('lr_scheduler', False):
            recs.append({
                'param': 'LR Scheduler', 'current': '❌ Desactivado',
                'suggested': '✅ Activar', 'direction': '✅',
                'reason': 'Reducción automática de LR cuando se estanca',
                'priority': 2
            })

    # --- Test vs Val Discrepancy ---
    if 'test_val_discrepancy' in diag_ids:
        # Aumentar regularización general
        if dropout < 0.5:
            new_dropout = min(dropout + 0.15, 0.5)
            if not any(r['param'] == 'Dropout' for r in recs):
                recs.append({
                    'param': 'Dropout', 'current': f'{dropout}',
                    'suggested': f'{new_dropout:.2f}', 'direction': '⬆️',
                    'reason': 'Mejorar generalización para test set',
                    'priority': 1
                })

        new_wd = max(weight_decay * 3, 1e-4) if weight_decay > 0 else 1e-4
        if not any(r['param'] == 'Weight Decay' for r in recs):
            recs.append({
                'param': 'Weight Decay', 'current': f'{weight_decay:.0e}',
                'suggested': f'{new_wd:.0e}', 'direction': '⬆️',
                'reason': 'Aumentar penalización para mejor generalización',
                'priority': 2
            })

    # --- Mejora solo en Train ---
    if 'train_only_improvement' in diag_ids:
        # Muy similar a overfitting moderado, pero más urgente
        if not any(r['param'] == 'Dropout' for r in recs):
            new_dropout = min(dropout + 0.1, 0.6)
            if new_dropout != dropout:
                recs.append({
                    'param': 'Dropout', 'current': f'{dropout}',
                    'suggested': f'{new_dropout:.2f}', 'direction': '⬆️',
                    'reason': 'Frenar memorización en progreso',
                    'priority': 1
                })

        # Considerar parar el entrenamiento ahora
        recs.append({
            'param': '⏸️ Considerar', 'current': 'Continuar entrenamiento',
            'suggested': 'Parar ahora o en 2-3 epochs', 'direction': '⏸️',
            'reason': 'El overfitting ya comenzó — pocas mejoras posibles',
            'priority': 2
        })

    # --- Capacidad excesiva ---
    if 'model_too_large' in diag_ids:
        # Reducir tamaño del modelo
        if embedding_dim > 64:
            new_emb = max(embedding_dim // 2, 32)
            if not any(r['param'] == 'Embedding Dim' for r in recs):
                recs.append({
                    'param': 'Embedding Dim', 'current': str(embedding_dim),
                    'suggested': str(new_emb), 'direction': '⬇️',
                    'reason': 'Modelo más pequeño apropiado para el dataset',
                    'priority': 2
                })

        if not any(r['param'] == 'Dropout' for r in recs) and dropout < 0.4:
            new_dropout = min(dropout + 0.15, 0.5)
            recs.append({
                'param': 'Dropout', 'current': f'{dropout}',
                'suggested': f'{new_dropout:.2f}', 'direction': '⬆️',
                'reason': 'Más regularización para modelo grande',
                'priority': 2
            })

    # --- Excelente ---
    if 'excellent' in diag_ids:
        recs.append({
            'param': '💾 Modelo', 'current': 'Sin guardar',
            'suggested': 'Guardar como baseline', 'direction': '🏆',
            'reason': '¡Gran resultado! Guárdalo para comparar en futuras sesiones',
            'priority': 3
        })

    # Deduplicar por param (mantener el de mayor prioridad)
    seen = {}
    unique_recs = []
    for r in sorted(recs, key=lambda x: x['priority']):
        if r['param'] not in seen:
            seen[r['param']] = True
            unique_recs.append(r)

    return sorted(unique_recs, key=lambda x: x['priority'])


def _build_natural_summary(recs, exp):
    """Construye un resumen en lenguaje natural de las recomendaciones principales."""
    if not recs:
        return ""

    high_priority = [r for r in recs if r['priority'] == 1]
    if not high_priority:
        high_priority = recs[:2]

    parts = []
    for r in high_priority[:3]:
        if r['direction'] in ('⬆️', '⬇️'):
            action = "Sube" if r['direction'] == '⬆️' else "Baja"
            parts.append(f"{action} **{r['param']}** a {r['suggested']}")
        elif r['direction'] == '✅':
            parts.append(f"Activa **{r['param']}**")
        elif r['direction'] == '🔄':
            parts.append(f"Cambia a **{r['suggested']}**")

    if not parts:
        return ""

    joined = ", ".join(parts[:-1])
    if len(parts) > 1:
        joined += f" y {parts[-1]}"
    else:
        joined = parts[0]

    return f"**Próximo paso:** {joined}."


def _render_experiment_recommendations(exp):
    """Renderiza tarjeta de recomendaciones para un experimento individual."""
    diagnostics = _diagnose_experiment(exp)
    recs = _generate_recommendations(exp, diagnostics)

    # Banner general
    gap = exp['history']['val_rmse'][-1] - exp['history']['train_rmse'][-1]
    test_rmse = exp['final_test_rmse']

    if test_rmse < 0.70 and gap < 0.10:
        health = "🟢"
        health_text = "Excelente"
    elif test_rmse < 0.75 and gap < 0.20:
        health = "🟡"
        health_text = "Bueno con margen de mejora"
    elif test_rmse < 0.85:
        health = "🟠"
        health_text = "Necesita ajustes"
    else:
        health = "🔴"
        health_text = "Requiere cambios significativos"

    st.markdown(
        f"**{health} Estado general:** {health_text} — "
        f"Test RMSE: **{test_rmse:.4f}** | Gap: **{gap:.3f}**"
    )

    # Diagnósticos
    if diagnostics:
        st.markdown("**🩺 Diagnósticos:**")
        for d in diagnostics:
            icon = {
                'critical': '🔴',
                'warning': '🟡',
                'info': 'ℹ️',
                'success': '🟢'
            }.get(d['severity'], 'ℹ️')
            st.markdown(f"  {icon} **{d['title']}** — {d['description']}")
    else:
        st.markdown("ℹ️ No se detectaron problemas específicos.")

    # Tabla de recomendaciones
    if recs:
        st.markdown("**🔧 Recomendaciones concretas:**")
        rec_table = []
        for r in recs:
            priority_icon = {1: '🔴 Alta', 2: '🟡 Media', 3: '🟢 Baja'}.get(r['priority'], '🟢')
            rec_table.append({
                'Prioridad': priority_icon,
                'Parámetro': f"{r['direction']} {r['param']}",
                'Actual': r['current'],
                'Sugerido': r['suggested'],
                'Razón': r['reason']
            })
        st.dataframe(pd.DataFrame(rec_table), use_container_width=True, hide_index=True)

        # Resumen natural
        summary = _build_natural_summary(recs, exp)
        if summary:
            st.info(f"💬 {summary}")
    else:
        st.success("🎉 ¡No hay recomendaciones pendientes! Este modelo tiene un gran rendimiento.")


def _render_recommendations_tab():
    """Renderiza la pestaña completa de mejoras recomendadas."""
    if len(st.session_state.experiments) > 0:
        st.markdown("### 💡 Mejoras Recomendadas por Experimento")
        st.caption(
            "El sistema analiza automáticamente las métricas y curvas de cada entrenamiento "
            "para generar recomendaciones específicas de hiperparámetros. "
            "Las sugerencias se basan en patrones comunes de overfitting, underfitting, "
            "inestabilidad y convergencia."
        )

        # Si hay múltiples experimentos, mostrar resumen comparativo
        if len(st.session_state.experiments) > 1:
            best_exp = min(st.session_state.experiments, key=lambda x: x['final_test_rmse'])
            worst_exp = max(st.session_state.experiments, key=lambda x: x['final_test_rmse'])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🏆 Mejor Experimento",
                    f"#{best_exp['id']} — {best_exp['final_test_rmse']:.4f}",
                    help="Experimento con menor Test RMSE"
                )
            with col2:
                improvement = worst_exp['final_test_rmse'] - best_exp['final_test_rmse']
                st.metric(
                    "📊 Rango de mejora",
                    f"{improvement:.4f} RMSE",
                    help="Diferencia entre el peor y el mejor experimento"
                )
            with col3:
                avg_gap = np.mean([
                    e['history']['val_rmse'][-1] - e['history']['train_rmse'][-1]
                    for e in st.session_state.experiments
                ])
                gap_icon = "🟢" if avg_gap < 0.10 else "🟡" if avg_gap < 0.20 else "🔴"
                st.metric(
                    f"{gap_icon} Gap Promedio",
                    f"{avg_gap:.3f}",
                    help="Promedio del gap Train-Val de todos los experimentos"
                )

            st.divider()

        # Mostrar cada experimento
        for exp in st.session_state.experiments:
            exp_label = (
                f"Experimento #{exp['id']} — "
                f"Test RMSE: {exp['final_test_rmse']:.4f} | "
                f"Red: Emb {exp['embedding_dim']} → {exp['hidden_layers']}"
            )
            with st.expander(exp_label, expanded=(len(st.session_state.experiments) == 1)):
                _render_experiment_recommendations(exp)
    else:
        st.info(
            "💡 **Aquí aparecerán recomendaciones personalizadas** para cada entrenamiento. "
            "Entrena uno o más modelos con diferentes hiperparámetros y el sistema "
            "analizará automáticamente las métricas para sugerirte mejoras concretas."
        )


def render_results(experiment, all_preds, all_actuals):
    """Renderizar resultados detallados del entrenamiento"""
    st.subheader("📊 Resultados Detallados")

    history = experiment['history']
    final_test_rmse = experiment['final_test_rmse']
    best_val_rmse = experiment['best_val_rmse']
    best_epoch = experiment['best_epoch']
    total_epochs_run = experiment['total_epochs']

    tab_curves, tab_preds, tab_compare, tab_recommendations = st.tabs([
        "📈 Curvas de Entrenamiento",
        "🎯 Predicciones",
        "🔬 Comparación de Experimentos",
        "💡 Mejoras Recomendadas"
    ])

    with tab_curves:
        # Explicación general
        st.info(
            "📖 **¿Cómo leer estos gráficos?** "
            "La línea **azul** muestra qué tan bien el modelo acierta con datos que ya conoce (Train). "
            "La línea **naranja** muestra qué tan bien acierta con datos de **validación** (Validation). "
            "Lo ideal es que **ambas líneas bajen juntas**. Si la azul baja mucho y la naranja sube, "
            "el modelo está memorizando en vez de aprender. El test final se evalúa al terminar."
        )

        # Métricas finales destacadas
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("🏆 Test RMSE Final", f"{final_test_rmse:.4f}",
                     help="Error en datos completamente nuevos (nunca vistos durante entrenamiento)")
        with metric_cols[1]:
            st.metric("✅ Mejor Val RMSE", f"{best_val_rmse:.4f}",
                     help=f"Mejor error en validación (epoch {best_epoch})")
        with metric_cols[2]:
            generalization_gap = final_test_rmse - best_val_rmse
            gap_emoji = "🟢" if abs(generalization_gap) < 0.05 else "🟡" if abs(generalization_gap) < 0.10 else "🔴"
            st.metric(f"{gap_emoji} Gap Test-Val", f"{generalization_gap:+.4f}",
                     help="Diferencia entre test y validación. Cerca de 0 = buena generalización")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de RMSE completo
            epochs_range = list(range(1, total_epochs_run + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs_range, y=history['train_rmse'],
                mode='lines+markers',
                name='Datos conocidos (Train)',
                line=dict(color='#2E86AB', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=epochs_range, y=history['val_rmse'],
                mode='lines+markers',
                name='Datos de validación (Val)',
                line=dict(color='#F18F01', width=2)
            ))
            fig.add_vline(x=best_epoch, line_dash="dash",
                         line_color="green",
                         annotation_text=f"Mejor punto: {best_val_rmse:.4f}")
            fig.update_layout(
                title='📉 ¿Cuánto se equivoca el modelo? (menor = mejor)',
                xaxis_title='Ronda de aprendizaje',
                yaxis_title='Error promedio (RMSE)',
                height=400, template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Salud del modelo
            gap_values = np.array(history['val_rmse']) - np.array(history['train_rmse'])
            final_gap = gap_values[-1]

            fig = go.Figure()

            # Zonas de colores de fondo
            fig.add_hrect(y0=0, y1=0.10, fillcolor="#06A77D", opacity=0.12,
                         line_width=0, annotation_text="Aprendiendo bien",
                         annotation_position="top left",
                         annotation_font_color="#06A77D")
            fig.add_hrect(y0=0.10, y1=0.20, fillcolor="#F18F01", opacity=0.12,
                         line_width=0, annotation_text="Empezando a memorizar",
                         annotation_position="top left",
                         annotation_font_color="#F18F01")
            fig.add_hrect(y0=0.20, y1=max(0.40, max(gap_values) * 1.2),
                         fillcolor="#D62828", opacity=0.12,
                         line_width=0, annotation_text="Memorizando datos",
                         annotation_position="top left",
                         annotation_font_color="#D62828")

            # Colorear la línea según la zona
            line_colors = []
            for g in gap_values:
                if g < 0.10:
                    line_colors.append('#06A77D')
                elif g < 0.20:
                    line_colors.append('#F18F01')
                else:
                    line_colors.append('#D62828')

            # Línea principal
            fig.add_trace(go.Scatter(
                x=epochs_range, y=gap_values,
                mode='lines+markers',
                name='Salud del modelo',
                line=dict(color=line_colors[-1], width=3),
                marker=dict(color=line_colors, size=8)
            ))

            fig.update_layout(
                title='🩺 ¿Está aprendiendo o memorizando?',
                xaxis_title='Época',
                yaxis_title='Diferencia de error',
                yaxis_range=[0, max(0.40, max(gap_values) * 1.2)],
                height=400, template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interpretación automática
            if final_gap < 0.10:
                st.success(
                    "🟢 **El modelo está aprendiendo correctamente.** "
                    "Lo que aprende con los datos de entrenamiento también "
                    "funciona con datos que nunca ha visto."
                )
            elif final_gap < 0.20:
                st.warning(
                    "🟡 **El modelo empieza a memorizar** en vez de aprender patrones. "
                    "Prueba a subir el Dropout, activar Batch Normalization, "
                    "o reducir el tamaño de la red."
                )
            else:
                st.error(
                    "🔴 **El modelo está memorizando los datos** en lugar de aprender. "
                    "Se sabe las respuestas de entrenamiento de memoria, pero falla "
                    "con datos nuevos. Sube el Dropout, reduce Epochs o usa una red más pequeña."
                )

        # Learning rate evolution
        if experiment['lr_scheduler']:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs_range, y=history['learning_rates'],
                mode='lines+markers', name='Velocidad de aprendizaje',
                line=dict(color='#06A77D', width=2),
                fill='tozeroy', fillcolor='rgba(6, 167, 125, 0.1)'
            ))
            fig.update_layout(
                title='🎚️ Velocidad de Aprendizaje (se reduce cuando el modelo se estanca)',
                xaxis_title='Ronda de aprendizaje',
                yaxis_title='Velocidad',
                yaxis_type='log',
                height=300, template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Cuando el modelo deja de mejorar, el sistema reduce "
                "automáticamente la velocidad de aprendizaje para hacer ajustes más finos."
            )

    with tab_preds:
        st.info(
            "📖 **¿Cómo leer estos gráficos?** "
            "A la izquierda: cada punto es una predicción. Si el modelo fuera perfecto, "
            "todos los puntos estarían sobre la línea roja diagonal. "
            "A la derecha: muestra cuánto se equivoca y con qué frecuencia. "
            "Lo ideal es una campana centrada en 0 (sin error)."
        )

        col1, col2 = st.columns(2)

        errors = all_actuals - all_preds

        with col1:
            # Scatter: actual vs predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=all_actuals, y=all_preds,
                mode='markers',
                marker=dict(size=4, opacity=0.3, color='#2E86AB'),
                name='Predicciones'
            ))
            fig.add_trace(go.Scatter(
                x=[1, 5], y=[1, 5],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Línea de acierto perfecto'
            ))
            fig.update_layout(
                title='🎯 ¿Qué tan cerca están las predicciones?',
                xaxis_title='Rating que el usuario dio realmente',
                yaxis_title='Rating que el modelo predijo',
                height=450, template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Distribución de errores
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=errors, nbinsx=40,
                marker_color='#A23B72',
                name='Predicciones'
            ))
            fig.add_vline(x=0, line_dash="dash", line_color="green",
                         annotation_text="Sin error",
                         annotation_position="top")
            fig.add_vrect(x0=-0.5, x1=0.5, fillcolor="green", opacity=0.08,
                         line_width=0, annotation_text="Muy buenas",
                         annotation_position="top left",
                         annotation_font_color="green")
            fig.update_layout(
                title='📊 ¿Cuánto se equivoca el modelo?',
                xaxis_title='Error (negativo = predijo de más, positivo = predijo de menos)',
                yaxis_title='Cantidad de predicciones',
                height=450, template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Métricas resumidas
        mae = np.mean(np.abs(errors))
        rmse_val = np.sqrt(np.mean(errors**2))
        pct_within_half = (np.abs(errors) < 0.5).mean() * 100
        pct_within_one = (np.abs(errors) < 1.0).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Error promedio", f"{mae:.2f} ⭐",
                     help="En promedio, el modelo se equivoca por esta cantidad de estrellas")
        with col2:
            st.metric("RMSE", f"{rmse_val:.4f}",
                     help="Métrica técnica de error. Menor = mejor")
        with col3:
            st.metric("Aciertos cercanos", f"{pct_within_half:.0f}%",
                     help="Predicciones con menos de media estrella de error")
        with col4:
            st.metric("Aciertos razonables", f"{pct_within_one:.0f}%",
                     help="Predicciones con menos de 1 estrella de error")

        # Interpretación
        if mae < 0.5:
            st.success(
                f"🎉 **Excelente precisión.** El modelo se equivoca en promedio "
                f"por solo **{mae:.2f} estrellas** y el **{pct_within_half:.0f}%** "
                f"de las predicciones están a menos de media estrella del valor real."
            )
        elif mae < 0.8:
            st.info(
                f"👍 **Buena precisión.** Error promedio de **{mae:.2f} estrellas**. "
                f"El **{pct_within_one:.0f}%** de predicciones están a menos "
                f"de 1 estrella del valor real."
            )
        else:
            st.warning(
                f"⚠️ **Precisión mejorable.** Error promedio de **{mae:.2f} estrellas**. "
                f"Prueba otros hiperparámetros para reducir el error."
            )

        # Muestra de predicciones
        st.markdown("**🎯 Ejemplos de Predicciones:**")
        sample_indices = np.random.choice(len(all_actuals), min(10, len(all_actuals)), replace=False)
        sample_data = []
        for i in sample_indices:
            error_abs = abs(all_actuals[i] - all_preds[i])
            if error_abs < 0.5:
                accuracy = "🟢 Excelente"
            elif error_abs < 1.0:
                accuracy = "🟡 Buena"
            else:
                accuracy = "🔴 Lejos"
            sample_data.append({
                'Rating Real': f"{all_actuals[i]:.1f} ⭐",
                'Rating Predicho': f"{all_preds[i]:.2f} ⭐",
                'Diferencia': f"{error_abs:.2f}",
                'Precisión': accuracy
            })
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)

    with tab_compare:
        if len(st.session_state.experiments) > 0:
            st.markdown("**📊 Historial de Experimentos:**")
            st.caption(
                "Cada fila es un entrenamiento que hiciste con diferentes configuraciones. "
                "Compara el \"Error\" (RMSE) para ver cuál funciona mejor."
            )

            # Tabla comparativa
            exp_table = []
            for exp in st.session_state.experiments:
                gap = exp['history']['test_rmse'][-1] - exp['history']['train_rmse'][-1]
                if gap < 0.10:
                    salud = "🟢 Saludable"
                elif gap < 0.20:
                    salud = "🟡 Atención"
                else:
                    salud = "🔴 Memorizando"

                exp_table.append({
                    '#': exp['id'],
                    'Red': f"Emb {exp['embedding_dim']} → {exp['hidden_layers']}",
                    'Dropout': exp['dropout'],
                    'Batch Norm': '✅' if exp['batch_norm'] else '❌',
                    'Velocidad (LR)': exp['learning_rate'],
                    'Optimizador': exp['optimizer'],
                    'Rondas': f"{exp['total_epochs']}/{exp['max_epochs']}",
                    'Paró antes': '✋ Manual' if exp.get('stopped_manually', False) else ('🛑 Auto' if exp['stopped_early'] else '✅ No'),
                    'Error Test (RMSE)': f"{exp['final_test_rmse']:.4f}",
                    'Salud': salud,
                    'Tiempo': f"{exp['train_time']}s"
                })

            exp_df = pd.DataFrame(exp_table)
            st.dataframe(exp_df, use_container_width=True, hide_index=True)

            # Gráfico de comparación
            if len(st.session_state.experiments) > 1:
                st.markdown("**📈 ¿Cuál aprende mejor?**")
                st.caption("Compara las curvas de validación de cada experimento. El número en la leyenda muestra el RMSE final en el test set.")
                fig = go.Figure()

                colors = px.colors.qualitative.Set2
                for i, exp in enumerate(st.session_state.experiments):
                    color = colors[i % len(colors)]
                    exp_epochs = list(range(1, len(exp['history']['val_rmse']) + 1))
                    fig.add_trace(go.Scatter(
                        x=exp_epochs,
                        y=exp['history']['val_rmse'],
                        mode='lines+markers',
                        name=f"#{exp['id']} — Test Final: {exp['final_test_rmse']:.4f}",
                        line=dict(color=color, width=2)
                    ))

                fig.update_layout(
                    title='Curvas de Validación por Experimento',
                    xaxis_title='Ronda de aprendizaje (Epoch)',
                    yaxis_title='Validation RMSE',
                    height=450,
                    template='plotly_white',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)

            # Ranking
            st.markdown("**🏆 Ranking (por Test RMSE):**")
            ranking = sorted(st.session_state.experiments,
                           key=lambda x: x['final_test_rmse'])
            for i, exp in enumerate(ranking):
                medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"#{i+1}"
                st.markdown(
                    f"{medal} **Experimento #{exp['id']}** — "
                    f"Test RMSE: **{exp['final_test_rmse']:.4f}** — "
                    f"Red: Emb {exp['embedding_dim']} → {exp['hidden_layers']}, "
                    f"Dropout: {exp['dropout']}, LR: {exp['learning_rate']}"
                )

            if st.button("🗑️ Limpiar Historial de Experimentos"):
                st.session_state.experiments = []
                st.rerun()
        else:
            st.info(
                "Aún no hay experimentos. Configura los hiperparámetros arriba "
                "y pulsa **Entrenar Modelo** para ver resultados aquí. "
                "Puedes entrenar varias veces con diferentes configuraciones "
                "y este panel te mostrará cuál funciona mejor."
            )

    with tab_recommendations:
        _render_recommendations_tab()
