"""
Módulo de entrenamiento NCF
Bucle de entrenamiento con UI en tiempo real, early stopping y guardado de modelos
"""
import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
from pathlib import Path
import sys

# Asegurar imports
WEB_DIR = Path(__file__).parent.parent.parent
PROJECT_DIR = WEB_DIR.parent
sys.path.insert(0, str(WEB_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from config import MODELS_DIR
from core.ncf_models import NCFDataset, NeuralCollaborativeFiltering, EarlyStopping

# PyTorch imports (ya validados en el módulo padre)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def run_training(ncf_data, hyperparams):
    """
    Ejecutar entrenamiento del modelo NCF con los hiperparámetros dados.
    Retorna: (experiment_dict, all_preds, all_actuals)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Crear datasets para Train, Validation y Test
    train_dataset = NCFDataset(
        ncf_data['train_df']['user_idx'].values,
        ncf_data['train_df']['product_idx'].values,
        ncf_data['train_df']['rating'].values
    )
    val_dataset = NCFDataset(
        ncf_data['val_df']['user_idx'].values,
        ncf_data['val_df']['product_idx'].values,
        ncf_data['val_df']['rating'].values
    )
    test_dataset = NCFDataset(
        ncf_data['test_df']['user_idx'].values,
        ncf_data['test_df']['product_idx'].values,
        ncf_data['test_df']['rating'].values
    )

    batch_size = hyperparams['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Crear modelo
    model = NeuralCollaborativeFiltering(
        n_users=ncf_data['n_users'],
        n_products=ncf_data['n_products'],
        embedding_dim=hyperparams['embedding_dim'],
        hidden_layers=hyperparams['hidden_layers'],
        dropout=hyperparams['dropout'],
        use_batch_norm=hyperparams['use_batch_norm']
    ).to(device)

    # Optimizador
    optimizer_choice = hyperparams['optimizer_choice']
    learning_rate = hyperparams['learning_rate']
    weight_decay = hyperparams['weight_decay']

    if "AdamW" in optimizer_choice:
        opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif "Adam" in optimizer_choice:
        opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        opt = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    criterion = nn.MSELoss()

    # Scheduler
    scheduler = None
    if hyperparams['use_scheduler']:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2, min_lr=1e-6)

    early_stopping = EarlyStopping(patience=hyperparams['patience'])

    # Historial
    history = {
        'train_rmse': [], 'val_rmse': [], 'test_rmse': [],
        'learning_rates': [], 'epoch_times': []
    }

    # UI de progreso
    st.markdown("### 🏋️ Entrenamiento en Progreso...")

    # Control de detención manual
    if 'stop_training' not in st.session_state:
        st.session_state.stop_training = False

    # Botón para detener entrenamiento
    stop_col1, stop_col2 = st.columns([1, 4])
    with stop_col1:
        if st.button("🛑 Detener", type="secondary", help="Detener el entrenamiento manualmente"):
            st.session_state.stop_training = True
    with stop_col2:
        if st.session_state.stop_training:
            st.warning("⚠️ Deteniendo el entrenamiento...")

    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_cols = st.columns(4)
    chart_placeholder = st.empty()

    start_time = time.time()
    stopped_early = False
    stopped_manually = False
    epochs = hyperparams['epochs']
    gradient_clip = hyperparams['gradient_clip']

    for epoch in range(epochs):
        # Verificar detención manual
        if st.session_state.stop_training:
            stopped_manually = True
            st.session_state.stop_training = False  # Reset para próximos entrenamientos
            break
        epoch_start = time.time()

        # === ENTRENAMIENTO ===
        model.train()
        train_loss = 0.0
        for users, products, ratings in train_loader:
            users, products, ratings = users.to(device), products.to(device), ratings.to(device)
            predictions = model(users, products)
            loss = criterion(predictions, ratings)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            opt.step()
            train_loss += loss.item() * len(users)

        train_loss /= len(train_dataset)
        train_rmse = np.sqrt(train_loss)

        # === VALIDACIÓN ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for users, products, ratings in val_loader:
                users, products, ratings = users.to(device), products.to(device), ratings.to(device)
                predictions = model(users, products)
                loss = criterion(predictions, ratings)
                val_loss += loss.item() * len(users)

        val_loss /= len(val_dataset)
        val_rmse = np.sqrt(val_loss)

        current_lr = opt.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        if scheduler:
            scheduler.step(val_rmse)

        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)

        # Actualizar UI
        progress_bar.progress((epoch + 1) / epochs)
        status_text.markdown(
            f"**Epoch {epoch+1}/{epochs}** — "
            f"Train RMSE: `{train_rmse:.4f}` | "
            f"Val RMSE: `{val_rmse:.4f}` | "
            f"LR: `{current_lr:.6f}` | "
            f"Tiempo: `{epoch_time:.1f}s`"
        )

        # Actualizar métricas
        with metrics_cols[0]:
            st.metric("Train RMSE", f"{train_rmse:.4f}",
                     delta=f"{train_rmse - history['train_rmse'][-2]:.4f}" if epoch > 0 else None)
        with metrics_cols[1]:
            st.metric("Val RMSE", f"{val_rmse:.4f}",
                     delta=f"{val_rmse - history['val_rmse'][-2]:.4f}" if epoch > 0 else None,
                     delta_color="inverse")
        with metrics_cols[2]:
            st.metric("Mejor Val RMSE", f"{min(history['val_rmse']):.4f}")
        with metrics_cols[3]:
            gap = val_rmse - train_rmse
            if gap < 0.10:
                health_label, health_icon = "Saludable", "🟢"
            elif gap < 0.20:
                health_label, health_icon = "Atención", "🟡"
            else:
                health_label, health_icon = "Memorizando", "🔴"
            st.metric(f"{health_icon} Salud", health_label)

        # Gráfico en tiempo real
        with chart_placeholder.container():
            epochs_range = list(range(1, len(history['train_rmse']) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=epochs_range, y=history['train_rmse'],
                mode='lines+markers', name='Train RMSE',
                line=dict(color='#2E86AB', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=epochs_range, y=history['val_rmse'],
                mode='lines+markers', name='Validation RMSE',
                line=dict(color='#F18F01', width=2)
            ))
            fig.update_layout(
                title='Curvas de Entrenamiento (RMSE)',
                xaxis_title='Época',
                yaxis_title='RMSE',
                height=350,
                template='plotly_white',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Early stopping
        early_stopping(val_rmse, epoch + 1)
        if early_stopping.early_stop:
            stopped_early = True
            break

    # === EVALUACIÓN FINAL EN TEST SET ===
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for users, products, ratings in test_loader:
            users, products, ratings = users.to(device), products.to(device), ratings.to(device)
            predictions = model(users, products)
            loss = criterion(predictions, ratings)
            test_loss += loss.item() * len(users)

    test_loss /= len(test_dataset)
    final_test_rmse = np.sqrt(test_loss)
    history['test_rmse'].append(final_test_rmse)

    total_time = time.time() - start_time
    best_val_rmse = min(history['val_rmse'])
    best_epoch = np.argmin(history['val_rmse']) + 1
    total_epochs_run = len(history['val_rmse'])

    progress_bar.progress(1.0)

    if stopped_manually:
        status_text.markdown(
            f"✋ **Detenido manualmente** en epoch {total_epochs_run} — "
            f"Mejor Val RMSE: **{best_val_rmse:.4f}** (epoch {best_epoch}) — "
            f"**Test Final RMSE: {final_test_rmse:.4f}** — "
            f"Tiempo total: **{total_time:.1f}s**"
        )
    elif stopped_early:
        status_text.markdown(
            f"🛑 **Early Stopping** en epoch {total_epochs_run} — "
            f"Mejor Val RMSE: **{best_val_rmse:.4f}** (epoch {best_epoch}) — "
            f"**Test Final RMSE: {final_test_rmse:.4f}** — "
            f"Tiempo total: **{total_time:.1f}s**"
        )
    else:
        status_text.markdown(
            f"✅ **Entrenamiento completado** ({total_epochs_run} epochs) — "
            f"Mejor Val RMSE: **{best_val_rmse:.4f}** (epoch {best_epoch}) — "
            f"**Test Final RMSE: {final_test_rmse:.4f}** — "
            f"Tiempo total: **{total_time:.1f}s**"
        )

    # Guardar en session state
    experiment = {
        'id': len(st.session_state.experiments) + 1,
        'embedding_dim': hyperparams['embedding_dim'],
        'hidden_layers': str(hyperparams['hidden_layers']),
        'dropout': hyperparams['dropout'],
        'batch_norm': hyperparams['use_batch_norm'],
        'learning_rate': hyperparams['learning_rate'],
        'optimizer': hyperparams['optimizer_choice'].split(' ')[0],
        'weight_decay': hyperparams['weight_decay'],
        'batch_size': hyperparams['batch_size'],
        'max_epochs': hyperparams['epochs'],
        'patience': hyperparams['patience'],
        'gradient_clip': hyperparams['gradient_clip'],
        'lr_scheduler': hyperparams['use_scheduler'],
        'total_params': hyperparams['total_params'],
        'n_train_samples': len(train_dataset),  # Para diagnóstico de capacidad del modelo
        'best_val_rmse': best_val_rmse,
        'final_test_rmse': final_test_rmse,
        'best_epoch': best_epoch,
        'total_epochs': total_epochs_run,
        'stopped_early': stopped_early,
        'stopped_manually': stopped_manually,
        'train_time': round(total_time, 1),
        'history': history
    }
    st.session_state.experiments.append(experiment)
    st.session_state.current_history = history
    st.session_state.training_done = True

    # Opción para guardar modelo
    st.success("🎉 Entrenamiento completado exitosamente")

    col_save1, col_save2 = st.columns([2, 1])
    with col_save1:
        model_name = st.text_input(
            "💾 Nombre del modelo (opcional)",
            value=f"ncf_exp{experiment['id']}_test{final_test_rmse:.4f}",
            help="Si deseas guardar este modelo para usarlo después"
        )
    with col_save2:
        st.write("")
        st.write("")
        if st.button("💾 Guardar Modelo (.pth)", type="secondary"):
            model_path = MODELS_DIR / f"{model_name}.pth"
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'n_users': ncf_data['n_users'],
                'n_products': ncf_data['n_products'],
                'embedding_dim': hyperparams['embedding_dim'],
                'hidden_layers': hyperparams['hidden_layers'],
                'dropout': hyperparams['dropout'],
                'use_batch_norm': hyperparams['use_batch_norm'],
                'user_to_idx': ncf_data['user_to_idx'],
                'product_to_idx': ncf_data['product_to_idx'],
                'idx_to_user': ncf_data['idx_to_user'],
                'idx_to_product': ncf_data['idx_to_product'],
                'best_val_rmse': best_val_rmse,
                'final_test_rmse': final_test_rmse,
                'best_epoch': best_epoch,
                'total_params': hyperparams['total_params'],
                'history': history,
                'hyperparameters': hyperparams
            }
            torch.save(checkpoint, str(model_path))
            st.success(f"✅ Modelo guardado en: `models/{model_name}.pth` ({model_path.stat().st_size / 1024:.1f} KB)")
            st.info("💡 Este archivo .pth contiene todos los pesos del modelo y puede ser cargado en notebooks o futuras sesiones.")

    st.divider()

    # Obtener predicciones completas para visualización
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for users, products, ratings in test_loader:
            users, products, ratings = users.to(device), products.to(device), ratings.to(device)
            preds = model(users, products)
            all_preds.extend(preds.cpu().numpy())
            all_actuals.extend(ratings.cpu().numpy())
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)

    return experiment, all_preds, all_actuals
