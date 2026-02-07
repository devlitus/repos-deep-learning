"""
Sistema Híbrido Profundo - Deep Hybrid Recommender System
Combina métodos tradicionales (User-CF, Item-CF, SVD) con Deep Learning (NCF)
usando PyTorch para aprender pesos automáticamente via attention mechanism.
"""

import sys
import io
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split as sklearn_train_test_split

# Configurar UTF-8
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

# Importar config
import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))
import config


# =============================================================================
# DATASET PARA PYTORCH
# =============================================================================

class RatingsDataset(Dataset):
    """Dataset de PyTorch para ratings de usuarios"""

    def __init__(self, user_ids, item_ids, ratings):
        """
        Args:
            user_ids: array de índices de usuarios
            item_ids: array de índices de productos
            ratings: array de ratings
        """
        self.user_ids = torch.LongTensor(user_ids)
        self.item_ids = torch.LongTensor(item_ids)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.ratings[idx]


# =============================================================================
# NEURAL COLLABORATIVE FILTERING (NCF)
# =============================================================================

class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering Model
    Aprende embeddings de usuarios y productos a través de capas neuronales
    """

    def __init__(self, num_users, num_items, embedding_dim=64,
                 hidden_dims=[128, 64, 32], dropout_rate=0.2):
        """
        Args:
            num_users: número de usuarios únicos
            num_items: número de productos únicos
            embedding_dim: dimensión de embeddings
            hidden_dims: lista de dimensiones de capas ocultas
            dropout_rate: tasa de dropout para regularización
        """
        super(NeuralCollaborativeFiltering, self).__init__()

        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        # Inicialización Xavier
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # MLP layers
        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim

        # Capa final (predice rating)
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        print(f"  ✅ NCF construido: {num_users} users, {num_items} items, {embedding_dim}D embeddings")

    def forward(self, user_ids, item_ids):
        """
        Forward pass

        Args:
            user_ids: tensor de índices de usuarios [batch_size]
            item_ids: tensor de índices de productos [batch_size]

        Returns:
            predictions: tensor de predicciones de rating [batch_size]
        """
        # Obtener embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        item_emb = self.item_embedding(item_ids)  # [batch_size, embedding_dim]

        # Concatenar embeddings
        concat = torch.cat([user_emb, item_emb], dim=1)  # [batch_size, 2*embedding_dim]

        # Pasar por MLP
        output = self.mlp(concat)  # [batch_size, 1]

        # Aplicar sigmoid y escalar a rango [1, 5]
        output = torch.sigmoid(output.squeeze()) * 4.0 + 1.0  # [batch_size]

        return output


# =============================================================================
# DEEP HYBRID RECOMMENDER (Sistema Completo)
# =============================================================================

class DeepHybridRecommender(nn.Module):
    """
    Sistema Híbrido Profundo que combina:
    - User-Based Collaborative Filtering
    - Item-Based Collaborative Filtering
    - SVD (Matrix Factorization)
    - Neural Collaborative Filtering (Deep Learning)

    Usa un mecanismo de atención para aprender pesos dinámicos
    """

    def __init__(self, num_users, num_items, ncf_config):
        """
        Args:
            num_users: número de usuarios
            num_items: número de productos
            ncf_config: diccionario con config de NCF (embedding_dim, hidden_dims, etc)
        """
        super(DeepHybridRecommender, self).__init__()

        # Componente Deep Learning (NCF)
        self.ncf = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=ncf_config.get('embedding_dim', 64),
            hidden_dims=ncf_config.get('hidden_dims', [128, 64, 32]),
            dropout_rate=ncf_config.get('dropout_rate', 0.2)
        )

        # Attention mechanism para combinar las 4 predicciones
        # Entrada: 4 features (User-CF, Item-CF, SVD, NCF)
        # Salida: 4 pesos que suman 1 (softmax)
        self.attention_layer = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(8, 4)
        )

        # Modelos tradicionales (se asignan después de entrenar fuera de PyTorch)
        self.user_cf_model = None
        self.item_cf_model = None
        self.svd_model = None

        # Mapeos de IDs
        self.user_to_idx = None
        self.item_to_idx = None
        self.idx_to_user = None
        self.idx_to_item = None

        print(f"  ✅ Deep Hybrid Recommender construido")

    def set_traditional_models(self, user_cf, item_cf, svd, user_map, item_map):
        """
        Asigna modelos tradicionales ya entrenados

        Args:
            user_cf: modelo User-Based CF (de surprise)
            item_cf: modelo Item-Based CF (de surprise)
            svd: modelo SVD (de surprise)
            user_map: dict {user_id_original: idx}
            item_map: dict {item_id_original: idx}
        """
        self.user_cf_model = user_cf
        self.item_cf_model = item_cf
        self.svd_model = svd
        self.user_to_idx = user_map
        self.item_to_idx = item_map
        self.idx_to_user = {v: k for k, v in user_map.items()}
        self.idx_to_item = {v: k for k, v in item_map.items()}

        print(f"  ✅ Modelos tradicionales asignados al híbrido")

    def get_traditional_predictions(self, user_indices, item_indices):
        """
        Obtiene predicciones de los modelos tradicionales

        Args:
            user_indices: lista de índices de usuarios
            item_indices: lista de índices de productos

        Returns:
            user_cf_preds: predicciones de User-CF
            item_cf_preds: predicciones de Item-CF
            svd_preds: predicciones de SVD
        """
        user_cf_preds = []
        item_cf_preds = []
        svd_preds = []

        for u_idx, i_idx in zip(user_indices, item_indices):
            # Convertir índices a IDs originales
            user_id = self.idx_to_user[u_idx]
            item_id = self.idx_to_item[i_idx]

            # Predicciones de cada modelo
            try:
                user_cf_pred = self.user_cf_model.predict(user_id, item_id).est
            except:
                user_cf_pred = 3.0  # Default fallback

            try:
                item_cf_pred = self.item_cf_model.predict(user_id, item_id).est
            except:
                item_cf_pred = 3.0

            try:
                svd_pred = self.svd_model.predict(user_id, item_id).est
            except:
                svd_pred = 3.0

            user_cf_preds.append(user_cf_pred)
            item_cf_preds.append(item_cf_pred)
            svd_preds.append(svd_pred)

        return (
            torch.tensor(user_cf_preds, dtype=torch.float32),
            torch.tensor(item_cf_preds, dtype=torch.float32),
            torch.tensor(svd_preds, dtype=torch.float32)
        )

    def forward(self, user_ids, item_ids):
        """
        Forward pass del sistema híbrido

        Args:
            user_ids: tensor de índices de usuarios [batch_size]
            item_ids: tensor de índices de productos [batch_size]

        Returns:
            final_predictions: tensor de predicciones [batch_size]
            attention_weights: pesos de atención [batch_size, 4]
        """
        device = user_ids.device

        # 1. Predicción Deep Learning (NCF)
        ncf_preds = self.ncf(user_ids, item_ids)  # [batch_size]

        # 2. Predicciones tradicionales
        user_indices = user_ids.cpu().numpy()
        item_indices = item_ids.cpu().numpy()

        user_cf_preds, item_cf_preds, svd_preds = self.get_traditional_predictions(
            user_indices, item_indices
        )

        # Mover a device correcto
        user_cf_preds = user_cf_preds.to(device)
        item_cf_preds = item_cf_preds.to(device)
        svd_preds = svd_preds.to(device)

        # 3. Stack todas las predicciones
        all_preds = torch.stack([
            user_cf_preds,
            item_cf_preds,
            svd_preds,
            ncf_preds
        ], dim=1)  # [batch_size, 4]

        # 4. Calcular pesos de atención
        attention_logits = self.attention_layer(all_preds)  # [batch_size, 4]
        attention_weights = F.softmax(attention_logits, dim=1)  # [batch_size, 4]

        # 5. Predicción final ponderada
        final_preds = (attention_weights * all_preds).sum(dim=1)  # [batch_size]

        # Clip a rango válido [1, 5]
        final_preds = torch.clamp(final_preds, 1.0, 5.0)

        return final_preds, attention_weights


# =============================================================================
# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN
# =============================================================================

def train_ncf_component(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """
    Entrena solo el componente NCF del modelo híbrido

    Args:
        model: DeepHybridRecommender
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        epochs: número de épocas
        lr: learning rate
        device: 'cpu' o 'cuda'

    Returns:
        history: diccionario con pérdidas y métricas por época
    """
    print(f"\n  🚀 Entrenando componente NCF...")
    print(f"    Épocas: {epochs}, LR: {lr}, Device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.ncf.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': []
    }

    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        # ===================== Training =====================
        model.train()
        train_losses = []

        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            # Forward pass (solo NCF)
            predictions = model.ncf(user_ids, item_ids)

            # Loss
            loss = criterion(predictions, ratings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.ncf.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        train_rmse = np.sqrt(avg_train_loss)

        # ===================== Validation =====================
        model.eval()
        val_losses = []

        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)

                predictions = model.ncf(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)
        val_rmse = np.sqrt(avg_val_loss)

        # Guardar métricas
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)

        # Print progreso
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Época {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val RMSE: {val_rmse:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    ⚠️  Early stopping en época {epoch+1}")
                break

    print(f"  ✅ Entrenamiento NCF completado")
    print(f"    Mejor Val RMSE: {np.sqrt(best_val_loss):.4f}")

    return history


def train_attention_mechanism(model, train_loader, val_loader, epochs=10, lr=0.0001, device='cpu'):
    """
    Entrena el mecanismo de atención del híbrido (fine-tuning)

    Args:
        model: DeepHybridRecommender (con NCF ya entrenado)
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        epochs: número de épocas
        lr: learning rate (más bajo para fine-tuning)
        device: 'cpu' o 'cuda'

    Returns:
        history: diccionario con métricas
    """
    print(f"\n  🎯 Entrenando mecanismo de atención (fine-tuning)...")
    print(f"    Épocas: {epochs}, LR: {lr}")

    model = model.to(device)

    # Solo optimizar attention layer y última capa de NCF
    optimizer = torch.optim.Adam([
        {'params': model.attention_layer.parameters(), 'lr': lr},
        {'params': model.ncf.mlp[-1].parameters(), 'lr': lr * 0.1}  # Última capa con LR menor
    ])

    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'attention_weights': []
    }

    for epoch in range(epochs):
        # ===================== Training =====================
        model.train()
        train_losses = []

        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            # Forward pass (híbrido completo)
            predictions, attention_weights = model(user_ids, item_ids)

            # Loss
            loss = criterion(predictions, ratings)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        train_rmse = np.sqrt(avg_train_loss)

        # ===================== Validation =====================
        model.eval()
        val_losses = []
        all_attention_weights = []

        with torch.no_grad():
            for user_ids, item_ids, ratings in val_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)

                predictions, attention_weights = model(user_ids, item_ids)
                loss = criterion(predictions, ratings)
                val_losses.append(loss.item())
                all_attention_weights.append(attention_weights.cpu().numpy())

        avg_val_loss = np.mean(val_losses)
        val_rmse = np.sqrt(avg_val_loss)

        # Pesos de atención promedio
        avg_attention = np.concatenate(all_attention_weights).mean(axis=0)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rmse'].append(train_rmse)
        history['val_rmse'].append(val_rmse)
        history['attention_weights'].append(avg_attention)

        print(f"    Época {epoch+1:2d}/{epochs} | "
              f"Val RMSE: {val_rmse:.4f} | "
              f"Pesos: User-CF={avg_attention[0]:.2f}, Item-CF={avg_attention[1]:.2f}, "
              f"SVD={avg_attention[2]:.2f}, NCF={avg_attention[3]:.2f}")

    print(f"  ✅ Fine-tuning completado")

    return history


def evaluate_deep_hybrid(model, test_loader, device='cpu'):
    """
    Evalúa el modelo híbrido en conjunto de prueba

    Args:
        model: DeepHybridRecommender
        test_loader: DataLoader de prueba
        device: 'cpu' o 'cuda'

    Returns:
        metrics: diccionario con RMSE, MAE, etc.
    """
    print(f"\n  📊 Evaluando Deep Hybrid en conjunto de prueba...")

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_attention_weights = []

    with torch.no_grad():
        for user_ids, item_ids, ratings in test_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)

            predictions, attention_weights = model(user_ids, item_ids)

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
            all_attention_weights.append(attention_weights.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_attention_weights = np.concatenate(all_attention_weights)

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    mae = mean_absolute_error(all_targets, all_predictions)

    # Pesos de atención promedio
    avg_attention = all_attention_weights.mean(axis=0)

    metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'predictions': all_predictions,
        'targets': all_targets,
        'attention_weights': avg_attention
    }

    print(f"    RMSE: {rmse:.4f}")
    print(f"    MAE: {mae:.4f}")
    print(f"    Pesos aprendidos: User-CF={avg_attention[0]:.3f}, "
          f"Item-CF={avg_attention[1]:.3f}, SVD={avg_attention[2]:.3f}, NCF={avg_attention[3]:.3f}")

    return metrics


# =============================================================================
# PIPELINE PRINCIPAL
# =============================================================================

def run_deep_hybrid_system(df_clean, user_cf_model, item_cf_model, svd_model):
    """
    Pipeline completo del sistema Deep Hybrid

    Args:
        df_clean: DataFrame con datos limpios (con user_idx, product_idx, rating)
        user_cf_model: modelo User-Based CF entrenado (surprise)
        item_cf_model: modelo Item-Based CF entrenado (surprise)
        svd_model: modelo SVD entrenado (surprise)

    Returns:
        results: diccionario con modelo, métricas e historia
    """
    print("\n" + "=" * 70)
    print("  🧠 DEEP HYBRID RECOMMENDER SYSTEM")
    print("=" * 70)

    # ========== 1. Preparar datos ==========
    print("\n  📋 Preparando datos para Deep Learning...")

    # Mapeos de IDs
    unique_users = df_clean['user_idx'].unique()
    unique_items = df_clean['product_idx'].unique()

    user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
    item_to_idx = {iid: idx for idx, iid in enumerate(unique_items)}

    # Convertir a índices consecutivos para embeddings
    df_mapped = df_clean.copy()
    df_mapped['user_idx_mapped'] = df_mapped['user_idx'].map(user_to_idx)
    df_mapped['item_idx_mapped'] = df_mapped['product_idx'].map(item_to_idx)

    num_users = len(unique_users)
    num_items = len(unique_items)

    print(f"    Usuarios: {num_users:,}")
    print(f"    Productos: {num_items:,}")
    print(f"    Interacciones: {len(df_mapped):,}")

    # ========== 2. Split datos ==========
    print("\n  🔀 Dividiendo datos (70% train, 15% val, 15% test)...")

    # Train (70%), Temp (30%)
    train_df, temp_df = sklearn_train_test_split(
        df_mapped, test_size=0.3, random_state=config.RANDOM_STATE
    )

    # Val (50% de temp = 15%), Test (50% de temp = 15%)
    val_df, test_df = sklearn_train_test_split(
        temp_df, test_size=0.5, random_state=config.RANDOM_STATE
    )

    print(f"    Train: {len(train_df):,} ({len(train_df)/len(df_mapped)*100:.1f}%)")
    print(f"    Val: {len(val_df):,} ({len(val_df)/len(df_mapped)*100:.1f}%)")
    print(f"    Test: {len(test_df):,} ({len(test_df)/len(df_mapped)*100:.1f}%)")

    # ========== 3. Crear DataLoaders ==========
    print("\n  📦 Creando DataLoaders...")

    train_dataset = RatingsDataset(
        train_df['user_idx_mapped'].values,
        train_df['item_idx_mapped'].values,
        train_df['rating'].values
    )

    val_dataset = RatingsDataset(
        val_df['user_idx_mapped'].values,
        val_df['item_idx_mapped'].values,
        val_df['rating'].values
    )

    test_dataset = RatingsDataset(
        test_df['user_idx_mapped'].values,
        test_df['item_idx_mapped'].values,
        test_df['rating'].values
    )

    batch_size = config.DL_BATCH_SIZE

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ========== 4. Crear modelo ==========
    print("\n  🏗️  Construyendo Deep Hybrid Recommender...")

    ncf_config = {
        'embedding_dim': config.DL_EMBEDDING_SIZE,
        'hidden_dims': config.DL_HIDDEN_LAYERS,
        'dropout_rate': config.DL_DROPOUT_RATE
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Device: {device}")

    model = DeepHybridRecommender(
        num_users=num_users,
        num_items=num_items,
        ncf_config=ncf_config
    )

    # Asignar modelos tradicionales
    model.set_traditional_models(
        user_cf_model,
        item_cf_model,
        svd_model,
        user_to_idx,
        item_to_idx
    )

    # ========== 5. Entrenar NCF component ==========
    ncf_history = train_ncf_component(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.DL_EPOCHS,
        lr=config.DL_LEARNING_RATE,
        device=device
    )

    # ========== 6. Fine-tune attention mechanism ==========
    attention_history = train_attention_mechanism(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,  # Menos épocas para fine-tuning
        lr=config.DL_LEARNING_RATE * 0.1,  # Learning rate más bajo
        device=device
    )

    # ========== 7. Evaluar en test set ==========
    test_metrics = evaluate_deep_hybrid(model, test_loader, device=device)

    # ========== 8. Guardar modelo ==========
    print("\n  💾 Guardando modelo...")

    model_save_path = config.MODELS_DIR / 'deep_hybrid_model.pth'

    torch.save({
        'model_state_dict': model.state_dict(),
        'ncf_config': ncf_config,
        'num_users': num_users,
        'num_items': num_items,
        'user_to_idx': user_to_idx,
        'item_to_idx': item_to_idx,
        'test_metrics': test_metrics,
        'ncf_history': ncf_history,
        'attention_history': attention_history
    }, model_save_path)

    print(f"    Guardado en: {model_save_path.name}")

    # ========== 9. Retornar resultados ==========
    results = {
        'model': model,
        'evaluation': test_metrics,
        'ncf_history': ncf_history,
        'attention_history': attention_history,
        'device': device,
        'num_users': num_users,
        'num_items': num_items
    }

    print("\n" + "=" * 70)
    print("  ✅ DEEP HYBRID SYSTEM COMPLETADO")
    print("=" * 70)

    return results


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def visualize_deep_hybrid_results(results, comparison_df=None):
    """
    Genera visualizaciones de los resultados del Deep Hybrid

    Args:
        results: diccionario con resultados de run_deep_hybrid_system()
        comparison_df: DataFrame con comparación de todos los métodos (opcional)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style('whitegrid')

    # ========== 1. Curvas de entrenamiento NCF ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ncf_history = results['ncf_history']
    epochs = range(1, len(ncf_history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, ncf_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, ncf_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Neural Collaborative Filtering - Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # RMSE
    axes[1].plot(epochs, ncf_history['train_rmse'], 'b-', label='Train RMSE', linewidth=2)
    axes[1].plot(epochs, ncf_history['val_rmse'], 'r-', label='Val RMSE', linewidth=2)
    axes[1].set_xlabel('Época', fontsize=12)
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('Neural Collaborative Filtering - RMSE', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = config.REPORTS_DIR / 'deep_hybrid_ncf_training.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Gráfica guardada: {save_path.name}")
    plt.close()

    # ========== 2. Pesos de atención aprendidos ==========
    fig, ax = plt.subplots(figsize=(10, 6))

    attention_weights = results['evaluation']['attention_weights']
    components = ['User-CF', 'Item-CF', 'SVD', 'NCF']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    bars = ax.bar(components, attention_weights, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Añadir valores en barras
    for bar, weight in zip(bars, attention_weights):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{weight:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Peso de Atención', fontsize=12)
    ax.set_title('Pesos Aprendidos por el Mecanismo de Atención\n(Deep Hybrid Recommender)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(attention_weights) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    save_path = config.REPORTS_DIR / 'deep_hybrid_attention_weights.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Gráfica guardada: {save_path.name}")
    plt.close()

    # ========== 3. Predicciones vs Real ==========
    fig, ax = plt.subplots(figsize=(10, 10))

    predictions = results['evaluation']['predictions']
    targets = results['evaluation']['targets']

    # Sample para visualización (max 5000 puntos)
    if len(predictions) > 5000:
        indices = np.random.choice(len(predictions), 5000, replace=False)
        predictions_sample = predictions[indices]
        targets_sample = targets[indices]
    else:
        predictions_sample = predictions
        targets_sample = targets

    ax.scatter(targets_sample, predictions_sample, alpha=0.4, s=20, color='#2E86AB')
    ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Perfect Prediction')

    rmse = results['evaluation']['RMSE']
    mae = results['evaluation']['MAE']

    ax.set_xlabel('Rating Real', fontsize=12)
    ax.set_ylabel('Rating Predicho', fontsize=12)
    ax.set_title(f'Deep Hybrid: Predicciones vs Real\nRMSE: {rmse:.4f}, MAE: {mae:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)

    plt.tight_layout()
    save_path = config.REPORTS_DIR / 'deep_hybrid_predictions.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  📊 Gráfica guardada: {save_path.name}")
    plt.close()

    # ========== 4. Comparación con otros métodos (si está disponible) ==========
    if comparison_df is not None:
        fig, ax = plt.subplots(figsize=(12, 7))

        x = np.arange(len(comparison_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, comparison_df['RMSE'], width,
                       label='RMSE', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, comparison_df['MAE'], width,
                       label='MAE', color='#A23B72', alpha=0.8)

        ax.set_xlabel('Algoritmo', fontsize=12)
        ax.set_ylabel('Error', fontsize=12)
        ax.set_title('Comparación de Todos los Métodos\n(Menor es Mejor)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Algoritmo'], rotation=15, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        # Añadir valores
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = config.REPORTS_DIR / 'deep_hybrid_full_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  📊 Gráfica guardada: {save_path.name}")
        plt.close()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  Este módulo debe ejecutarse desde main.py")
    print("  El Deep Hybrid requiere modelos tradicionales pre-entrenados")
    print("=" * 70)
    print("\n  Para ejecutar el pipeline completo:")
    print("    python main.py")
    print("\n" + "=" * 70)
