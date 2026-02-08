"""
Script para entrenar SOLO el modelo de Deep Learning (NCF) sin modelos tradicionales.
No requiere scikit-surprise - solo PyTorch.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config

print("=" * 80)
print("🚀 ENTRENAMIENTO DE DEEP LEARNING - NEURAL COLLABORATIVE FILTERING")
print("=" * 80)

# Verificar GPU/CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n📱 Dispositivo: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memoria disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print(f"   ⚠️  Usando CPU (puede ser lento)")

# ================================
# 1. CARGAR DATASET
# ================================
print("\n" + "=" * 80)
print("📊 CARGANDO DATASET")
print("=" * 80)

data_path = config.DATASET_FILE
print(f"Archivo: {data_path.name}")

# Cargar JSON Lines (cada línea es un objeto JSON)
data_list = []
with open(str(data_path), 'r') as f:
    for line in f:
        data_list.append(json.loads(line.strip()))

df = pd.DataFrame(data_list)

# Renombrar columnas para consistencia
df = df.rename(columns={
    'reviewerID': 'user_id',
    'asin': 'product_id',
    'overall': 'rating'
})

print(f"✅ Dataset cargado: {len(df)} interacciones")
print(f"   - Usuarios: {df['user_id'].nunique()}")
print(f"   - Productos: {df['product_id'].nunique()}")
print(f"   - Ratings: {df['rating'].min():.1f} - {df['rating'].max():.1f}")

# ================================
# 2. PREPARAR DATOS
# ================================
print("\n" + "=" * 80)
print("🔧 PREPARANDO DATOS PARA DEEP LEARNING")
print("=" * 80)

# 1. Split PRIMERO (70% train, 15% val, 15% test)
from sklearn.model_selection import train_test_split

temp_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
train_df, val_df = train_test_split(temp_df, test_size=0.176, random_state=42)

print(f"📊 Split de datos:")
print(f"   - Train: {len(train_df)} interacciones ({len(train_df)/len(df)*100:.1f}%)")
print(f"   - Val:   {len(val_df)} interacciones ({len(val_df)/len(df)*100:.1f}%)")
print(f"   - Test:  {len(test_df)} interacciones ({len(test_df)/len(df)*100:.1f}%)")

# 2. Crear mappings SOLO desde train (sin data leakage)
train_user_ids = train_df['user_id'].unique()
train_product_ids = train_df['product_id'].unique()

user_to_idx = {uid: idx for idx, uid in enumerate(train_user_ids)}
product_to_idx = {pid: idx for idx, pid in enumerate(train_product_ids)}

n_users = len(train_user_ids)
n_products = len(train_product_ids)

print(f"\n✅ Mapeo completado (desde train):")
print(f"   - Usuarios: {n_users} (índices 0-{n_users-1})")
print(f"   - Productos: {n_products} (índices 0-{n_products-1})")

# 3. Aplicar mappings
train_df['user_idx'] = train_df['user_id'].map(user_to_idx)
train_df['product_idx'] = train_df['product_id'].map(product_to_idx)

# Val/Test: filtrar usuarios/productos no vistos en train
val_df['user_idx'] = val_df['user_id'].map(user_to_idx)
val_df['product_idx'] = val_df['product_id'].map(product_to_idx)
val_before = len(val_df)
val_df = val_df.dropna(subset=['user_idx', 'product_idx'])
val_df['user_idx'] = val_df['user_idx'].astype(int)
val_df['product_idx'] = val_df['product_idx'].astype(int)

test_df['user_idx'] = test_df['user_id'].map(user_to_idx)
test_df['product_idx'] = test_df['product_id'].map(product_to_idx)
test_before = len(test_df)
test_df = test_df.dropna(subset=['user_idx', 'product_idx'])
test_df['user_idx'] = test_df['user_idx'].astype(int)
test_df['product_idx'] = test_df['product_idx'].astype(int)

print(f"\n📋 Filtrado de val/test (usuarios/productos no vistos en train):")
print(f"   - Val: {val_before - len(val_df)} interacciones filtradas ({len(val_df)} restantes)")
print(f"   - Test: {test_before - len(test_df)} interacciones filtradas ({len(test_df)} restantes)")

# ================================
# 3. DEFINIR MODELO NCF
# ================================
print("\n" + "=" * 80)
print("🧠 DEFINIENDO ARQUITECTURA NCF")
print("=" * 80)

class NCFDataset(Dataset):
    """Dataset para PyTorch"""
    def __init__(self, user_indices, product_indices, ratings):
        self.users = torch.LongTensor(user_indices)
        self.products = torch.LongTensor(product_indices)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.products[idx], self.ratings[idx]


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering con embeddings de usuarios y productos
    Versión mejorada con Batch Normalization y dropout configurable
    """
    def __init__(self, n_users, n_products, embedding_dim=64, hidden_layers=None,
                 dropout=0.2, use_batch_norm=False):
        super(NeuralCollaborativeFiltering, self).__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.n_users = n_users
        self.n_products = n_products
        self.embedding_dim = embedding_dim

        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.product_embedding = nn.Embedding(n_products, embedding_dim)

        # MLP layers
        input_dim = embedding_dim * 2
        layers = []

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))

            # Batch Normalization (opcional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        # Inicialización
        self._init_weights()

    def _init_weights(self):
        """Inicialización de pesos"""
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.product_embedding.weight, std=0.01)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, user_indices, product_indices):
        """Forward pass"""
        user_emb = self.user_embedding(user_indices)
        product_emb = self.product_embedding(product_indices)

        # Concatenar embeddings
        x = torch.cat([user_emb, product_emb], dim=-1)

        # MLP
        output = self.mlp(x)

        # Escalar a rango [1, 5]
        output = torch.sigmoid(output) * 4 + 1

        return output.squeeze()


# Configuración con mejoras para prevenir overfitting
embedding_dim = config.DL_EMBEDDING_SIZE
hidden_layers = config.DL_HIDDEN_LAYERS
batch_size = 512  # Aumentado de 256 para más generalización
learning_rate = 0.0005  # Reducido de 0.001 para empezar más lento
n_epochs = config.DL_EPOCHS
dropout_rate = 0.5  # Aumentado de 0.3 a 0.5 (regularización agresiva)
weight_decay = 5e-4  # Aumentado de 1e-4 a 5e-4 (más L2)

print(f"Arquitectura:")
print(f"   - Embedding dimension: {embedding_dim}")
print(f"   - Hidden layers: {hidden_layers}")
print(f"   - Dropout rate: {dropout_rate}")
print(f"   - Usuarios: {n_users}")
print(f"   - Productos: {n_products}")
print(f"\nHiperparámetros:")
print(f"   - Batch size: {batch_size}")
print(f"   - Learning rate: {learning_rate}")
print(f"   - Weight decay: {weight_decay}")
print(f"   - Epochs: {n_epochs}")

# Crear modelo con Batch Normalization y dropout aumentado
model = NeuralCollaborativeFiltering(
    n_users=n_users,
    n_products=n_products,
    embedding_dim=embedding_dim,
    hidden_layers=hidden_layers,
    dropout=dropout_rate,
    use_batch_norm=True  # Activar Batch Normalization
).to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 Parámetros del modelo:")
print(f"   - Total: {total_params:,}")
print(f"   - Entrenables: {trainable_params:,}")

# ================================
# 4. PREPARAR TRAINING
# ================================
print("\n" + "=" * 80)
print("⚙️  PREPARANDO ENTRENAMIENTO")
print("=" * 80)

# Datasets
train_dataset = NCFDataset(
    train_df['user_idx'].values,
    train_df['product_idx'].values,
    train_df['rating'].values
)

val_dataset = NCFDataset(
    val_df['user_idx'].values,
    val_df['product_idx'].values,
    val_df['rating'].values
)

test_dataset = NCFDataset(
    test_df['user_idx'].values,
    test_df['product_idx'].values,
    test_df['rating'].values
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ DataLoaders creados:")
print(f"   - Train batches: {len(train_loader)}")
print(f"   - Val batches: {len(val_loader)}")
print(f"   - Test batches: {len(test_loader)}")

# Loss, optimizer y scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning Rate Scheduler - reduce LR cuando val loss se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

print(f"\n📈 Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
print(f"📉 Loss: MSE")
print(f"🔄 Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")

# ================================
# 5. ENTRENAR MODELO
# ================================
print("\n" + "=" * 80)
print("🏋️  ENTRENANDO MODELO NCF")
print("=" * 80)

# Early Stopping
class EarlyStopping:
    """Early stopping para prevenir overfitting"""
    def __init__(self, patience=5, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        self.best_model_state = None

    def __call__(self, val_loss, epoch, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = model.state_dict().copy()
            if self.verbose:
                print(f"   💾 Guardando mejor modelo (época {epoch+1})")
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"   ⚠️  Val RMSE no mejoró ({self.counter}/{self.patience})")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n🛑 Early stopping activado! Mejor modelo en época {self.best_epoch+1}")
        else:
            if self.verbose:
                print(f"   ✅ Val RMSE mejoró: {self.best_loss:.4f} → {val_loss:.4f}")
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_model_state = model.state_dict().copy()
            self.counter = 0

early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=True)  # Reducido de 5 a 3

history = {
    'train_rmse': [],
    'val_rmse': [],
    'learning_rates': []
}

start_time = datetime.now()

for epoch in range(n_epochs):
    # ========== TRAIN ==========
    model.train()
    train_mse = 0.0

    for users, products, ratings in train_loader:
        users = users.to(device)
        products = products.to(device)
        ratings = ratings.to(device)

        predictions = model(users, products)
        loss = criterion(predictions, ratings)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_mse += loss.item() * len(users)

    train_rmse = np.sqrt(train_mse / len(train_dataset))

    # ========== VALIDACIÓN ==========
    model.eval()
    val_mse = 0.0

    with torch.no_grad():
        for users, products, ratings in val_loader:
            users = users.to(device)
            products = products.to(device)
            ratings = ratings.to(device)

            predictions = model(users, products)
            loss = criterion(predictions, ratings)

            val_mse += loss.item() * len(users)

    val_rmse = np.sqrt(val_mse / len(val_dataset))

    # Guardar métricas
    current_lr = optimizer.param_groups[0]['lr']
    history['train_rmse'].append(train_rmse)
    history['val_rmse'].append(val_rmse)
    history['learning_rates'].append(current_lr)

    print(f"Epoch {epoch+1:02d}/{n_epochs} | "
          f"Train RMSE: {train_rmse:.4f} | "
          f"Val RMSE: {val_rmse:.4f} | "
          f"LR: {current_lr:.6f}")

    # Learning Rate Scheduler
    scheduler.step(val_rmse)

    # Early Stopping
    early_stopping(val_rmse, epoch, model)
    if early_stopping.early_stop:
        print(f"\n🛑 Entrenamiento detenido en época {epoch+1}")
        print(f"   Mejor modelo fue en época {early_stopping.best_epoch+1} con Val RMSE: {early_stopping.best_loss:.4f}")
        # Restaurar mejor modelo
        model.load_state_dict(early_stopping.best_model_state)
        break

# ========== EVALUACIÓN FINAL EN TEST ==========
model.eval()
test_mse = 0.0

with torch.no_grad():
    for users, products, ratings in test_loader:
        users = users.to(device)
        products = products.to(device)
        ratings = ratings.to(device)

        predictions = model(users, products)
        loss = criterion(predictions, ratings)

        test_mse += loss.item() * len(users)

test_rmse = np.sqrt(test_mse / len(test_dataset))

training_time = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 80)
print(f"⏱️  Tiempo total: {training_time:.1f} segundos ({training_time/60:.1f} minutos)")
print(f"\n📊 Resultados finales:")
print(f"   - Train RMSE (final): {history['train_rmse'][-1]:.4f}")
print(f"   - Val RMSE (final):   {history['val_rmse'][-1]:.4f}")
print(f"   - Mejor Val RMSE: {min(history['val_rmse']):.4f} (época {np.argmin(history['val_rmse'])+1})")
print(f"   - Test RMSE:  {test_rmse:.4f}")
print(f"   - Épocas entrenadas: {len(history['train_rmse'])}/{n_epochs}")
print(f"   - Gap Train-Val: {abs(history['train_rmse'][-1] - history['val_rmse'][-1]):.4f}")

# ================================
# 6. GUARDAR MODELO
# ================================
print("\n" + "=" * 80)
print("💾 GUARDANDO MODELO")
print("=" * 80)

# Crear directorio si no existe
config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Guardar modelo PyTorch
model_path = config.MODELS_DIR / 'ncf_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'n_users': n_users,
    'n_products': n_products,
    'embedding_dim': embedding_dim,
    'hidden_layers': hidden_layers,
    'user_to_idx': user_to_idx,
    'product_to_idx': product_to_idx,
    'history': history,
    'best_val_rmse': float(min(history['val_rmse'])),
    'test_rmse': float(test_rmse),
    'training_time': training_time
}, str(model_path))

print(f"✅ Modelo guardado: {model_path.name}")

# Guardar métricas en JSON
metrics_path = config.MODELS_DIR / 'ncf_metrics.json'
metrics = {
    'model_type': 'Neural Collaborative Filtering',
    'architecture': {
        'n_users': n_users,
        'n_products': n_products,
        'embedding_dim': embedding_dim,
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'batch_normalization': True,
        'total_params': total_params,
        'trainable_params': trainable_params
    },
    'hyperparameters': {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'epochs': n_epochs,
        'epochs_trained': len(history['train_rmse']),
        'early_stopping_patience': 5,
        'gradient_clipping': 1.0
    },
    'results': {
        'train_rmse_final': float(history['train_rmse'][-1]),
        'val_rmse_final': float(history['val_rmse'][-1]),
        'best_val_rmse': float(min(history['val_rmse'])),
        'best_epoch': int(np.argmin(history['val_rmse']) + 1),
        'test_rmse': float(test_rmse),
        'train_val_gap': float(abs(history['train_rmse'][-1] - history['val_rmse'][-1]))
    },
    'training': {
        'time_seconds': training_time,
        'device': str(device),
        'early_stopped': len(history['train_rmse']) < n_epochs
    },
    'dataset': {
        'total_interactions': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df)
    },
    'history': {
        'train_rmse': [float(x) for x in history['train_rmse']],
        'val_rmse': [float(x) for x in history['val_rmse']],
        'learning_rates': [float(x) for x in history['learning_rates']]
    }
}

with open(str(metrics_path), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✅ Métricas guardadas: {metrics_path.name}")

# ================================
# 7. VISUALIZACIÓN
# ================================
print("\n" + "=" * 80)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 80)

import matplotlib.pyplot as plt

# Crear directorio de reportes
reports_dir = config.REPORTS_DIR / 'figures'
reports_dir.mkdir(parents=True, exist_ok=True)

# Gráfico de RMSE por época
n_trained = len(history['train_rmse'])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: RMSE
ax1.plot(range(1, n_trained + 1), history['train_rmse'], label='Train RMSE', marker='o', markersize=4)
ax1.plot(range(1, n_trained + 1), history['val_rmse'], label='Val RMSE', marker='s', markersize=4)
ax1.axhline(y=test_rmse, color='r', linestyle='--', alpha=0.7, label=f'Test RMSE: {test_rmse:.4f}')
best_epoch = np.argmin(history['val_rmse']) + 1
ax1.axvline(x=best_epoch, color='g', linestyle=':', alpha=0.5, label=f'Mejor época: {best_epoch}')
ax1.set_xlabel('Época', fontsize=12)
ax1.set_ylabel('RMSE', fontsize=12)
ax1.set_title('Evolución del RMSE durante el Entrenamiento', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Learning Rate
ax2.plot(range(1, n_trained + 1), history['learning_rates'], marker='o', markersize=4, color='purple')
ax2.set_xlabel('Época', fontsize=12)
ax2.set_ylabel('Learning Rate', fontsize=12)
ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = reports_dir / 'ncf_training_history.png'
plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Gráfico guardado: {plot_path.name}")

print("\n" + "=" * 80)
print("🎉 PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print(f"\n📁 Archivos generados:")
print(f"   - Modelo: {model_path.name}")
print(f"   - Métricas: {metrics_path.name}")
print(f"   - Gráfico: {plot_path.name}")
print(f"\n💡 Para hacer predicciones, carga el modelo desde: {model_path}")
