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

# Mapear IDs a índices
user_ids = df['user_id'].unique()
product_ids = df['product_id'].unique()

user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
product_to_idx = {product_id: idx for idx, product_id in enumerate(product_ids)}

df['user_idx'] = df['user_id'].map(user_to_idx)
df['product_idx'] = df['product_id'].map(product_to_idx)

n_users = len(user_ids)
n_products = len(product_ids)

print(f"✅ Mapeo completado:")
print(f"   - Usuarios: {n_users} (índices 0-{n_users-1})")
print(f"   - Productos: {n_products} (índices 0-{n_products-1})")

# Split train/test (80/20)
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\n📊 Split de datos:")
print(f"   - Train: {len(train_df)} interacciones ({len(train_df)/len(df)*100:.1f}%)")
print(f"   - Test:  {len(test_df)} interacciones ({len(test_df)/len(df)*100:.1f}%)")

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
    """
    def __init__(self, n_users, n_products, embedding_dim=64, hidden_layers=[128, 64, 32]):
        super(NeuralCollaborativeFiltering, self).__init__()

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
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
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


# Configuración
embedding_dim = config.DL_EMBEDDING_SIZE
hidden_layers = config.DL_HIDDEN_LAYERS
batch_size = config.DL_BATCH_SIZE
learning_rate = config.DL_LEARNING_RATE
n_epochs = config.DL_EPOCHS

print(f"Arquitectura:")
print(f"   - Embedding dimension: {embedding_dim}")
print(f"   - Hidden layers: {hidden_layers}")
print(f"   - Usuarios: {n_users}")
print(f"   - Productos: {n_products}")
print(f"\nHiperparámetros:")
print(f"   - Batch size: {batch_size}")
print(f"   - Learning rate: {learning_rate}")
print(f"   - Epochs: {n_epochs}")

# Crear modelo
model = NeuralCollaborativeFiltering(
    n_users=n_users,
    n_products=n_products,
    embedding_dim=embedding_dim,
    hidden_layers=hidden_layers
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

test_dataset = NCFDataset(
    test_df['user_idx'].values,
    test_df['product_idx'].values,
    test_df['rating'].values
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"✅ DataLoaders creados:")
print(f"   - Train batches: {len(train_loader)}")
print(f"   - Test batches: {len(test_loader)}")

# Loss y optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"\n📈 Optimizer: Adam (lr={learning_rate})")
print(f"📉 Loss: MSE")

# ================================
# 5. ENTRENAR MODELO
# ================================
print("\n" + "=" * 80)
print("🏋️  ENTRENANDO MODELO NCF")
print("=" * 80)

history = {
    'train_loss': [],
    'train_rmse': [],
    'test_loss': [],
    'test_rmse': []
}

start_time = datetime.now()

for epoch in range(n_epochs):
    # ========== TRAIN ==========
    model.train()
    train_loss = 0.0
    train_mse = 0.0

    for users, products, ratings in train_loader:
        users = users.to(device)
        products = products.to(device)
        ratings = ratings.to(device)

        # Forward
        predictions = model(users, products)
        loss = criterion(predictions, ratings)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(users)
        train_mse += loss.item() * len(users)

    train_loss /= len(train_dataset)
    train_rmse = np.sqrt(train_mse / len(train_dataset))

    # ========== TEST ==========
    model.eval()
    test_loss = 0.0
    test_mse = 0.0

    with torch.no_grad():
        for users, products, ratings in test_loader:
            users = users.to(device)
            products = products.to(device)
            ratings = ratings.to(device)

            predictions = model(users, products)
            loss = criterion(predictions, ratings)

            test_loss += loss.item() * len(users)
            test_mse += loss.item() * len(users)

    test_loss /= len(test_dataset)
    test_rmse = np.sqrt(test_mse / len(test_dataset))

    # Guardar historia
    history['train_loss'].append(train_loss)
    history['train_rmse'].append(train_rmse)
    history['test_loss'].append(test_loss)
    history['test_rmse'].append(test_rmse)

    # Progreso
    print(f"Epoch {epoch+1:02d}/{n_epochs} | "
          f"Train RMSE: {train_rmse:.4f} | "
          f"Test RMSE: {test_rmse:.4f}")

training_time = (datetime.now() - start_time).total_seconds()

print("\n" + "=" * 80)
print("✅ ENTRENAMIENTO COMPLETADO")
print("=" * 80)
print(f"⏱️  Tiempo total: {training_time:.1f} segundos ({training_time/60:.1f} minutos)")
print(f"\n📊 Resultados finales:")
print(f"   - Train RMSE: {history['train_rmse'][-1]:.4f}")
print(f"   - Test RMSE:  {history['test_rmse'][-1]:.4f}")
print(f"   - Mejor Test RMSE: {min(history['test_rmse']):.4f} (epoch {np.argmin(history['test_rmse'])+1})")

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
    'test_rmse': history['test_rmse'][-1],
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
        'total_params': total_params,
        'trainable_params': trainable_params
    },
    'hyperparameters': {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': n_epochs
    },
    'results': {
        'train_rmse': float(history['train_rmse'][-1]),
        'test_rmse': float(history['test_rmse'][-1]),
        'best_test_rmse': float(min(history['test_rmse'])),
        'best_epoch': int(np.argmin(history['test_rmse']) + 1)
    },
    'training': {
        'time_seconds': training_time,
        'device': str(device)
    },
    'dataset': {
        'total_interactions': len(df),
        'train_size': len(train_df),
        'test_size': len(test_df)
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
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_epochs + 1), history['train_rmse'], label='Train RMSE', marker='o', markersize=4)
plt.plot(range(1, n_epochs + 1), history['test_rmse'], label='Test RMSE', marker='s', markersize=4)
plt.xlabel('Época', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Evolución del RMSE durante el Entrenamiento - NCF', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
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
