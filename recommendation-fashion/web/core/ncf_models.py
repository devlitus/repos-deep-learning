"""
Modelos NCF (Neural Collaborative Filtering) con PyTorch
Incluye: NCFDataset, NeuralCollaborativeFiltering, EarlyStopping
"""

# Verificar si PyTorch está disponible
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    PYTORCH_AVAILABLE = True
except ImportError:
    pass


if PYTORCH_AVAILABLE:
    class NCFDataset(Dataset):
        """Dataset personalizado para PyTorch"""
        def __init__(self, user_indices, product_indices, ratings):
            self.users = torch.LongTensor(user_indices)
            self.products = torch.LongTensor(product_indices)
            self.ratings = torch.FloatTensor(ratings)

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
            return self.users[idx], self.products[idx], self.ratings[idx]

    class NeuralCollaborativeFiltering(nn.Module):
        """Red Neuronal para Sistema de Recomendación con mejoras"""
        def __init__(self, n_users, n_products, embedding_dim=64,
                     hidden_layers=None, dropout=0.2, use_batch_norm=True):
            super(NeuralCollaborativeFiltering, self).__init__()
            if hidden_layers is None:
                hidden_layers = [128, 64, 32]

            self.user_embedding = nn.Embedding(n_users, embedding_dim)
            self.product_embedding = nn.Embedding(n_products, embedding_dim)

            input_dim = embedding_dim * 2
            layers = []
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(input_dim, hidden_dim))
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, 1))
            self.mlp = nn.Sequential(*layers)

            self._init_weights()

        def _init_weights(self):
            nn.init.normal_(self.user_embedding.weight, std=0.01)
            nn.init.normal_(self.product_embedding.weight, std=0.01)
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

        def forward(self, user_indices, product_indices):
            user_emb = self.user_embedding(user_indices)
            product_emb = self.product_embedding(product_indices)
            x = torch.cat([user_emb, product_emb], dim=-1)
            output = self.mlp(x)
            output = torch.sigmoid(output) * 4 + 1  # Rango [1, 5]
            return output.squeeze()

    class EarlyStopping:
        """Early Stopping para prevenir overfitting"""
        def __init__(self, patience=3, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_loss = None
            self.early_stop = False
            self.best_epoch = 0

        def __call__(self, val_loss, epoch):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_epoch = epoch
            elif val_loss > self.best_loss - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.counter = 0
