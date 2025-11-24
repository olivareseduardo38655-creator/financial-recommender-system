import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# Configuración de Dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

# --- 1. Definición del Dataset Pytorch ---
class FinancialDataset(Dataset):
    def __init__(self, user_ids, product_ids, ratings):
        self.users = torch.tensor(user_ids, dtype=torch.long)
        self.products = torch.tensor(product_ids, dtype=torch.long)
        self.ratings = torch.tensor(ratings, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.products[idx], self.ratings[idx]

# --- 2. Arquitectura de la Red Neuronal (NCF) ---
class NCF(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, user, product):
        u_emb = self.user_embedding(user)
        p_emb = self.product_embedding(product)
        x = torch.cat([u_emb, p_emb], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x.squeeze()

# --- 3. Pipeline de Entrenamiento ---
def train_ncf():
    data_path = 'data/processed'
    model_path = 'models_store'
    
    print('Cargando datos procesados...')
    df = pd.read_csv(f'{data_path}/interactions_processed.csv')
    
    with open(f'{model_path}/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
        n_users = len(encoders['user_id_map'])
        n_products = len(encoders['product_id_map'])
        
    print(f'Dimensiones: {n_users} Usuarios | {n_products} Productos')

    dataset = FinancialDataset(
        df['user_idx'].values,
        df['product_idx'].values,
        df['implicit_rating'].values
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = NCF(n_users, n_products).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    print(f'Iniciando entrenamiento por {epochs} épocas...')
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for user, product, rating in dataloader:
            user, product, rating = user.to(device), product.to(device), rating.to(device)
            optimizer.zero_grad()
            prediction = model(user, product)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f'Época {epoch+1}/{epochs} - Loss: {total_loss/len(dataloader):.4f}')

    torch.save(model.state_dict(), f'{model_path}/ncf_model.pth')
    print('Modelo Neuronal guardado exitosamente.')
    
    print('\n--- REPORTE DE INFERENCIA NEURONAL (Test) ---')
    model.eval()
    test_user = torch.tensor([5]).to(device)
    print(f'Predicciones para Usuario Index 5:')
    with torch.no_grad():
        for i in range(5):
            test_product = torch.tensor([i]).to(device)
            score = model(test_user, test_product)
            print(f'   Producto Index {i}: Score {score.item():.4f}')

if __name__ == '__main__':
    train_ncf()
