import pandas as pd
import numpy as np
import torch
import pickle
import sys
import os

# Añadir src al path para poder importar las clases
sys.path.append('src')

# Importación condicional para evitar errores circulares si se ejecuta como script
try:
    from train_collaborative import CollaborativeRecommender
    from train_ncf import NCF
except ImportError:
    # Fallback si se ejecuta desde otra ruta
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from train_collaborative import CollaborativeRecommender
    from train_ncf import NCF

class HybridEngine:
    """
    Motor de Recomendación Híbrido.
    Orquesta: SVD + Deep Learning + Reglas de Negocio Financieras.
    """
    
    def __init__(self, model_path='models_store', data_path='data/processed'):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Artefactos
        self.svd_model = None
        self.ncf_model = None
        self.encoders = None
        self.products_df = None
        self.users_df = None
        self.product_map = {} # Mapa Index -> ID Real
        self.user_id_to_idx = {} # Mapa ID Real -> Index
        
        self.load_artifacts()

    def load_artifacts(self):
        print('Cargando modelos y metadatos...')
        
        # 1. Cargar Encoders (Diccionarios de mapeo)
        with open(f'{self.model_path}/encoders.pkl', 'rb') as f:
            self.encoders = pickle.load(f)
            
        # CORRECCIÓN: Invertir el diccionario del encoder directamente
        # product_id_map es { 'P001': 0, 'P002': 1 } -> Queremos { 0: 'P001', 1: 'P002' }
        self.product_map = {v: k for k, v in self.encoders['product_id_map'].items()}
        self.user_id_to_idx = self.encoders['user_id_map']
            
        # 2. Cargar SVD (Pickle)
        with open(f'{self.model_path}/svd_model.pkl', 'rb') as f:
            self.svd_model = pickle.load(f)
            
        # 3. Cargar NCF (State Dict)
        n_users = len(self.encoders['user_id_map'])
        n_products = len(self.encoders['product_id_map'])
        self.ncf_model = NCF(n_users, n_products).to(self.device)
        # map_location asegura que cargue en CPU si no hay GPU
        self.ncf_model.load_state_dict(torch.load(f'{self.model_path}/ncf_model.pth', map_location=self.device))
        self.ncf_model.eval()
        
        # 4. Cargar Datos Crudos (para reglas de negocio de riesgo)
        # Usamos try/except por si la ruta varía
        try:
            self.users_raw = pd.read_csv('data/raw/users.csv')
        except FileNotFoundError:
            self.users_raw = pd.read_csv('../data/raw/users.csv')

    def predict_hybrid(self, user_id_raw, top_k=5):
        """
        Genera recomendaciones híbridas filtradas por riesgo.
        """
        # 1. Validar Usuario
        if user_id_raw not in self.user_id_to_idx:
            print(f"Usuario {user_id_raw} no encontrado en el set de entrenamiento.")
            return []
        
        user_idx = self.user_id_to_idx[user_id_raw]
        
        # 2. Obtener Perfil de Riesgo (Business Logic)
        user_row = self.users_raw[self.users_raw['user_id'] == user_id_raw]
        if user_row.empty:
            print("Datos demográficos no encontrados.")
            credit_score = 650 # Default
            income = 15000
        else:
            credit_score = user_row.iloc[0]['credit_score']
            income = user_row.iloc[0]['monthly_income']
        
        print(f'\n--- ANÁLISIS DE PERFIL: {user_id_raw} ---')
        print(f'Credit Score: {credit_score} | Ingreso Mensual: ')
        
        # 3. Generar Candidatos (Todos los productos)
        all_product_indices = list(self.product_map.keys())
        
        # Tensor de usuario para NCF (repetido para todos los productos)
        user_tensor = torch.tensor([user_idx] * len(all_product_indices)).to(self.device)
        prod_tensor = torch.tensor(all_product_indices).to(self.device)
        
        # Inferencia Batch NCF
        with torch.no_grad():
            ncf_scores = self.ncf_model(user_tensor, prod_tensor).cpu().numpy()
            
        # Combinación y Filtrado
        final_recommendations = []
        
        print("\nEvaluando reglas de negocio...")
        rejected_count = 0
        
        for i, p_idx in enumerate(all_product_indices):
            p_id = self.product_map.get(p_idx, 'Unknown')
            ncf_s = float(ncf_scores[i])
            
            # --- REGLAS DE NEGOCIO (RISK ENGINE) ---
            
            # Regla 1: Tarjetas Platinum requieren Score > 720
            # (Simulamos que los productos con ID alto o cierto hash son Platinum)
            is_platinum = "Platinum" in p_id or (p_idx % 7 == 0) 
            
            if is_platinum and credit_score < 720:
                rejected_count += 1
                continue # Riesgo rechazado
            
            # Regla 2: Capacidad de Pago (Ingreso vs Anualidad simulada)
            # Simulamos que productos impares son caros
            is_expensive = (p_idx % 2 != 0)
            if is_expensive and income < 10000:
                rejected_count += 1
                continue

            final_recommendations.append((p_id, ncf_s))
            
        print(f"-> {rejected_count} productos bloqueados por normativa de riesgo.")
            
        # Ordenar por Score
        final_recommendations.sort(key=lambda x: x[1], reverse=True)
        return final_recommendations[:top_k]

if __name__ == '__main__':
    engine = HybridEngine()
    
    # Probamos con el usuario U000005
    test_user = 'U000005' 
    recs = engine.predict_hybrid(test_user)
    
    if recs:
        print(f'\nTop Recomendaciones Híbridas (Risk-Adjusted):')
        for item in recs:
            print(f'Producto: {item[0]} | Score: {item[1]:.4f}')
