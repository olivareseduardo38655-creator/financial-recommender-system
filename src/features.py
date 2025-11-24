import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
import os

class FeatureEngineeringPipeline:
    """
    Pipeline de Ingeniería de Características para Sistema de Recomendación Híbrido.
    Transforma datos crudos en vectores numéricos normalizados.
    """

    def __init__(self, input_path='data/raw', output_path='data/processed', model_store='models_store'):
        self.input_path = input_path
        self.output_path = output_path
        self.model_store = model_store
        
        # Artefactos de datos
        self.users = None
        self.products = None
        self.interactions = None
        
        # Encoders (para guardar y usar en inferencia)
        self.scalers = {}
        self.encoders = {}

    def load_data(self):
        """Carga datos crudos."""
        print('Cargando datos raw...')
        self.users = pd.read_csv(f'{self.input_path}/users.csv')
        self.products = pd.read_csv(f'{self.input_path}/products.csv')
        self.interactions = pd.read_csv(f'{self.input_path}/interactions.csv')

    def process_users(self):
        """
        Genera User Embeddings basados en contenido.
        - Normaliza ingresos y score crediticio.
        - Codifica variables categóricas (Ciudad, Género, Empleo).
        """
        print('Procesando características de usuarios...')
        df = self.users.copy()
        
        # 1. Feature: Income Decile (Segmentación)
        df['income_decile'] = pd.qcut(df['monthly_income'], 10, labels=False)
        
        # 2. Normalización Numérica (MinMax para Deep Learning)
        scaler = MinMaxScaler()
        numeric_cols = ['age', 'monthly_income', 'credit_score']
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['user_scaler'] = scaler
        
        # 3. Encoding Categórico
        cat_cols = ['gender', 'city', 'employment_status']
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.encoders[f'user_{col}'] = le
            
        self.users_processed = df
        return df

    def process_products(self):
        """
        Genera Item Embeddings basados en contenido.
        - One-Hot Encoding de categorías.
        - Normalización de tasas y límites.
        """
        print('Procesando características de productos...')
        df = self.products.copy()
        
        # 1. Normalización
        scaler = MinMaxScaler()
        # Rellenar NaNs en productos que no aplican (ej. Seguros sin tasa)
        df[['interest_rate', 'credit_limit']] = df[['interest_rate', 'credit_limit']].fillna(0)
        df[['interest_rate', 'credit_limit']] = scaler.fit_transform(df[['interest_rate', 'credit_limit']])
        self.scalers['product_scaler'] = scaler
        
        # 2. Encoding de Categoría (Label Encoding para Embeddings, One-Hot es opcional)
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        self.encoders['product_category'] = le
        
        self.products_processed = df
        return df

    def create_interaction_matrix(self):
        """
        Prepara dataset para entrenamiento (User ID, Item ID, Rating, Timestamp).
        """
        print('Generando dataset de entrenamiento...')
        df = self.interactions.copy()
        
        # Mapear IDs a índices enteros (necesario para PyTorch/TensorFlow Embedding Layers)
        user_ids = self.users_processed['user_id'].unique()
        product_ids = self.products_processed['product_id'].unique()
        
        user2idx = {o:i for i,o in enumerate(user_ids)}
        product2idx = {o:i for i,o in enumerate(product_ids)}
        
        df['user_idx'] = df['user_id'].map(user2idx)
        df['product_idx'] = df['product_id'].map(product2idx)
        
        # Guardamos los mapas para decodificar recomendaciones después
        self.encoders['user_id_map'] = user2idx
        self.encoders['product_id_map'] = product2idx
        
        self.interactions_processed = df[['user_idx', 'product_idx', 'implicit_rating', 'interaction_type']]
        return self.interactions_processed

    def save_artifacts(self):
        """Guarda datos procesados y objetos encoders."""
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.model_store, exist_ok=True)
        
        # Datos CSV listos para ML
        self.users_processed.to_csv(f'{self.output_path}/users_processed.csv', index=False)
        self.products_processed.to_csv(f'{self.output_path}/products_processed.csv', index=False)
        self.interactions_processed.to_csv(f'{self.output_path}/interactions_processed.csv', index=False)
        
        # Guardar Encoders (Pickle)
        with open(f'{self.model_store}/encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        with open(f'{self.model_store}/scalers.pkl', 'wb') as f:
            pickle.dump(self.scalers, f)
            
        print(f'Pipeline finalizado. Artefactos guardados en {self.output_path} y {self.model_store}')

if __name__ == '__main__':
    pipeline = FeatureEngineeringPipeline()
    pipeline.load_data()
    pipeline.process_users()
    pipeline.process_products()
    pipeline.create_interaction_matrix()
    pipeline.save_artifacts()
