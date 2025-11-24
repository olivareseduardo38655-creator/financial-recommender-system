import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import pickle
import os

class CollaborativeRecommender:
    """
    Motor de Recomendación SVD (Matrix Factorization).
    """

    def __init__(self, n_components=20, data_path='data/processed', model_path='models_store'):
        self.n_components = n_components
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.matrix_pivot = None
        self.user_map = None
        self.product_map = None
        self.reverse_product_map = None

    def load_and_pivot(self):
        print('Cargando datos procesados...')
        interactions = pd.read_csv(f'{self.data_path}/interactions_processed.csv')
        products = pd.read_csv(f'{self.data_path}/products_processed.csv')
        
        self.product_names = dict(zip(products['product_id'], products['product_name']))
        
        with open(f'{self.model_path}/encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
            self.user_map = {v: k for k, v in encoders['user_id_map'].items()}
            self.reverse_product_map = {v: k for k, v in encoders['product_id_map'].items()}

        print('Agregando interacciones repetidas (Sumar Interés)...')
        # CORRECCIÓN CRÍTICA: Groupby para eliminar duplicados de índice
        interactions_grouped = interactions.groupby(['user_idx', 'product_idx'])['implicit_rating'].sum().reset_index()

        print('Creando matriz de utilidad (User-Item)...')
        self.matrix_pivot = interactions_grouped.pivot(
            index='user_idx', 
            columns='product_idx', 
            values='implicit_rating'
        ).fillna(0)
        
        self.sparse_matrix = csr_matrix(self.matrix_pivot.values)
        return self.sparse_matrix

    def train(self):
        print(f'Entrenando SVD con {self.n_components} factores latentes...')
        self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.model.fit(self.sparse_matrix)
        
        explained_variance = self.model.explained_variance_ratio_.sum()
        print(f'--> Entrenamiento completado. Varianza explicada acumulada: {explained_variance:.2%}')
        
        with open(f'{self.model_path}/svd_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def recommend(self, user_idx, n_recommendations=5):
        if user_idx >= self.matrix_pivot.shape[0]:
            return [], []

        user_vector = self.matrix_pivot.iloc[user_idx].values.reshape(1, -1)
        user_latent = self.model.transform(user_vector)
        predicted_scores = np.dot(user_latent, self.model.components_)
        
        consumed_indices = np.where(user_vector[0] > 0)[0]
        final_scores = predicted_scores.flatten()
        final_scores[consumed_indices] = -np.inf 
        
        top_indices = final_scores.argsort()[::-1][:n_recommendations]
        return top_indices, final_scores[top_indices]

    def generate_report_example(self, test_user_idx=10):
        try:
            print('\n' + '='*50)
            print(f' REPORTE DE RECOMENDACIÓN :: USUARIO INDEX {test_user_idx}')
            print('='*50)
            
            real_user_id = self.user_map[test_user_idx]
            print(f'Cliente ID: {real_user_id}')
            
            recs, scores = self.recommend(test_user_idx)
            
            print('\n[MODELO SVD] Recomendamos ofrecerle:')
            if len(recs) == 0:
                print("   Usuario sin historial suficiente para SVD.")
            else:
                for i, idx in enumerate(recs):
                    p_id = self.reverse_product_map.get(idx, "Unknown")
                    score = scores[i]
                    print(f'   {i+1}. {p_id} (Score de Afinidad: {score:.4f})')
            print('='*50 + '\n')
        except Exception as e:
            print(f"Error generando reporte: {e}")

if __name__ == '__main__':
    recsys = CollaborativeRecommender(n_components=15)
    recsys.load_and_pivot()
    recsys.train()
    recsys.generate_report_example(test_user_idx=5)
