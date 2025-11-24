import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Configuración de reproducibilidad científica
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
Faker.seed(SEED)

fake = Faker('es_MX')

class FinancialDataGenerator:
    def __init__(self, n_users=1000, n_products=50):
        self.n_users = n_users
        self.n_products = n_products
        self.users_df = None
        self.products_df = None
        self.interactions_df = None

    def generate_users(self):
        print(f'Generando {self.n_users} usuarios sintéticos...')
        users_data = []
        for i in range(self.n_users):
            age = random.randint(18, 75)
            monthly_income = round(np.random.lognormal(mean=9.5, sigma=0.6), 2)
            base_score = 650
            income_factor = (monthly_income / 10000) * 10
            age_factor = (age / 10) * 5
            noise = np.random.normal(0, 40)
            credit_score = int(np.clip(base_score + income_factor + age_factor + noise, 300, 850))

            user = {
                'user_id': f'U{str(i).zfill(6)}',
                'age': age,
                'gender': random.choice(['M', 'F']),
                'city': fake.city(),
                'monthly_income': monthly_income,
                'credit_score': credit_score,
                'employment_status': random.choices(['Salaried', 'Self-Employed', 'Unemployed', 'Retired'], weights=[0.6, 0.25, 0.05, 0.1])[0],
                'created_at': fake.date_between(start_date='-2y', end_date='today')
            }
            users_data.append(user)
        self.users_df = pd.DataFrame(users_data)
        return self.users_df

    def generate_products(self):
        print(f'Generando catálogo de {self.n_products} productos...')
        product_types = ['Credit Card', 'Personal Loan', 'Mortgage', 'Investment Fund', 'Insurance']
        products_data = []
        for i in range(self.n_products):
            p_type = random.choice(product_types)
            if p_type == 'Credit Card':
                interest_rate = round(np.random.uniform(15.0, 60.0), 2)
                limit = round(np.random.choice([5000, 10000, 20000, 50000, 100000]), 2)
            elif p_type == 'Personal Loan':
                interest_rate = round(np.random.uniform(10.0, 35.0), 2)
                limit = round(np.random.uniform(10000, 500000), 2)
            elif p_type == 'Mortgage':
                interest_rate = round(np.random.uniform(8.0, 12.0), 2)
                limit = round(np.random.uniform(500000, 5000000), 2)
            else:
                interest_rate = 0.0
                limit = 0.0
            
            product = {
                'product_id': f'P{str(i).zfill(4)}',
                'category': p_type,
                'product_name': f'{p_type} {fake.word().capitalize()} {random.choice(["Gold", "Platinum", "Basic", "Pro"])}',
                'interest_rate': interest_rate,
                'credit_limit': limit
            }
            products_data.append(product)
        self.products_df = pd.DataFrame(products_data)
        return self.products_df

    def generate_interactions(self, n_interactions=10000):
        print(f'Generando {n_interactions} interacciones...')
        if self.users_df is None or self.products_df is None: raise ValueError('Users and Products must be generated first.')
        interactions_data = []
        user_ids = self.users_df['user_id'].values
        product_ids = self.products_df['product_id'].values
        interaction_types = ['view', 'click', 'add_to_cart', 'conversion']
        interaction_weights = [0.5, 0.3, 0.15, 0.05]
        rating_map = {'view': 1, 'click': 2, 'add_to_cart': 3, 'conversion': 5}
        
        for _ in range(n_interactions):
            user = random.choice(user_ids)
            product = random.choice(product_ids)
            action = random.choices(interaction_types, weights=interaction_weights)[0]
            interactions_data.append({
                'user_id': user,
                'product_id': product,
                'interaction_type': action,
                'implicit_rating': rating_map[action],
                'timestamp': fake.date_time_between(start_date='-1y', end_date='now')
            })
        self.interactions_df = pd.DataFrame(interactions_data)
        return self.interactions_df

    def save_data(self, output_path='data/raw'):
        os.makedirs(output_path, exist_ok=True)
        self.users_df.to_csv(f'{output_path}/users.csv', index=False)
        self.products_df.to_csv(f'{output_path}/products.csv', index=False)
        self.interactions_df.to_csv(f'{output_path}/interactions.csv', index=False)
        print(f'Datos guardados exitosamente en {output_path}')

if __name__ == '__main__':
    generator = FinancialDataGenerator(n_users=2000, n_products=100)
    generator.generate_users()
    generator.generate_products()
    generator.generate_interactions(n_interactions=30000)
    generator.save_data()
