import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Veri Yükleme
train_data = pd.read_csv('data/processed/train_scaled.csv')
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Arama Havuzu
param_grid = [
    {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100], 
     'solver': ['liblinear'], 'class_weight': ['balanced']},
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100], 
     'solver': ['lbfgs'], 'class_weight': ['balanced']}
]

print("Lojistik Regresyon için GridSearch başlatılıyor...")
grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42), 
    param_grid, cv=5, scoring='recall'
)
grid_search.fit(X_train, y_train)

print(f"\n[SONUÇ] En İyi Parametreler: {grid_search.best_params_}")