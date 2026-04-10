import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Veri Yükleme
train_data = pd.read_csv('data/processed/train_scaled.csv')
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# Sınıf dengesi oranı
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# Arama Havuzu
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'scale_pos_weight': [ratio] # Senin hesapladığın oran
}

print("XGBoost için GridSearch başlatılıyor... (Hataları düzelterek ilerliyor)")
grid_search = GridSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'), 
    param_grid, cv=5, scoring='recall', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\n[SONUÇ] En İyi Parametreler: {grid_search.best_params_}")