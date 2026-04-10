import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 1. Veri Yükleme
train_data = pd.read_csv('data/processed/train_scaled.csv')
X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

# 2. Parametre Havuzu
# C: Regülarizasyon (Küçük C daha geniş sınır, Büyük C daha az hata toleransı)
# kernel: 'linear' (doğrusal), 'rbf' (dairesel/karmaşık)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 0.01], # Sadece rbf için kritik
    'class_weight': ['balanced']
}

print("SVM için GridSearch başlatılıyor... (Sınırlar çiziliyor)")
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42), # probability=True olasılık hesaplaması (ROC-AUC) için gerekli
    param_grid, 
    cv=5, 
    scoring='recall', 
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\n[SONUÇ] En İyi Parametreler: {grid_search.best_params_}")
print(f"En İyi Eğitim Recall Skoru: {grid_search.best_score_:.4f}")