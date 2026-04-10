import pandas as pd
import numpy as np
import xgboost as xgb
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# 1. Veri Yükleme
train_data = pd.read_csv('data/processed/train_scaled.csv')
test_data = pd.read_csv('data/processed/test_scaled.csv')

X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# 2. Bireysel Modelleri En İyi Parametrelerle Tanımlıyoruz
# (Daha önceki adımlarda bulduğumuz altın değerler)
clf1 = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced_subsample', random_state=42)
clf3 = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, scale_pos_weight=1.29, random_state=42, eval_metric='logloss')
clf4 = SVC(C=100, kernel='linear', class_weight='balanced', probability=True, random_state=42)

# 3. Voting Classifier (Süper Takım)
# voting='soft': Olasılıkların ortalamasını alarak daha hassas karar verir.
ensemble_model = VotingClassifier(
    estimators=[
        ('lr', clf1), 
        ('rf', clf2), 
        ('xgb', clf3), 
        ('svm', clf4)
    ],
    voting='soft' 
)

print("Ensemble model eğitiliyor... (4 dev birleşiyor)")
ensemble_model.fit(X_train, y_train)

# 4. Tahmin ve Değerlendirme
y_pred_ens = ensemble_model.predict(X_test)

print("\n" + "="*40)
print("--- ENSEMBLE (VOTING) MODEL SONUÇLARI ---")
print(f"Recall   : {recall_score(y_test, y_pred_ens):.4f} (Hedef: > 0.80)")
print(f"Accuracy : {accuracy_score(y_test, y_pred_ens):.4f}")
print("="*40)

print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred_ens, target_names=['Benign (0)', 'Malignant (1)']))

# 5. Görselleştirme
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_ens), annot=True, fmt='d', cmap='YlGnBu')
plt.title('Ensemble Model - Confusion Matrix')
plt.show()