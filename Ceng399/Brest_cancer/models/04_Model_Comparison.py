import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

# 1. Veri Yükleme
train_data = pd.read_csv('data/processed/train_scaled.csv')
test_data = pd.read_csv('data/processed/test_scaled.csv')

X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# 2. Modelleri En İyi Parametrelerle Tanımlama ve Eğitme
print("Modeller eğitiliyor, lütfen bekleyin...")

# Logistic Regression
lr_model = LogisticRegression(C=0.1, penalty='l2', solver='liblinear', class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=2, 
                                  min_samples_split=10, class_weight='balanced_subsample', random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=1.0,
                              scale_pos_weight=1.29, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# SVM
svm_model = SVC(C=100, kernel='linear', class_weight='balanced', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# 3. Görsel Karşılaştırma (ROC Curve)
models = [
    ('Logistic Regression', lr_model),
    ('Random Forest', rf_model),
    ('XGBoost', xgb_model),
    ('SVM', svm_model)
]

plt.figure(figsize=(10, 7))

for name, model in models:
    # Olasılık skorlarını alıyoruz (Malignant sınıfı için)
    y_score = model.predict_proba(X_test)[:, 1]
    
    # ROC ve AUC hesaplama
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Çizgi ekleme
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

# Şans çizgisi
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Alarm Oranı)')
plt.ylabel('True Positive Rate (Yakalanan Kanser Oranı)')
plt.title('Brest Cancer - Model Karşılaştırması (ROC Curve)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

# Kaydetme ve Gösterme
os.makedirs('../reports/figures', exist_ok=True)
plt.savefig('../reports/figures/05_Comparison_ROC_Curve.png', dpi=300)
print("Karşılaştırma grafiği '../reports/figures/' klasörüne kaydedildi.")
plt.show()