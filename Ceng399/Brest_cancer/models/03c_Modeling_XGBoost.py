import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
import warnings
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1: Veri Yükleme
# ---------------------------------------------------------
train_data = pd.read_csv('data/processed/train_scaled.csv')
test_data = pd.read_csv('data/processed/test_scaled.csv')

X_train = train_data.drop(columns=['target'])
y_train = train_data['target']

X_test = test_data.drop(columns=['target'])
y_test = test_data['target']

# ---------------------------------------------------------
# Step 2: Final Model (GridSearch Sonuçlarına Göre)
# ---------------------------------------------------------
# Bulduğumuz en iyi parametreleri buraya nakşediyoruz:
final_xgb_model = xgb.XGBClassifier(
    n_estimators=200, 
    learning_rate=0.05, 
    max_depth=3,
    subsample=1.0,
    scale_pos_weight=1.290246768507638, # GridSearch'ten gelen hassas oran
    random_state=42,
    eval_metric='logloss'
)

final_xgb_model.fit(X_train, y_train)
print("--- XGBoost Final Model Eğitimi Tamamlandı ---")

# ---------------------------------------------------------
# Step 3: Tahminler ve Performans Analizi
# ---------------------------------------------------------
y_pred = final_xgb_model.predict(X_test)
y_pred_proba = final_xgb_model.predict_proba(X_test)[:, 1]

print("\n--- XGBOOST OPTİMİZE EDİLMİŞ SONUÇLAR ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}  <-- (Kritik Metrik)")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

# ---------------------------------------------------------
# Step 4: Görselleştirme (Confusion Matrix)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
            xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
            yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'])

plt.title('Final Confusion Matrix - XGBoost (Optimized)')
plt.ylabel('True Label (Gerçek Durum)')
plt.xlabel('Predicted Label (Modelin Tahmini)')
plt.tight_layout()

# Kaydetme
os.makedirs('../reports/figures', exist_ok=True)
plt.savefig('../reports/figures/03_XGB_Final_Confusion_Matrix.png', dpi=300)

plt.show()