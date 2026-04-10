import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# 1. Veri Yükleme
train_data = pd.read_csv('data/processed/train_scaled.csv')
test_data = pd.read_csv('data/processed/test_scaled.csv')

X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# 2. Final Model (GridSearch'ten gelen en iyi parametreler sabitlendi)
final_lr_model = LogisticRegression(
    C=0.1, 
    penalty='l2', 
    solver='liblinear', 
    class_weight='balanced', 
    random_state=42,
    max_iter=1000
)
final_lr_model.fit(X_train, y_train)

# 3. Tahmin ve Metrikler
y_pred = final_lr_model.predict(X_test)
y_pred_proba = final_lr_model.predict_proba(X_test)[:, 1]

print("\n--- FINAL LOGISTIC REGRESSION SONUÇLARI ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred))

# 4. Görselleştirme
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
            yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'])

plt.title('Final Optimized Confusion Matrix - Logistic Regression')
plt.tight_layout()

# Kaydetme
os.makedirs('../reports/figures', exist_ok=True)
plt.savefig('../reports/figures/01_LR_Final_Confusion_Matrix.png', dpi=300)
plt.show()