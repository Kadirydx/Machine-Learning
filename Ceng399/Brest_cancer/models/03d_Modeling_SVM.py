import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Step 1: Veri Yükleme
# ---------------------------------------------------------
train_data = pd.read_csv('data/processed/train_scaled.csv')
test_data = pd.read_csv('data/processed/test_scaled.csv')

X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# ---------------------------------------------------------
# Step 2: Final Model (GridSearch Sonuçlarına Göre)
# ---------------------------------------------------------
# kernel='linear' ve C=100 seçildi.
# probability=True: ROC-AUC hesaplayabilmek için gereklidir.
final_svm_model = SVC(
    C=100, 
    kernel='linear', 
    class_weight='balanced', 
    probability=True, 
    random_state=42
)

final_svm_model.fit(X_train, y_train)
print("--- SVM Final Model Eğitimi Tamamlandı ---")

# ---------------------------------------------------------
# Step 3: Tahminler ve Performans Analizi
# ---------------------------------------------------------
y_pred = final_svm_model.predict(X_test)
y_pred_proba = final_svm_model.predict_proba(X_test)[:, 1]

print("\n--- SVM OPTİMİZE EDİLMİŞ SONUÇLAR ---")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}  <-- Rekor Kırıldı mı?")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n--- Detaylı Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

# ---------------------------------------------------------
# Step 4: Görselleştirme (Confusion Matrix)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
# SVM için mor/pembe tonları (PuRd) kullanalım
sns.heatmap(cm, annot=True, fmt='d', cmap='PuRd', 
            xticklabels=['Predicted Benign (0)', 'Predicted Malignant (1)'],
            yticklabels=['Actual Benign (0)', 'Actual Malignant (1)'])

plt.title('Final Confusion Matrix - SVM (Optimized)')
plt.ylabel('True Label (Gerçek Durum)')
plt.xlabel('Predicted Label (Modelin Tahmini)')
plt.tight_layout()

# Kaydetme
os.makedirs('../reports/figures', exist_ok=True)
plt.savefig('../reports/figures/04_SVM_Final_Confusion_Matrix.png', dpi=300)

plt.show()