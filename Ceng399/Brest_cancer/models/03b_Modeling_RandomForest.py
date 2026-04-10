import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# 1. Veri Hazırlığı
train_data = pd.read_csv('data/processed/train_scaled.csv')
test_data = pd.read_csv('data/processed/test_scaled.csv')

X_train, y_train = train_data.drop(columns=['target']), train_data['target']
X_test, y_test = test_data.drop(columns=['target']), test_data['target']

# 2. Final Model (Bulduğumuz en iyi parametrelerle)
# GridSearch yapmadan direkt değerleri yazıyoruz
final_model = LogisticRegression(
    C=0.1, 
    penalty='l2', 
    solver='liblinear', 
    class_weight='balanced', 
    random_state=42
)

final_model.fit(X_train, y_train)

# 3. Tahmin ve Değerlendirme
y_pred = final_model.predict(X_test)
print("\n--- FINAL MODEL SONUÇLARI ---")
print(f"Recall Skoru: {recall_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# 4. Görselleştirme
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Final Model - Confusion Matrix')
plt.show()