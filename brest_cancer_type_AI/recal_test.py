import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Model ve Scaler'ı Yükle ---
model = joblib.load("model/boob_monster.pkl")
scaler = joblib.load("model/bm_scaler.pkl")

# --- 2. Test Verisini ve Gerçek Etiketleri Yükle ---
# Eğitimde kullandığın veriyi bölerek oluşturduysan .pkl olarak kaydetmiş olmalısın
# Eğer hala bellekte duruyorsa, buraya test verilerini oku

# Örnek olarak X_test ve y_test bir .pkl dosyasında diyelim:
X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

# --- 3. Ölçeklendir ---
X_test_scaled = scaler.transform(X_test)

# --- 4. Tahmin Yap ---
y_pred = model.predict(X_test_scaled)

# --- 5. Performansı Yazdır ---
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
