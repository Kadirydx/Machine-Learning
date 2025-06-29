import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib



df = pd.read_csv("data/veri.csv")

x = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


joblib.dump(model, "model/boob_monster.pkl")
joblib.dump(scaler, "model/bm_scaler.pkl")
joblib.dump(X_test, "data/X_test.pkl")
joblib.dump(y_test, "data/y_test.pkl")


# to re call the model and scaler later, uncomment the following lines:
#"import joblib
#
#model = joblib.load("model_logistic.pkl")
#yeni_tahmin = model.predict(X_test_scaled)
#"