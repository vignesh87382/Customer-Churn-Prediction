import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

np.random.seed(42)

data_size = 1000

tenure = np.random.randint(1, 72, data_size)
monthly_charges = np.random.randint(20, 120, data_size)
total_charges = tenure * monthly_charges + np.random.randint(-100, 100, data_size)

churn = (
    (tenure < 12).astype(int) +
    (monthly_charges > 80).astype(int)
)

churn = (churn > 1).astype(int)

data = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "churn": churn
})

X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

pickle.dump(model, open("churn_model.pkl", "wb"))

print("\nModel saved successfully.")
