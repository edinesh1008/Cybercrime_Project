import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("cybercrime.csv")

data = data.dropna()

X = data[["City","Crime_Type","Amount"]]
y = data["Location"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "cybercrime_model.pkl")

print("Model Trained Successfully")
