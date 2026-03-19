import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

os.makedirs("models", exist_ok=True)

df = pd.read_csv('data/cleaned_churn_data.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']

joblib.dump(X.columns.tolist(), 'models/model_columns.pkl')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

log = LogisticRegression(max_iter=300)
log.fit(X_train, y_train)

joblib.dump(rf, 'models/random_forest.pkl')
joblib.dump(log, 'models/logistic_model.pkl')

print("✅ Models saved!")
