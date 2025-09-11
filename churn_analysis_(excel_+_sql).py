# -*- coding: utf-8 -*-
"""Churn Analysis (Excel + SQL).ipynb

import pandas as pd
from google.colab import files

uploaded = files.upload()
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print('The shape of this dataset is {}'.format(df.shape))
df.info()
df.head()

df.shape  #number of columns and rows
df.isnull().sum()

df.duplicated().sum()
df.describe

from scipy.stats import zscore
z_scores = zscore(df['MonthlyCharges'])
df = df[(z_scores < 20) | (z_scores > 75)]

df.to_csv('cleaned_churn_data.csv', index=False)
files.download('cleaned_churn_data.csv')

print(df.columns.tolist())

from sklearn.preprocessing import LabelEncoder

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']
print(X.dtypes.value_counts())

print(df.columns)

import pandas as pd
import sqlite3

df = pd.read_csv('cleaned_churn_data.csv')
conn = sqlite3.connect('churn_data.db')
df.to_sql('churn_data', conn, if_exists='replace', index=False)

result1 = pd.read_sql_query("""WITH avg_charges AS (SELECT InternetService, ROUND(AVG(MonthlyCharges), 2) AS avg_monthly FROM churn_data WHERE Churn = 'Yes' GROUP BY InternetService) SELECT InternetService, avg_monthly, RANK() OVER(ORDER BY avg_monthly DESC) AS avg_rank FROM avg_charges ORDER by avg_rank;""", conn)

result2 = pd.read_sql_query("""WITH churn_rate_data AS (SELECT InternetService, Contract, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churns FROM churn_data GROUP BY InternetService, Contract) SELECT InternetService, Contract, churns, ROW_NUMBER() OVER(PARTITION BY InternetService ORDER BY churns DESC) AS churn_rank FROM churn_rate_data ORDER BY churn_rank DESC;""", conn)

result3 = pd.read_sql_query("""WITH churn_rate_by_tenure AS (SELECT CASE WHEN tenure <=12 THEN '0-1 year' WHEN tenure <=24 THEN '1-2 years' ELSE '2+ years' END AS tenure_by_group, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churnss FROM churn_data GROUP BY tenure_by_group) SELECT tenure_by_group, churnss, RANK() OVER(ORDER BY churnss ASC) as churn_rank FROM churn_rate_by_tenure ORDER BY churn_rank;""", conn)

result4 = pd.read_sql_query("""SELECT gender, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate FROM churn_data GROUP BY gender;""", conn)

result5 = pd.read_sql_query("""WITH churn_by_contract AS (SELECT Contract, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate FROM churn_data GROUP BY Contract) SELECT Contract, churn_rate, ROW_NUMBER() OVER (ORDER BY churn_rate DESC) AS churn_rank FROM churn_by_contract;""", conn)

result6 = pd.read_sql_query("""SELECT PaymentMethod, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rates FROM churn_data GROUP BY PaymentMethod ORDER BY churn_rates ASC;""", conn)

result7 = pd.read_sql_query("""WITH churn_by_charge_range AS (SELECT CASE WHEN TotalCharges < 1000 THEN '$1K' WHEN TotalCharges < 5000 THEN '$1K - $5K' ELSE '>$5K' END AS charging, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate FROM churn_data GROUP BY charging) SELECT charging, churn_rate, RANK() OVER (ORDER BY churn_rate DESC) AS churn_rank FROM churn_by_charge_range;""", conn)

result8 = pd.read_sql_query("""SELECT InternetService, PaymentMethod, Contract, ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)*100.0 / COUNT(*), 2) AS churing FROM churn_data GROUP BY InternetService, PaymentMethod, Contract ORDER BY churing ASC LIMIT 5;""", conn)

result9 = pd.read_sql_query("""SELECT StreamingTV, Churn, ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY StreamingTV), 2) AS percent FROM churn_data GROUP BY StreamingTV, Churn ORDER BY percent ASC;""", conn)

result10 = pd.read_sql_query("""SELECT ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS churn_rate FROM churn_data;""", conn)

result11 = pd.read_sql_query("""SELECT COUNT(*) AS total_customers FROM churn_data;""", conn)

result12 = pd.read_sql_query("""SELECT COUNT(*) AS churned_customers FROM churn_data WHERE Churn = 'Yes';""", conn)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
model = RandomForestClassifier(n_estimators=90, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of the model:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC score of the model:", auc)


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of the model:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC score of the model:", auc)

from google.colab import files

with pd.ExcelWriter('cleaned_churn_data.xlsx') as writer:
    result1.to_excel(writer, sheet_name='Avg Monthly Charges', index=False)
    result2.to_excel(writer, sheet_name='Churn by Contract', index=False)
    result3.to_excel(writer, sheet_name='Churn by Tenure', index=False)
    result4.to_excel(writer, sheet_name='Churn by Gender', index=False)
    result5.to_excel(writer, sheet_name='Churn by Contract Type', index=False)
    result6.to_excel(writer, sheet_name='Churn by Payment Method', index=False)
    result7.to_excel(writer, sheet_name='Churn by Total Charges', index=False)
    result8.to_excel(writer, sheet_name='Low Churn Combos', index=False)
    result9.to_excel(writer, sheet_name='StreamingTV Churn %', index=False)
    result10.to_excel(writer, sheet_name='Overall Churn Rate', index=False)
    result11.to_excel(writer, sheet_name='Total Customers', index=False)
    result12.to_excel(writer, sheet_name='Churned Customers', index=False)

conn.close()
files.download('cleaned_churn_data.xlsx')

import joblib
joblib.dump(model, 'random_fores_model.pkl')
joblib.dump(model, 'decisiontree_model.pkl')
files.download('random_fores_model.pkl')
files.download('decisiontree_model.pkl')

print(df.columns.tolist())

from google.colab import files

uploaded = files.upload()

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv('cleaned_churn_data.csv')

features = ['Contract', 'tenure', 'PaymentMethod']
target = 'Churn'

X_cat = df[features].select_dtypes(include=['object'])
X_num = df[features].select_dtypes(exclude=['object'])

encoder = OneHotEncoder(drop='first', sparse_output=False)
X_cat_enc = pd.DataFrame(encoder.fit_transform(X_cat), columns=encoder.get_feature_names_out(X_cat.columns))

X = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
cv = cross_validate(model, X, y, cv=5, scoring='accuracy', return_train_score=True)

print("Cross-validation scores:", cv['test_score'])
print("Mean cross-validation score:", cv['test_score'].mean())

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score of the model:", accuracy)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("AUC score of the model:", roc_auc_score(y_test, y_proba))

from sklearn.model_selection import GridSearchCV

rf = LogisticRegression(random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'saga'],  # Solvers
    'penalty': ['l1','l2'],  # Regularization type
    'max_iter': [100, 200, 300]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best parameters found by grid search:", best_params)

results = X_test.copy()
results['Actual_Churn'] = y_test
results['Predicted_Churn'] = y_pred
results['Predicted_Probability'] = y_proba

results.to_csv('logistic_regression_results.csv', index=False)
files.download('logistic_regression_results.csv')

from scipy.stats import chi2_contingency, pointbiserialr

contingency_table = pd.crosstab(df['Contract'], df['Churn'])
chi2, p, dof, expected = chi2_contingency(contingency_table)

if p < 0.05:
    print("There is a significant association between Contract and Churn.")
else:
    print("There is no significant association between Contract and Churn.")

df['Churn_binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
correlation, p = pointbiserialr(df['tenure'], df['Churn_binary'])

if p < 0.05:
    print("There is a significant correlation between tenure and Churn.")
else:
    print("There is no significant correlation between tenure and Churn.")

sns.barplot(x = 'Contract', y = 'MonthlyCharges', data = df)
plt.title('Monthly Charges by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Monthly Charges')
plt.show()

plt.pie(df['Contract'].value_counts(), labels=df['Contract'].unique(), autopct='%1.1f%%')
plt.title('Distribution of Contract Types')
plt.show()

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print(df['Churn'].unique())

highrisk = df[(df['Contract'] == 'Month-to-month') & (df['tenure'] > 12) & (df['PaymentMethod'] == 'Electronic check')]
df['Churn'] = df['Churn'].astype(int)
churnrate = highrisk['Churn'].mean()
total = highrisk.shape[0]

monthlyrev = highrisk['MonthlyCharges'].mean()
preventedchurns = int(total * churnrate * 0.25)
savings = preventedchurns * monthlyrev

cost = preventedchurns * 10
roi = (savings - cost) / cost

print(total)
print(roi)
