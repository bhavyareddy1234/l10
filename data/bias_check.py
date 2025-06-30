# scripts/bias_check.py
import pandas as pd
import joblib
from sklearn.metrics import classification_report

df = pd.read_csv('data/adult.csv', header=None, na_values=' ?').dropna()
df.columns = [...]  # same as before
for col in df.select_dtypes(include='object').columns:
    df[col] = pd.factorize(df[col])[0]

X = df.drop('income', axis=1)
y = df['income']
model = joblib.load('models/model.joblib')
y_pred = model.predict(X)

report = classification_report(y, y_pred, output_dict=True)
with open('results/bias_report.txt', 'w') as f:
    f.write(str(report))
