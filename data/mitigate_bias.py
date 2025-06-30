# scripts/mitigate_bias.py
from sklearn.utils import resample

# Separate by protected attribute
df_min = df[df['sex'] == 0]
df_maj = df[df['sex'] == 1]

df_min_upsampled = resample(df_min, replace=True, n_samples=len(df_maj), random_state=42)
df_balanced = pd.concat([df_maj, df_min_upsampled])

# Re-train
X = df_balanced.drop('income', axis=1)
y = df_balanced['income']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
joblib.dump(model, 'models/balanced_model.joblib')
