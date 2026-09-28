import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
import pickle

# Load dataset
data = pd.read_csv(r'D:\diabetes_ai_agent\data\diabetes.csv')

# Encode categorical columns
le_gender = LabelEncoder()
le_smoke = LabelEncoder()

data['gender'] = le_gender.fit_transform(data['gender'])
data['smoking_history'] = le_smoke.fit_transform(data['smoking_history'])

# Features & target
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("📊 Model Performance")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")

# Save model + encoders
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump((model, le_gender, le_smoke), f)

print("✅ Model trained & saved successfully")
