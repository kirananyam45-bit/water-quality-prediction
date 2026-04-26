import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    'pH': [7, 6.5, 8, 5.5, 7.2, 6.8, 7.5],
    'Turbidity': [3, 5, 2, 8, 3, 4, 2],
    'Safe': [1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

X = df[['pH', 'Turbidity']]
y = df['Safe']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))