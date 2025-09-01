import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("weather.csv")
print("Original Data:\n", data)

# Encode categorical features
encoder = preprocessing.LabelEncoder()
for col in ["outlook", "temp", "humidity", "wind", "play"]:
    data[col] = encoder.fit_transform(data[col])

print("\nEncoded Data:\n", data)

# Features & target
X = data.drop(columns="play")
y = data["play"]

# Split into training & test sets (70% train, 30% test this time)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

print("\nTraining Features:\n", X_train)
print("\nTraining Labels:\n", y_train)

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict
y_pred = nb_model.predict(X_test)
print("\nActual Labels:", list(y_test))
print("Predicted Labels:", list(y_pred))

# Evaluate
acc = accuracy_score(y_test, y_pred)
print("\nNaive Bayes Accuracy:", round(acc, 3))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Confusion Matrix - Naive Bayes")
plt.show()
