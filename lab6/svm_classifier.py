import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("weather.csv")
encoder = preprocessing.LabelEncoder()
for col in ["outlook", "temp", "humidity", "wind", "play"]:
    data[col] = encoder.fit_transform(data[col])

# Train-test split
X = data.drop(columns="play")
y = data["play"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Train SVM
svm_model = SVC(kernel="linear", C=1.0)
svm_model.fit(X_train, y_train)

# Test
y_pred = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
