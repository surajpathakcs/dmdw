import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("weather.csv")

# Encode categorical values
encoder = preprocessing.LabelEncoder()
for col in ["outlook", "temp", "humidity", "wind", "play"]:
    data[col] = encoder.fit_transform(data[col])

# Split data
X = data.drop(columns="play")
y = data["play"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Train Decision Tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
clf.fit(X_train, y_train)

# Plot tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()

# Evaluate
y_pred = clf.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
