import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

data = pd.read_csv(
    "Misty.csv"
)
print(data)

encoder = preprocessing.LabelEncoder()
data["outlook"] = encoder.fit_transform(data["outlook"])
data["temp"] = encoder.fit_transform(data["temp"])
data["humidity"] = encoder.fit_transform(data["humidity"])
data["wind"] = encoder.fit_transform(data["wind"])
data["play"] = encoder.fit_transform(data["play"])
print(data)

x = data.drop(columns="play")
y = data["play"].values
print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

model = GaussianNB()
model.fit(x_train, y_train)

model = SVC(kernel="linear")

predictedValue = model.predict(x_test)
print(y_test)
print(predictedValue)

accuracy = accuracy_score(y_test, predictedValue)
print(accuracy)

cm = confusion_matrix(y_test, predictedValue)
cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
cmd.plot()
plt.show()
