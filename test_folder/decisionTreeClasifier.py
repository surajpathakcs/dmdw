import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

data=pd.read_csv("Misty.csv")
encoder = preprocessing.LabelEncoder()
data['outlook'] = encoder.fit_transform(data["outlook"])
data['temp'] = encoder.fit_transform(data["temp"])
data['humidity'] = encoder.fit_transform(data["humidity"])
data['wind'] = encoder.fit_transform(data["wind"])
data['play'] = encoder.fit_transform(data["play"]) 
print("Data after encoding is: ")
print(data)

x = data.drop(columns="play")
y = data['play'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

# model = DecisionTreeClassifier(criterion="entropy")
model = SVC(kernel="linear")
model.fit(x_train,y_train)
fig = plt.figure(figsize=(30,20))
# tree.plot_tree(model)
fig.savefig("decistion_tree.png")
result=model.predict(x_test)
score = accuracy_score(y_test,result)
print("Accuracy score is "+str(score))
cm=confusion_matrix(y_test, result)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

