import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

data=pd.read_csv("Misty.csv");

encoder=LabelEncoder()

data['outlook'] = encoder.fit_transform(data["outlook"])
data['temp'] = encoder.fit_transform(data["temp"])
data['humidity'] = encoder.fit_transform(data["humidity"])
data['wind'] = encoder.fit_transform(data["wind"])
data['play'] = encoder.fit_transform(data["play"])


X = data.drop(columns="play")
y = data["play"].values.reshape(-1, 1)

X = X / X.max(axis=0) #data normalization

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input_size = X_train.shape[1]
hidden_size = 5
output_size = 1

np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.rand(hidden_size, output_size)
b2 = np.zeros((1, output_size))
print("Weights ")
print(W1)
print("Biases ")
print(b1)

#Train the network
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X_train, W1) + b1  # dot product 
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Backpropagation
    error = y_train - a2
    d_a2 = error * sigmoid_derivative(a2)

    error_hidden = d_a2.dot(W2.T)
    d_a1 = error_hidden * sigmoid_derivative(a1)

    # Update weights and biases
    W2 += a1.T.dot(d_a2) * learning_rate
    b2 += np.sum(d_a2, axis=0, keepdims=True) * learning_rate
    W1 += X_train.T.dot(d_a1) * learning_rate
    b1 += np.sum(d_a1, axis=0, keepdims=True) * learning_rate

    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {epoch} Loss: {loss:.4f}")


print("Updated Weights ")
print(W1)
print("Updated Biases ")
print(b1)
#Predict on test set
z1_test = np.dot(X_test, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = sigmoid(z2_test)
y_pred = (a2_test > 0.5).astype(int)
cm = confusion_matrix(y_test,y_pred)
cmd=ConfusionMatrixDisplay(cm)
cmd.plot()
plt.show()

print("\nAccuracy on test data:", accuracy_score(y_test, y_pred))
