import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = pd.read_csv("weather.csv")
encoder = LabelEncoder()
for col in ["outlook", "temp", "humidity", "wind", "play"]:
    data[col] = encoder.fit_transform(data[col])

X = data.drop(columns="play").values
y = data["play"].values.reshape(-1, 1)

# Normalize
X = X / X.max(axis=0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Sigmoid
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return z * (1 - z)

# Network structure
input_size, hidden_size, output_size = X_train.shape[1], 6, 1
np.random.seed(0)
W1, b1 = np.random.rand(input_size, hidden_size), np.zeros((1, hidden_size))
W2, b2 = np.random.rand(hidden_size, output_size), np.zeros((1, output_size))

# Training loop
epochs, lr = 8000, 0.1
for epoch in range(epochs):
    # Forward
    z1, a1 = np.dot(X_train, W1) + b1, None
    a1 = sigmoid(z1)
    z2, a2 = np.dot(a1, W2) + b2, sigmoid(np.dot(a1, W2) + b2)

    # Backprop
    error = y_train - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = d2.dot(W2.T) * sigmoid_deriv(a1)

    # Update
    W2 += a1.T.dot(d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += X_train.T.dot(d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss: {np.mean(error**2):.4f}")

# Test
a1_test = sigmoid(np.dot(X_test, W1) + b1)
a2_test = sigmoid(np.dot(a1_test, W2) + b2)
y_pred = (a2_test > 0.5).astype(int)

print("Backpropagation Accuracy:", accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()
plt.show()
