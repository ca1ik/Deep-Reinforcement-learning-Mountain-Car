import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Veri hazırlık
data = df.values
X = data[:, 0].reshape(-1, 1)
y = data[:, 1]
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Eğim azalması algoritması
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n + 1)
    X_bias = np.c_[np.ones((m, 1)), X]  # Bias ekleme
    for epoch in range(epochs):
        predictions = X_bias.dot(theta)
        errors = predictions - y
        gradients = 1 / m * X_bias.T.dot(errors)
        theta -= learning_rate * gradients
    return theta

theta = gradient_descent(X_norm, y)
print(f"Theta: {theta}")