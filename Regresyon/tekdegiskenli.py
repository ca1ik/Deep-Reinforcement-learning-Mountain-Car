import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Veri tanımlama
data = """
MAKINA,KAR_ZARAR
6.1101,17.592
5.5277,9.1302
8.5186,13.662
7.0032,11.854
5.8598,6.8233
"""  # Buraya tüm veriyi ekleyebilirsiniz.

# DataFrame oluşturma
from io import StringIO
df = pd.read_csv(StringIO(data))

# Veriyi işleme
X = df['MAKINA'].values.reshape(-1, 1)  # Bağımsız değişken
y = df['KAR_ZARAR'].values  # Bağımlı değişken

# Normalizasyon
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Eğim azalması algoritması
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n + 1)  # Parametreler (θ)
    X_bias = np.c_[np.ones((m, 1)), X]  # Bias ekleme
    for epoch in range(epochs):
        predictions = X_bias.dot(theta)
        errors = predictions - y
        gradients = 1 / m * X_bias.T.dot(errors)
        theta -= learning_rate * gradients
    return theta

# Modeli eğit
theta = gradient_descent(X_norm, y)
print(f"Model Parametreleri (Theta): {theta}")

# Tahmin ve grafik
predictions = np.c_[np.ones((X_norm.shape[0], 1)), X_norm].dot(theta)
plt.scatter(X, y, label='Gerçek Değerler', color='blue')
plt.plot(X, predictions, label='Tahmin', color='red')
plt.xlabel('MAKINA')
plt.ylabel('KAR_ZARAR')
plt.legend()
plt.title('Tek Değişkenli Regresyon Analizi')
plt.show()