import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # MinMaxScaler'ı buradan import ediyoruz
from sklearn.model_selection import train_test_split

# Çok değişkenli veri tanımlama (örnek veri veya dosyadan okuma)
data = """
X1,X2,X3,Y
6.1101,3.2,1.1,17.592
5.5277,2.8,1.0,9.1302
8.5186,5.1,1.4,13.662
7.0032,4.5,1.2,11.854
5.8598,3.0,1.1,6.8233
"""  # Buraya verinizin tamamını ekleyebilirsiniz.

# Veri çerçevesi oluşturma
from io import StringIO
df = pd.read_csv(StringIO(data))

# Giriş ve çıkış değişkenlerini ayırma
X = df[['X1', 'X2', 'X3']].values  # Bağımsız değişkenler
y = df['Y'].values  # Bağımlı değişken

# Veriyi normalize etme
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# Bias terimi ekleme
X_bias = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]

# Çok değişkenli regresyon için eğim azalması
def gradient_descent_multi(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros(n)  # Parametreler
    for epoch in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (1 / m) * X.T.dot(errors)
        theta -= learning_rate * gradients
    return theta

# Modeli eğit
theta = gradient_descent_multi(X_bias, y)
print(f"Model Parametreleri (Theta): {theta}")
