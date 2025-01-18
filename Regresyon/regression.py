from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import math
from sklearn.model_selection import KFold


def regression_experiments(points):
    # Veriyi 5 kat capraz gecerleme ile parcalayarak modelleri egit
    points = np.asarray(points, dtype="object")
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(points):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = points[train_index,0], points[test_index,0]
        y_train, y_test = points[train_index,1], points[test_index,1]
        # 3 ayri ogrenme modelini egit ve test et

##################################################
# TEK DEGISKENLI REGRESYON PROBLEMI
##################################################
data = pd.read_csv('C:/Users/ceyda/codes/ai/odev4/data_regression.txt')
f1 = data.MAKINA
target = data.KAR_ZARAR

fig = plt.figure()
ax = fig.gca()
ax.plot(f1, target, 'rx')
ax.set_xlabel('Makina')
ax.set_ylabel('Kar')
# c=target, cmap='Greens' cmap='viridis', linewidth=0.5)
# show it
plt.show(block=True)

# veri kumesini duzenle ve normallestir
points = []
for i in range(f1.shape[0]):
    points.append((data.values[i,1:-1],data.values[i,0]))

regression_experiments(points)

