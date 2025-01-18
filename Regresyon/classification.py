from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import math
from sklearn.model_selection import KFold
import scipy.io
from PCA import *
from LDA import *

##################################################
# IKI SINIFLI SINIFLANDIRMA PROBLEMI
##################################################
# veriyi yukle
data = scipy.io.loadmat('C:/Users/ceyda/codes/ai/odev4/data_classification.mat')
# Veri gorselleme amacli grafikleri  cizdir
features = np.asarray(data['features'])
labels = np.asarray(data['classes'])
data_zeros_mask = np.where(labels==0)
data_zeros = features[data_zeros_mask[0],:]
plt.plot(data_zeros[:,0], data_zeros[:,1], 'mo')
data_ones_mask = np.where(labels==1)
data_ones = features[data_ones_mask[0],:]
plt.plot(data_ones[:,0], data_ones[:,1], 'gx')
plt.xlabel('Ozellik 1')
plt.ylabel('Ozellik 2')
plt.show(block=True)

# veri kumesini duzenle ve normallestir
points = []
for i in range(features.shape[0]):
    points.append((features[i,:][:],labels[i,0]))

# veri kumesini duzenle ve normallestir
# lojistik regresyon ile egitimi gerceklestir
# basarimi degerlendir
 # Veriyi 5 kat capraz gecerleme ile parcalayarak modelleri egit

kf = KFold(n_splits=5, random_state=42, shuffle=True)
points = np.asarray(points, dtype="object")

for train_index, test_index in kf.split(points):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = points[train_index,0], points[test_index,0]
    y_train, y_test = points[train_index,1], points[test_index,1]
    # myLogisticRegression()
    
