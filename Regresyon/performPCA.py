from PIL import Image
from numpy import *
from pylab import *
from PCA import pca
from keras.datasets import mnist
import matplotlib.pyplot as plt

#Download and load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
indices = y_train == 1
im = x_train[0] # choose one image to get size
m,n = im.shape[0:2] # get the size of the images
imnbr = len(x_train[indices]) # get the number of images
# form matrix to store all flattened images
immatrix = array([im.flatten()
    for im in x_train[indices]],'f')
# perform PCA
V,S,immean = pca(immatrix)

figure()
gray()
subplot(2,4,1)

for i in range(8):
    subplot(2,4,i+1)
    imshow(immatrix[i].reshape(m,n))
show()

# show some images (mean and 7 first modes)
figure()
gray()
subplot(2,4,1)
imshow(immean.reshape(m,n))


for i in range(7):
    subplot(2,4,i+2)
    print(np.shape(V[i]))
    imshow(V[i].reshape(m,n))    
show()

figure()
gray()
for i in range(8):
    subplot(2,4,i+1)
    transformedImage = np.matmul(immatrix[i]-immean,V)
    transformedImage = transformedImage.reshape(m,n)
    #transformedImage = transformedImage-np.min(transformedImage)
    #transformedImage = transformedImage/np.max(transformedImage)
    imshow(transformedImage)
show()

figure()
gray()
for i in range(8):
    subplot(2,4,i+1)
    params = 300
    transformedImage = np.matmul(immatrix[i]-immean,V)
    variance = np.matmul(V[:,0:params],transformedImage[0:params])
    reprojectedImage = immean+variance
    reprojectedImage = reprojectedImage.reshape(m,n)
    imshow(reprojectedImage)
show()

num = S.size//10
p = (100*(S[:num]**2).sum())/(S**2).sum()
print("Sum of eigen values for 10 {%} of features: ", p)
print("Seven features of first most significant seven eigenvectors: ",
    (dot(immatrix[0:3],V[0:7].T)))