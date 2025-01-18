from LDA import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def load_data(cols, load_all=False, head=False):
    iris = sns.load_dataset("iris")

    if not load_all:
        if head:
            iris = iris.head(100)
        else:
            iris = iris.tail(100)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(iris["species"])

    X = iris.drop(["species"], axis=1)

    if len(cols) > 0:
        X = X[cols]

    return X.values, y

def exp1():       
    cols = ["petal_length", "petal_width"]
    X, y = load_data(cols, load_all=False, head=True)
    print(X.shape)

    lda = LDA()
    eig_vecs = lda.fit(X, y)
    W = eig_vecs[:, :1]

    colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(figsize=(10, 8))
    for point, pred in zip(X, y):
        ax.scatter(point[0], point[1], color=colors[pred], alpha=0.3)
        proj = (np.dot(point, W) * W) / np.dot(W.T, W)

        ax.scatter(proj[0], proj[1], color=colors[pred], alpha=0.3)

    plt.show()

def exp2():
    cols = ["petal_length", "petal_width"]
    X, y = load_data(cols, load_all=True, head=True)
    print(X.shape)

    lda = LDA()
    eig_vecs = lda.fit(X, y)
    W = eig_vecs[:, :1]

    colors = ['red', 'green', 'blue']
    fig, ax = plt.subplots(figsize=(10, 8))
    for point, pred in zip(X, y):
        ax.scatter(point[0], point[1], color=colors[pred], alpha=0.3)
        proj = (np.dot(point, W) * W) / np.dot(W.T, W)

        ax.scatter(proj[0], proj[1], color=colors[pred], alpha=0.3)

    plt.show()

def exp3():
    X, y = load_data([], load_all=True, head=True)
    print(X.shape)

    lda = LDA()
    eig_vecs = lda.fit(X, y)
    W = eig_vecs[:, :2]

    transformed = X.dot(W)
    fig, ax = plt.subplots(figsize=(10, 8))
    # ax_y = np.ones(transformed.shape)
    plt.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap=plt.cm.Set1)
    plt.show()

def exp4():
    X, y = load_data([], load_all=True, head=True)
    print(X.shape)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X, y)
    transformed = clf.transform(X)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.scatter(transformed[:, 0], transformed[:, 1], c=y, cmap=plt.cm.Set1)
    plt.show()

exp1()
exp2()
exp3()
exp4()
print('end')