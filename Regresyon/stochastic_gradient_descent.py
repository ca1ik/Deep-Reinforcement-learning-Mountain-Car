import numpy as np
###########################################
# Yapay veri uretimi
#points = [(np.array([2]),4), (np.array([4]),2)]
#d = 1

true_w = np.array([1, 2, 3, 4, 5])
d = len(true_w)
points = []
for i in range(1000000):
    x = np.random.randn(d)
    y = true_w.dot(x) + np.random.randn()
    # print(x,y)
    points.append((x,y))

def sF(w, i):
    x,y = points[i]
    return (w.dot(x)-y)**2/2

def sdF(w, i):
    x,y = points[i]
    return (w.dot(x)-y)*x

###########################################
# Algorithms
def stochasticGradientDescent(sF, sdF, d, n):
    w = np.zeros(d)
    constant = 2.0
    numOfUpdates = 0
    for t in range(100):
        for i in range(n):
            value = sF(w, i)
            gradient = sdF(w, i)
            numOfUpdates += 1
            alpha = constant/numOfUpdates
            w = w - alpha * gradient
        print('Iteration {}: w = {}, F(w) = {} '.format(t, w, value))

stochasticGradientDescent(sF, sdF, d, len(points))


