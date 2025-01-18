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

def F(w):
    return sum((w.dot(x)-y)**2 for x,y in points)/(2*len(points))

def dF(w):
    return sum((w.dot(x)-y)*x for x,y in points)/(len(points))

###########################################
# Algorithms
def gradientDescent(F, dF, d):
    w = np.zeros(d) #initialization
    alpha = 0.5

    for t in range(100):
        value = F(w)
        gradient = dF(w)
        w = w - alpha * gradient
        print('Iteration {}: w = {}, F(w) = {} '.format(t, w, value))

gradientDescent(F, dF, d)


