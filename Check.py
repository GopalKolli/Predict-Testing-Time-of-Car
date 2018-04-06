import numpy as np
from sklearn.utils import shuffle

X = np.array([[1., 0.], [2., 1.], [0., 0.],[11., 10.], [12., 11.], [10., 10.]])
y = np.array([0, 1, 2,10, 11, 12])

print(X)
print(y)

X, y = shuffle(X, y, random_state=4)

print(X)
print(y)