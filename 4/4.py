import numpy as np

x1 = np.array([1, -1, 1])
x2 = np.array([-1, 1, -1])
k = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
n = 3

x1t = np.expand_dims(x1, axis=1)
x2t = np.expand_dims(x2, axis=1)

w =  (1 / 3) * (np.dot(x1t, x1t.T) + np.dot(x2t, x2t.T) - 2 * k)

def fun(w, x):
    y = np.array([])
    for i in range(len(w)):
        s = np.array([])
        for j in range(len(w)):
            s = np.append(s, w[i][j])
        y = np.append(y, np.dot(s, x))
    y = np.where(y >= 0, 1, -1)
    return y


y0 = np.array([-1, -1, 1])
y1 = fun(w, y0)
print(y1)
while not np.array_equal(y0, y1):
    y0 = y1
    y1 = fun(w, y0)
    print(y1)

