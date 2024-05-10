import numpy as np
import matplotlib.pyplot as plt
def func(x, a, b, c, d):
    if x <= a:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)
    else:
        return 0
def term(term, x):
    if term == 1:
        return func(x, -1000, -200, -100, -50)
    elif term == 2:
        return func(x, -100, -50, -50, -10)
    elif term == 3:
        return func(x, -50, -10, 0, 0)
    elif term == 4:
        return func(x, 0, 0, 10, 50)
    elif term == 5:
        return func(x, 10, 50, 50, 100)
    elif term == 6:
        return func(x, 50, 100, 200, 1000)
def zfp(fi, w, y, rules):
    x_ = 0
    for i in rules:
        alfa = term(i[0], fi)
        beta = term(i[1], w)
        gamma = min(alfa, beta)
        delta = term(i[2], y)
        x = min(gamma, delta)
        if x_ < x:
            x_ = x
    return x_
if __name__ == '__main__':
    t = 0
    end = 50
    fi_grad = 50
    w_grad_s = 4
    fi = fi_grad / 180 * np.pi
    w = w_grad_s / 180 * np.pi
    dy = .1
    dt = .01
    fi_min = - np.pi
    fi_max = np.pi
    w_min = - np.pi / 9
    w_max = np.pi / 9
    y_min = - np.pi / 36
    y_max = np.pi / 36
    fi_t, w_t, y_t, t_n = [], [], [], []
    rules = np.array([
    [1, 1, 6], [1, 2, 6], [1, 3, 6], [1, 4, 6], [1, 5, 6], [1, 6, 6],
    [2, 1, 6], [2, 2, 6], [2, 3, 6], [2, 4, 5], [2, 5, 5], [2, 6, 5],
    [3, 1, 6], [3, 2, 6], [3, 3, 6], [3, 4, 5], [3, 5, 5], [3, 6, 5],
    [4, 1, 2], [4, 2, 2], [4, 3, 2], [4, 4, 1], [4, 5, 1], [4, 6, 1],
    [5, 1, 2], [5, 2, 2], [5, 3, 2], [5, 4, 1], [5, 5, 1], [5, 6, 1],
    [6, 1, 1], [6, 2, 1], [6, 3, 1], [6, 4, 1], [6, 5, 1], [6, 6, 1],
    ])
    while t < end - 0.5 * dt:
        fi_ = 200 * (fi - fi_min) / (fi_max - fi_min) - 100
        w_ = 200 * (w - w_min) / (w_max - w_min) - 100
        y_ = -100
        s1 = 0
        s2 = 0
        while y_ < 100 - 0.5 * dy:
            x_ = zfp(fi_, w_, y_, rules)
            s1 = s1 + y_ * x_ * dy
            s2 = s2 + x_ * dy
            y_ += dy

        yx = s1 / s2 if s2 != 0 else 0
        y = (yx + 100) * (y_max - y_min) / 200 + y_min
        fi = fi + w * dt
        w = w + y * dt
        fi_t.append(fi)
        w_t.append(w)
        y_t.append(y)
        t_n.append(t)
        t += dt
    plt.plot(t_n, fi_t)
    plt.plot(t_n, w_t)
    plt.plot(t_n, y_t)
    plt.legend(['кут', 'кутова швидкість', 'кутове прискорення'])
    plt.grid(True)
    plt.xlabel('t')
    plt.savefig