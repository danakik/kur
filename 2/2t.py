import numpy as np
import matplotlib.pyplot as plt

def membership_function(phi, omega, mu, rules):
    xi_dash = 0

    for rule in rules:
        alfa = term_function(rule[0], phi)
        beta = term_function(rule[1], omega)
        gamma = min(alfa, beta)
        delta = term_function(rule[2], mu)
        xi = min(gamma, delta)
        if xi_dash < xi:
            xi_dash = xi
    return xi_dash  

def function(x, a, b, c, d):  
    if x <= a:
        return 0
    elif x > a and x <= b:
        return (x - a) / (b - a)
    elif x > b and x <= c:
        return 1
    elif x > c and x <= d:
        return (d - x) / (d - c)
    else:
        return 0

def term_function(term, x):
    if term == 1:
        return function(x, -1000, -200, -100, -50)
    elif term == 2:
        return function(x, -100, -50, -50, -10)
    elif term == 3:
        return function(x, -50, -10, 0, 0)
    elif term == 4:
        return function(x, 0, 0, 10, 50)
    elif term == 5:
        return function(x, 10, 50, 50, 100)
    elif term == 6:
        return function(x, 50, 100, 200, 1000)


rules = np.array([
    [1,1,6], [1,2,6], [1,3,6], [1,4,6], [1,5,6], [1,6,6],
    [2,1,6], [2,2,6], [2,3,6], [2,4,5], [2,5,5], [2,6,5],
    [3,1,6], [3,2,6], [3,3,6], [3,4,5], [3,5,5], [3,6,5],
    [4,1,2], [4,2,2], [4,3,2], [4,4,1], [4,5,1], [4,6,1],
    [5,1,2], [5,2,2], [5,3,2], [5,4,1], [5,5,1], [5,6,1],
    [6,1,1], [6,2,1], [6,3,1], [6,4,1], [6,5,1], [6,6,1],
])

t = 0
t_max = 50
dt = 0.01
dmu = 0.1

phi = 50 * np.pi / 180 #0
omega = 4 / 180 * np.pi #6
phi_min, phi_max = -np.pi, np.pi
omega_min, omega_max = -np.pi / 9, np.pi / 9
mu_min, mu_max = -np.pi / 36, np.pi / 36

phi_t, omega_t, mu_t, t_n = [], [], [], []
z =0
b = 0
while t < t_max - 0.5 * dt:
    phi_ = 200 * (phi - phi_min) / (phi_max - phi_min) - 100
    omega_ = 200 * (omega - omega_min) / (omega_max - omega_min) - 100

    mu_ = -100
    s1, s2 = 0, 0
    print(z)
    while mu_ < 100 - 0.5 * dmu:
        xi_dash = membership_function(phi_, omega_, mu_, rules)
        s1 += mu_ * xi_dash * dmu
        s2 += xi_dash * dmu
        mu_ += dmu
        print(b,'b')
    b+=1

    mu_dash = s1 / s2 if s2 != 0 else 0
    mu = (mu_dash + 100) * (mu_max - mu_min) / 200 + mu_min

    phi += omega * dt
    omega += mu * dt

    phi_t.append(phi)
    omega_t.append(omega)
    mu_t.append(mu)
    t_n.append(t)


    t += dt
    z+=1

plt.plot(t_n, phi_t, 'r')
plt.plot(t_n, omega_t, 'b')
plt.plot(t_n, mu_t, 'k')

plt.legend(['кут', 'кутова швидкість', 'кутове прискорення'])
plt.grid(True)
plt.xlabel('time, s')
plt.show()
