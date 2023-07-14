import numpy as np
import matplotlib.pyplot as plt

G = 1.0  # Constante gravitacional
m = 1.0  # Massa das partículas

def f(t, x):
    r = np.linalg.norm(x)  # Distância entre as partículas
    return -G * m * x / r**3  # Aceleração devida à interação gravitacional

def g(t, v):
    return v

def verlet_method(t0, tf, dt, x0, v0, f):
    num_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros((num_steps, 2))
    v = np.zeros_like(x)
    E = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x0)

    for i in range(1, num_steps):
        x[i] = x[i-1] + dt * v[i-1] + 0.5 * dt**2 * f(t[i-1], x[i-1])
        v[i] = v[i-1] + dt * (f(t[i-1], x[i-1]) + f(t[i], x[i])) / 2
        E[i] = 0.5 * m * np.linalg.norm(v[i])**2 - G * m / np.linalg.norm(x[i])

    return t, x, E

def euler_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros((num_steps, 2))
    v = np.zeros_like(x)
    E = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x0)

    for i in range(1, num_steps):
        v[i] = v[i-1] + dt * f(t[i-1], x[i-1])
        x[i] = x[i-1] + dt * g(t[i-1], v[i-1])
        E[i] = 0.5 * m * np.linalg.norm(v[i])**2 - G * m / np.linalg.norm(x[i])

    return t, x, E

def semi_implicit_euler_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros((num_steps, 2))
    v = np.zeros_like(x)
    E = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x0)

    for i in range(1, num_steps):
        v[i] = v[i-1] + dt * f(t[i-1], x[i-1])
        x[i] = x[i-1] + dt * g(t[i], v[i])
        E[i] = 0.5 * m * np.linalg.norm(v[i])**2 - G * m / np.linalg.norm(x[i])

    return t, x, E

def runge_kutta_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros((num_steps, 2))
    v = np.zeros_like(x)
    E = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x0)

    for i in range(1, num_steps):
        k11 = g(t[i], v[i-1])
        k12 = f(t[i], x[i-1])
        k21 = g(t[i] + 0.5*dt, v[i-1] + 0.5*k11*dt)
        k22 = f(t[i] + 0.5*dt, x[i-1] + 0.5*k12*dt)
        k31 = g(t[i] + 0.5*dt, v[i-1] + 0.5*k21*dt)
        k32 = f(t[i] + 0.5*dt, x[i-1] + 0.5*k22*dt)
        k41 = g(t[i] + dt, v[i-1] + k31*dt)
        k42 = f(t[i] + dt, x[i-1] + k32*dt)
        v[i] = v[i-1] + (k12 + 2*k22 + 2*k32 + k42)*dt / 6
        x[i] = x[i-1] + (k11 + 2*k21 + 2*k31 + k41)*dt / 6
        E[i] = 0.5 * m * np.linalg.norm(v[i])**2 - G * m / np.linalg.norm(x[i])

    return t, x, E

def leapfrog_method(t0, tf, dt, x0, v0, f):
    num_steps = int((tf - t0) / dt)
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros((num_steps, 2))
    v = np.zeros_like(x)
    E = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x0)

    for i in range(1, num_steps):
        x[i] = x[i-1] + dt * v[i-1] + 0.5 * dt**2 * f(t[i-1], x[i-1])
        v[i] = v[i-1] + dt * (f(t[i-1], x[i-1]) + f(t[i], x[i])) / 2
        E[i] = 0.5 * m * np.linalg.norm(v[i])**2 - G * m / np.linalg.norm(x[i])

    return t, x, E

# Parâmetros iniciais
t0 = 0.0
tf = 10.0
dt = 0.01
x0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])

# Chamada das funções
t_verlet, x_verlet, E_verlet = verlet_method(t0, tf, dt, x0, v0, f)
t_euler, x_euler, E_euler = euler_method(t0, tf, dt, x0, v0, f, g)
t_semi_euler, x_semi_euler, E_semi_euler = semi_implicit_euler_method(t0, tf, dt, x0, v0, f, g)
t_rk, x_rk, E_rk = runge_kutta_method(t0, tf, dt, x0, v0, f, g)
t_leapfrog, x_leapfrog, E_leapfrog = leapfrog_method(t0, tf, dt, x0, v0, f)

# Plot dos resultados
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x_verlet[:, 0], x_verlet[:, 1], label='Verlet')
plt.plot(x_euler[:, 0], x_euler[:, 1], label='Euler')
plt.plot(x_semi_euler[:, 0], x_semi_euler[:, 1], label='Euler Semi-implícito')
plt.plot(x_rk[:, 0], x_rk[:, 1], label='Runge-Kutta')
plt.plot(x_leapfrog[:, 0], x_leapfrog[:, 1], label='Leapfrog')
plt.title('Órbita - Posição')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_verlet, E_verlet, label='Verlet')
plt.plot(t_euler, E_euler, label='Euler')
plt.plot(t_semi_euler, E_semi_euler, label='Euler Semi-implícito')
plt.plot(t_rk, E_rk, label='Runge-Kutta')
plt.plot(t_leapfrog, E_leapfrog, label='Leapfrog')
plt.title('Energia Total')
plt.xlabel('Tempo')
plt.ylabel('Energia')
plt.legend()

plt.tight_layout()
plt.show()
