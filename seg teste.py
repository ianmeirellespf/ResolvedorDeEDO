import numpy as np
import matplotlib.pyplot as plt

G = 1.0  # Constante gravitacional
m = 1.0  # Massa das partículas

def f(t, x):
    r = np.linalg.norm(x)  # Distância entre as partículas
    return -G * m * x / r**3  # Aceleração devida à interação gravitacional

def g(t, v):
    return v

def modelo_orbita_runge_kutta(t, y):   #equações do sistema para runge-kutta
    # Parâmetros do sistema
    Grk = 1.0/100       # Gravitational constant
    m1 = 100.0    # Mass of the central body (e.g., planet)
    m2 = 1.0      # Mass of the satellite (e.g., spacecraft)

    # Variáveis de estado
    x, y, vx, vy = y[:4]  # Position and velocity components

    # Distance between the two bodies
    r = np.sqrt(x**2 + y**2)

    # Gravitational force components
    Fx = -Grk * m1 * m2 * x / r**3
    Fy = -Grk * m1 * m2 * y / r**3

    # Equations of motion for the orbital system
    dxdt = vx
    dydt = vy
    dvxdt = Fx / m2
    dvydt = Fy / m2

    return np.array([dxdt, dydt, dvxdt, dvydt])

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

def adams_bashforth3_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros((num_steps, 2))
    v = np.zeros_like(x)
    E = np.zeros(num_steps)

    x[0] = x0
    v[0] = v0
    x[1] = x[0] + dt * g(t[0], v[0])
    v[1] = v[0] + dt * f(t[0], x[0])
    x[2] = x[1] + dt * (1.5 * g(t[1], v[1]) - 0.5 * g(t[0], v[0]))
    v[2] = v[1] + dt * (1.5 * f(t[1], x[1]) - 0.5 * f(t[0], x[0]))
    E[0] = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x0)
    E[1] = 0.5 * m * np.linalg.norm(v[1])**2 - G * m / np.linalg.norm(x[1])
    E[2] = 0.5 * m * np.linalg.norm(v[2])**2 - G * m / np.linalg.norm(x[2])

    for i in range(3, num_steps):
        x[i] = x[i-1] + dt * ((23/12) * g(t[i-1], v[i-1]) - (16/12) * g(t[i-2], v[i-2]) + (5/12) * g(t[i-3], v[i-3]))
        v[i] = v[i-1] + dt * ((23/12) * f(t[i-1], x[i-1]) - (16/12) * f(t[i-2], x[i-2]) + (5/12) * f(t[i-3], x[i-3]))
        E[i] = 0.5 * m * np.linalg.norm(v[i])**2 - G * m / np.linalg.norm(x[i])

    return t, x, E

# Função para calcular a solução exata da órbita
def exact_solution(t, x0, v0):
    omega = np.sqrt(G * m / np.linalg.norm(x0)**3)
    x = np.zeros((len(t), 2))
    for i in range(len(t)):
        x[i] = x0 * np.cos(omega * t[i]) + v0 / omega * np.sin(omega * t[i])
    return x

#código separado para runge-kutta
def calcular_energia_orbita_runge_kutta(y):
    # Parâmetros do sistema
    Grk = 1.0 /100      # Gravitational constant
    m1 = 100.0    # Mass of the central body (e.g., planet)
    m2 = 1.0      # Mass of the satellite (e.g., spacecraft)

    # Variáveis de estado
    x, y, vx, vy = y[:4]  # Position and velocity components

    # Distance between the two bodies
    r = np.sqrt(x**2 + y**2)

    # Kinetic energy
    kinetic_energy = 0.5 * m2 * (vx**2 + vy**2)

    # Potential energy
    potential_energy = -Grk * m1 * m2 / r

    # Total energy (specific energy)
    energy = kinetic_energy + potential_energy

    return energy

def runge_kutta4(t0, y0, h, num_pontos):
    t = np.zeros(num_pontos)
    y = np.zeros((num_pontos, len(y0)))
    energia = np.zeros(num_pontos)
    t[0] = t0
    y[0] = y0
    energia[0] = calcular_energia_orbita_runge_kutta(y0)

    for i in range(1, num_pontos):
        k1 = h * modelo_orbita_runge_kutta(t[i-1], y[i-1])
        k2 = h * modelo_orbita_runge_kutta(t[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * modelo_orbita_runge_kutta(t[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * modelo_orbita_runge_kutta(t[i-1] + h, y[i-1] + k3)

        t[i] = t[i-1] + h
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        energia[i] = calcular_energia_orbita_runge_kutta(y[i])

    return t, y, energia

# Parâmetros gerais
t0 = 0.0
tf = 10.0
dt = 0.01
x0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])


#parametros para runge kutta
xr0 = 1.0  # Initial x-coordinate of the satellite
yr0 = 0.0  # Initial y-coordinate of the satellite
vx0 = 0.0  # Initial x-velocity of the satellite
vy0 = 1.0  # Initial y-velocity of the satellite
y0 = np.array([xr0, yr0, vx0, vy0])
t_inicio = 0.0
t_fim = 10.0
num_pontos = 1000
h = (t_fim - t_inicio) / num_pontos



# Chamada das funções
num_steps = int((tf - t0) / dt)
t_verlet, x_verlet, E_verlet = verlet_method(t0, tf, dt, x0, v0, f)
t_euler, x_euler, E_euler = euler_method(t0, tf, dt, x0, v0, f, g)
t_semi_euler, x_semi_euler, E_semi_euler = semi_implicit_euler_method(t0, tf, dt, x0, v0, f, g)
t_leapfrog, x_leapfrog, E_leapfrog = leapfrog_method(t0, tf, dt, x0, v0, f)
t_adams3, x_adams3, E_adams3 = adams_bashforth3_method(t0, tf, dt, x0, v0, f, g)
t_rk, x_rk, E_rk = runge_kutta4(t0, y0, h, num_pontos)
# Cálculo da solução exata
t_exact = np.linspace(t0, tf, num_steps)
x_exact = exact_solution(t_exact, x0, v0)
E_exact = 0.5 * m * np.linalg.norm(v0)**2 - G * m / np.linalg.norm(x_exact, axis=1)

# Plot dos resultados
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x_verlet[:, 0], x_verlet[:, 1],'r', label='Verlet')
plt.plot(x_euler[:, 0], x_euler[:, 1],'crimson', label='Euler')
plt.plot(x_semi_euler[:, 0], x_semi_euler[:, 1],'darkgreen', label='Euler Semi-implícito')
plt.plot(x_rk[:, 0], x_rk[:, 1],'lime', label='Runge-Kutta')
plt.plot(x_leapfrog[:, 0], x_leapfrog[:, 1],'b--', label='Leapfrog')
plt.plot(x_adams3[:, 0], x_adams3[:, 1],'fuchsia', label='Adams-Bashforth 3rd order')
plt.plot(x_exact[:, 0], x_exact[:, 1],'black' ,label='Solução Exata')
plt.title('Órbita - Posição')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t_verlet, E_verlet,'r', label='Verlet')
plt.plot(t_euler, E_euler,'crimson', label='Euler')
plt.plot(t_semi_euler, E_semi_euler, 'darkgreen',label='Euler Semi-implícito')
plt.plot(t_rk, E_rk, 'lime', label='Runge-Kutta')
plt.plot(t_leapfrog, E_leapfrog,'b--', label='Leapfrog')
plt.plot(t_adams3, E_adams3,'fuchsia', label='Adams-Bashforth 3rd order')
plt.plot(t_exact, E_exact, 'black',label='Solução Exata')
plt.title('Energia Total')
plt.xlabel('Tempo')
plt.ylabel('Energia')
plt.legend()

plt.tight_layout()
plt.show()
