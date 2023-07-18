import numpy as np
import matplotlib.pyplot as plt

def modelo_orbita(t, y):
    # Parâmetros do sistema
    G = 1.0       # Gravitational constant
    m1 = 100.0    # Mass of the central body (e.g., planet)
    m2 = 1.0      # Mass of the satellite (e.g., spacecraft)

    # Variáveis de estado
    x, y, vx, vy = y[:4]  # Position and velocity components

    # Distance between the two bodies
    r = np.sqrt(x**2 + y**2)

    # Gravitational force components
    Fx = -G * m1 * m2 * x / r**3
    Fy = -G * m1 * m2 * y / r**3

    # Equations of motion for the orbital system
    dxdt = vx
    dydt = vy
    dvxdt = Fx / m2
    dvydt = Fy / m2

    return np.array([dxdt, dydt, dvxdt, dvydt])

def calcular_energia_orbita(y):
    # Parâmetros do sistema
    G = 1.0       # Gravitational constant
    m1 = 100.0    # Mass of the central body (e.g., planet)
    m2 = 1.0      # Mass of the satellite (e.g., spacecraft)

    # Variáveis de estado
    x, y, vx, vy = y[:4]  # Position and velocity components

    # Distance between the two bodies
    r = np.sqrt(x**2 + y**2)

    # Kinetic energy
    kinetic_energy = 0.5 * m2 * (vx**2 + vy**2)

    # Potential energy
    potential_energy = -G * m1 * m2 / r

    # Total energy (specific energy)
    energy = kinetic_energy + potential_energy

    return energy

def runge_kutta4(t0, y0, h, num_pontos):
    t = np.zeros(num_pontos)
    y = np.zeros((num_pontos, len(y0)))
    energia = np.zeros(num_pontos)
    t[0] = t0
    y[0] = y0
    energia[0] = calcular_energia_orbita(y0)

    for i in range(1, num_pontos):
        k1 = h * modelo_orbita(t[i-1], y[i-1])
        k2 = h * modelo_orbita(t[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * modelo_orbita(t[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * modelo_orbita(t[i-1] + h, y[i-1] + k3)

        t[i] = t[i-1] + h
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        energia[i] = calcular_energia_orbita(y[i])

    return t, y, energia

# Condições iniciais
t0 = 0.0
x0 = 1.0  # Initial x-coordinate of the satellite
y0 = 0.0  # Initial y-coordinate of the satellite
vx0 = 0.0  # Initial x-velocity of the satellite
vy0 = 0.5  # Initial y-velocity of the satellite
y0 = np.array([x0, y0, vx0, vy0])

# Tempo de integração
t_inicio = 0.0
t_fim = 100.0
num_pontos = 10001
h = (t_fim - t_inicio) / num_pontos

# Integração numérica usando Runge-Kutta de quarta ordem
t, y, E = runge_kutta4(t0, y0, h, num_pontos)

# Extração das variáveis de estado
x = y[:, 0]
y = y[:, 1]

# Plotagem do gráfico
plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Órbita do Satélite')
plt.grid(True)
plt.axis('equal')  # Equal aspect ratio for x and y axes
plt.show()
