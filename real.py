import numpy as np
import matplotlib.pyplot as plt

def modelo_hamiltoniano(t, y):
    # Parâmetros do sistema
    k = 1.0  # Constante da mola
    m = 1.0  # Massa do peso

    # Variáveis de estado
    x, p = y[:2]  # Posição e momento

    # Equações de movimento para o sistema hamiltoniano
    dxdt = p / m
    dpdt = -k * x

    return np.array([dxdt, dpdt])

def calcular_energia(y):
    # Parâmetros do sistema
    k = 1.0  # Constante da mola
    m = 1.0  # Massa do peso

    # Variáveis de estado
    x, p = y[:2]  # Posição e momento

    # Cálculo da energia
    energia = 0.5 * (p**2 / m + k * x**2)

    return energia

def runge_kutta4(t0, y0, h, num_pontos):
    t = np.zeros(num_pontos)
    y = np.zeros((num_pontos, len(y0)))
    energia = np.zeros(num_pontos)
    t[0] = t0
    y[0] = y0
    energia[0] = calcular_energia(y0)

    for i in range(1, num_pontos):
        k1 = h * modelo_hamiltoniano(t[i-1], y[i-1])
        k2 = h * modelo_hamiltoniano(t[i-1] + h/2, y[i-1] + k1/2)
        k3 = h * modelo_hamiltoniano(t[i-1] + h/2, y[i-1] + k2/2)
        k4 = h * modelo_hamiltoniano(t[i-1] + h, y[i-1] + k3)

        t[i] = t[i-1] + h
        y[i] = y[i-1] + (k1 + 2*k2 + 2*k3 + k4) / 6
        energia[i] = calcular_energia(y[i])

    return t, y, energia

# Condições iniciais
t0 = 0.0
x0 = 1.0
p0 = 0.0
y0 = np.array([x0, p0])

# Tempo de integração
t_inicio = 0.0
t_fim = 10.0
num_pontos = 100
h = (t_fim - t_inicio) / num_pontos

# Integração numérica usando Runge-Kutta de quarta ordem
t, y, energia = runge_kutta4(t0, y0, h, num_pontos)

# Extração das variáveis de estado
x = y[:, 0]
p = y[:, 1]

# Plotagem do gráfico
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='Posição')
plt.plot(t, p, label='Momento')
plt.xlabel('Tempo')
plt.ylabel('Variáveis de Estado')
plt.title('Sistema Hamiltoniano de Segunda Ordem')
plt.legend()
plt.grid(True)
plt.show()

# Plotagem da energia ao longo do tempo
plt.figure(figsize=(10, 6))
plt.plot(t, energia)
plt.xlabel('Tempo')
plt.ylabel('Energia')
plt.title('Energia do Sistema Hamiltoniano de Segunda Ordem')
plt.grid(True)
plt.show()
