import numpy as np
import matplotlib.pyplot as plt

m=1
L=1
gr=1

def adams_bashforth2_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    x[1] = x[0] + dt * g(t[0], v[0])
    v[1] = v[0] + dt * f(t[0], x[0])
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    E[1] = 0.5 * m * (v[1]*L)**2 + m*gr*L * (1-np.cos(x[1]))
    for i in range(2, num_steps):
        x[i] = x[i-1] + dt * (1.5 * g(t[i-1], v[i-1]) - 0.5 * g(t[i-2], v[i-2]))
        v[i] = v[i-1] + dt * (1.5 * f(t[i-1], x[i-1]) - 0.5 * f(t[i-2], x[i-2]))
        E[i] = 0.5 * m * (v[i]*L)**2 + m*gr*L * (1-np.cos(x[i]))
    return t, x, E

def adams_bashforth3_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    x[1] = x[0] + dt * g(t[0], v[0])
    v[1] = v[0] + dt * f(t[0], x[0])
    x[2] = x[1] + dt * (1.5 * g(t[1], v[1]) - 0.5 * g(t[0], v[0]))
    v[2] = v[1] + dt * (1.5 * f(t[1], x[1]) - 0.5 * f(t[0], x[0]))
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    E[1] = 0.5 * m * (v[1]*L)**2 + m*gr*L * (1-np.cos(x[1]))
    E[2] = 0.5 * m * (v[2]*L)**2 + m*gr*L * (1-np.cos(x[2]))
    for i in range(3, num_steps):
        x[i] = x[i-1] + dt * ((23/12) * g(t[i-1], v[i-1]) - (16/12) * g(t[i-2], v[i-2]) + (5/12) * g(t[i-3], v[i-3]))
        v[i] = v[i-1] + dt * ((23/12) * f(t[i-1], x[i-1]) - (16/12) * f(t[i-2], x[i-2]) + (5/12) * f(t[i-3], x[i-3]))
        E[i] = 0.5 * m * (v[i]*L)**2 + m*gr*L * (1-np.cos(x[i]))
    return t, x, E

def semi_implicit_euler_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    for i in range(1, num_steps):
        v[i] = v[i-1] + dt * f(t[i-1], x[i-1])
        x[i] = x[i-1] + dt * g(t[i-1], v[i-1])
        E[i] = 0.5 * m * (v[i]*L)**2 + m*gr*L * (1-np.cos(x[i]))
    return t, x, E

def euler_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    for i in range(1, num_steps):
        v[i] = v[i-1] + dt * f(t[i-1], x[i-1])
        x[i] = x[i-1] + dt * g(t[i-1], v[i-1])
        E[i] = 0.5 * m * (v[i]*L)**2 + m*gr*L * (1-np.cos(x[i]))
    return t, x, E

def verlet_method(t0, tf, dt, x0, v0, f):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    for i in range(1, num_steps):
        x[i] = x[i-1] + dt * v[i-1] + 0.5 * dt**2 * f(t[i-1], x[i-1])
        v[i] = v[i-1] + 0.5 * dt * (f(t[i-1], x[i-1]) + f(t[i], x[i]))
        E[i] = 0.5 * m * (v[i]*L)**2 + m*gr*L * (1-np.cos(x[i]))
    return t, x, E

def leapfrog_method(t0, tf, dt, x0, v0, f):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    for i in range(1, num_steps):
        x[i] = x[i-1] + dt * v[i-1] + 0.5 * dt**2 * f(t[i-1], x[i-1])
        v[i] = v[i-1] + dt * (f(t[i-1], x[i-1]) + f(t[i], x[i])) / 2
        E[i] = 0.5 * m * (v[i]*L)**2 + m*gr*L * (1-np.cos(x[i]))
    return t, x, E

# Exemplo de EDO
# Exemplo de EDO - Pendulo
def f(t, x): #v'=w'= f(t,x)
    L=1
    g=1
    m=1
    return -(g/L)*np.sin(x)
def g(t,v): #x'=teta'=g(t,v)
    return v
'''
def exact_solution(t, x0, v0):
    omega = np.sqrt(k / m)
    x = x0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)
    v = v0 * np.cos(omega * t) - (x0 * omega * np.sin(omega * t))
    return x,v'''

#código separado para runge_kutta
def modelo_hamiltoniano(t, y):
    # Parâmetros do sistema
    L=1
    gr=1
    m=1

    # Variáveis de estado
    x, p = y[:2]  # Posição e momento

    # Equações de movimento para o sistema hamiltoniano
    dxdt = p / (m*L**2)
    dpdt = -m*gr*L*np.sin(x)

    return np.array([dxdt, dpdt])

def calcular_energia(y):
    # Parâmetros do sistema
    L=1
    gr=1
    m=1

    # Variáveis de estado
    x, p = y[:2]  # Posição e momento

    # Cálculo da energia
    energia = 0.5 *m* ((p/(m*L)))**2 + m*gr*L * (1-np.cos(x))


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
#parâmetros gerais
t0 = 0.0
tf = 10.0
dt = 0.01
x0 = 1.0
v0 = 0.0

#parâmetros para runge-kutta
p0 = 0.0
y0 = np.array([x0, p0])
t_inicio = 0.0
t_fim = 10.0
num_pontos = 1000
h = (t_fim - t_inicio) / num_pontos


t_adams2, x_adams2, E_adams2 = adams_bashforth2_method(t0, tf, dt, x0, v0, f, g)
t_adams3, x_adams3, E_adams3 = adams_bashforth3_method(t0, tf, dt, x0, v0, f, g)
t_euler_semi, x_euler_semi, E_euler_semi = semi_implicit_euler_method(t0, tf, dt, x0, v0, f, g)
t_euler, x_euler, E_euler = euler_method(t0, tf, dt, x0, v0, f, g)
t_verlet, x_verlet, E_verlet = verlet_method(t0, tf, dt, x0, v0, f)
t_leapfrog, x_leapfrog, E_leapfrog = leapfrog_method(t0, tf, dt, x0, v0, f)
t_rk4, y, E_rk4 = runge_kutta4(t0, y0, h, num_pontos)
x_rk4=y[:,0]

#t_exact = np.linspace(t0, tf, int((tf - t0) / dt) + 1)
#x_exact,v_exact = exact_solution(t_exact, x0, v0)
#E_exact = 0.5 * m * v_exact**2 + 0.5 * k * x_exact**2


plt.plot(t_adams2, E_adams2,'--',color='indigo', label='Adams-Bashforth 2nd order')
plt.plot(t_adams3, E_adams3,'crimson', label='Adams-Bashforth 3rd order')
plt.plot(t_euler_semi, E_euler_semi,'darkgreen', label='Euler Semi-implicit')
plt.plot(t_euler, E_euler, 'lime',label='Euler')
plt.plot(t_verlet, E_verlet,'aqua', label='Verlet')
plt.plot(t_rk4, E_rk4, '--',color='fuchsia',label='Runge-Kutta')
plt.plot(t_leapfrog, E_leapfrog,'b--' , label='Leapfrog')
#plt.plot(t_exact, x_exact, label='Exact Solution')
plt.xlabel('Time')
plt.ylabel('Total Energy')
plt.legend()
plt.show()