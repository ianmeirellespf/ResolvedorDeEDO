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

def runge_kutta_method(t0, tf, dt, x0, v0, f, g):
    num_steps = int((tf - t0) / dt) + 1
    t = np.linspace(t0, tf, num_steps)
    x = np.zeros_like(t)
    v = np.zeros_like(t)
    E = np.zeros_like(t)
    x[0] = x0
    v[0] = v0
    E[0] = 0.5 * m * (v[0]*L)**2 + m*gr*L * (1-np.cos(x[0]))
    for i in range(1, num_steps):
        k11 = g(t[i-1], v[i-1])
        k12 = f(t[i-1], x[i-1])
        k21 = g(t[i-1] + 0.5*dt, v[i-1] + 0.5*k11*dt)
        k22 = f(t[i-1] + 0.5*dt, x[i-1] + 0.5*k12*dt)
        k31 = g(t[i-1] + 0.5*dt, v[i-1] + 0.5*k21*dt)
        k32 = f(t[i-1] + 0.5*dt, x[i-1] + 0.5*k22*dt)
        k41 = g(t[i-1] + dt, v[i-1] + k31*dt)
        k42 = f(t[i-1] + dt, x[i-1] + k32*dt)
        v[i] = v[i-1] + (k12 + 2*k22 + 2*k32 + k42) * dt / 6
        x[i] = x[i-1] + (k11 + 2*k21 + 2*k31 + k41) * dt / 6
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
    return -m*g*L*np.sin(x)
def g(t,v): #x'=teta'=g(t,v)
    return v
'''
def exact_solution(t, x0, v0):
    omega = np.sqrt(k / m)
    x = x0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)
    v = v0 * np.cos(omega * t) - (x0 * omega * np.sin(omega * t))
    return x,v'''

t0 = 0.0
tf = 10.0
dt = 0.01
x0 = 1.0
v0 = 0.0

t_adams2, x_adams2, E_adams2 = adams_bashforth2_method(t0, tf, dt, x0, v0, f, g)
t_adams3, x_adams3, E_adams3 = adams_bashforth3_method(t0, tf, dt, x0, v0, f, g)
t_euler_semi, x_euler_semi, E_euler_semi = semi_implicit_euler_method(t0, tf, dt, x0, v0, f, g)
t_euler, x_euler, E_euler = euler_method(t0, tf, dt, x0, v0, f, g)
t_verlet, x_verlet, E_verlet = verlet_method(t0, tf, dt, x0, v0, f)
t_rk4, x_rk4, E_rk4 = runge_kutta_method(t0, tf, dt, x0, v0, f, g)
t_leapfrog, x_leapfrog, E_leapfrog = leapfrog_method(t0, tf, dt, x0, v0, f)

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