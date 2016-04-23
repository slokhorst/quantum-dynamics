import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from numba import jit

Lx = 2
Nx = 1024
dt = 1
X = np.linspace(0, Lx, num=Nx)
mom = np.linspace(-np.pi/Lx, +np.pi/Lx, num=Nx)
Vwall = 100000

def update(psi0, V, dt):
	A = np.exp(-1j*dt*V)
	B = np.exp(-1j*dt*(mom)**2/2)
	psi1 = np.fft.ifft(B*np.fft.fft(A*psi0))
	return psi1

def wavepacket(x, sigma0, p0):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-x**2/(4*sigma0**2) + 1j*p0*x)

# IC:
V = 0*X
V[X<0.05*Lx] = Vwall
V[X>0.95*Lx] = Vwall
V += -1*(np.absolute(X-1)<0.01)
psi = np.ndarray([Nx], dtype=complex)
psi = wavepacket(X-0.2*Lx, 0.1, -5)

# plot:
fig = plt.figure()
ax = plt.axes(xlim=(0, Lx), ylim=(-2, 10))
psi_line, = ax.plot([], [], lw=2)

def init():
    psi_line.set_data([], [])
    return psi_line,

def animate(i):
	psi[:] = update(psi, V, dt)
	psi_line.set_data(X,np.square(np.absolute(psi)))
	return psi_line,

plt.plot(X,V)
anim = ani.FuncAnimation(fig, animate, init_func=init, interval=10, blit=True)
plt.show()
