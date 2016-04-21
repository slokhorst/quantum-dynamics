import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Lx = 2
Nx = 1024
Nt = 100
dx = Lx/Nx
dt = 1
X = np.linspace(0, Lx, num=Nx)
mom = np.linspace(0, 1/Lx, num=Nx)

T = np.arange(0, dt*Nt, step=dt, dtype=float)

def update(psi0, V, dx, dt):
	A = np.exp(-dt*V)
	B = np.exp(-dt*(mom-1/Lx/2)**2/2)

	psi1 = np.fft.ifft(np.fft.ifftshift(B*np.fft.fftshift(np.fft.fft(A*psi0))))
	return psi1

def wavepacket(x, sigma0, p0):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-x**2/(4*sigma0**2) + 1j*p0*x)

V = 0*X#(X-1)**2
#V = (np.absolute(X)>0.5)

psi = np.ndarray([Nt,Nx], dtype=complex)
#psi[0] = 1/(X-Lx/2)
psi[0] = wavepacket(X-Lx/2, 0.1, 1)
#psi[0] = (np.absolute(X-Lx/2)<0.1)
for i in range(1,Nt):
	psi[i] = update(psi[i-1], V, dx, dt)

plt.plot(X,V,X,np.transpose(np.square(np.absolute(psi))))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
XX, TT = np.meshgrid(X, T)
ax.plot_surface(XX,TT,np.square(np.absolute(psi)))
plt.show()