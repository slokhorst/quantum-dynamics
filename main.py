import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation

Lx = 2
Nx = 1024
Nt = 100
dx = Lx/Nx
dt = 1
X = np.linspace(0, Lx, num=Nx)
mom = np.linspace(0, 2*np.pi/Lx, num=Nx)

T = np.arange(0, dt*Nt, step=dt, dtype=float)

def update(psi0, V, dx, dt):
	A = np.exp(-1j*dt*V)
	B = np.exp(-1j*dt*(mom)**2/2)

	psi1 = np.fft.ifft(B*np.fft.fft(A*psi0))
	return psi1

def wavepacket(x, sigma0, p0):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-x**2/(4*sigma0**2) + 1j*p0*x)

#V = 0*X#(X-1)**2
#V = (np.absolute(X)>0.5)
V = 1.5*(np.sign(X-0.7)+1-(np.sign(X-0.71)+1))*0
psi = np.ndarray([Nt,Nx], dtype=complex)
#psi[0] = 1/(X-Lx/2)
psi[0] = wavepacket(X-Lx/3, 0.1, 0)
#psi[0] = (np.absolute(X-Lx/2)<0.1)
for i in range(1,Nt):
	psi[i] = update(psi[i-1], V, dx, dt)

fig = plt.figure()
ax = plt.axes(xlim=(0, Lx), ylim=(0, 10))
line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,

def animate(i):
	x = X
	y = np.square(np.absolute(psi[i]))
	line.set_data(x,y)
	return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, animate, init_func=init,
                               #frames=Nt, interval=10, blit=True)
j = np.array(range(1,50,10))
plt.plot(X,V,X,np.transpose(np.square(np.absolute(psi[j]))))
plt.plot(X,V)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
XX, TT = np.meshgrid(X, T)
ax.plot_surface(XX,TT,np.square(np.absolute(psi)))
plt.show()