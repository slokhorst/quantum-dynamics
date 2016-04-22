import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation

Lx = 2
Nx = 1024
Nt = 10000
dt = 1
X = np.linspace(0, Lx, num=Nx)
mom = np.linspace(-np.pi/Lx, +np.pi/Lx, num=Nx)

T = np.arange(0, dt*Nt, step=dt, dtype=float)

def update(psi0, V, dt):
	A = np.exp(-1j*dt*V)
	B = np.exp(-1j*dt*(mom)**2/2)

	psi1 = np.fft.ifft(B*np.fft.fft(A*psi0))
	return psi1

def wavepacket(x, sigma0, p0):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-x**2/(4*sigma0**2) + 1j*p0*x)

V = 0*X
V[X<0.05*Lx] = 10
V[X>0.95*Lx] = 10
V += -1*(np.absolute(X-1)<0.005)
#V = 0*X#1.5*(np.sign(X-0.7)+1-(np.sign(X-0.71)+1))*0
psi = np.ndarray([Nt,Nx], dtype=complex)
psi[0] = wavepacket(X-0.2*Lx, 0.1, -50)
for i in range(1,Nt):
	psi[i] = update(psi[i-1], V, dt)

fig = plt.figure()
ax = plt.axes(xlim=(0, Lx), ylim=(-2, 10))
psi_line, = ax.plot([], [], lw=2)
def init():
    psi_line.set_data([], [])
    return psi_line,

def animate(i):
	x = X
	y = np.square(np.absolute(psi[i]))
	psi_line.set_data(x,y)
	return psi_line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=1, blit=False)

#j = np.array(range(1,50,10))
#plt.plot(X,V,X,np.transpose(np.square(np.absolute(psi[j]))))
plt.plot(X,V)
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# XX, TT = np.meshgrid(X, T)
# ax.plot_surface(XX,TT,np.square(np.absolute(psi)))
# plt.show()