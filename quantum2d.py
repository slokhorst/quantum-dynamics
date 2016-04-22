import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation
import time

Lx = 2
Ly = 2
Nx = 128
Ny = 128
Nt = 1000
dt = 1
X = np.linspace(0, Lx, num=Nx)
Y = np.linspace(0, Ly, num=Ny)
momx = np.linspace(-np.pi/Lx, +np.pi/Lx, num=Nx)
momy = np.linspace(-np.pi/Ly, +np.pi/Ly, num=Ny)

XX, YY = np.meshgrid(X,Y)
MX, MY = np.meshgrid(momx,momy)

T = np.arange(0, dt*Nt, step=dt, dtype=float)

def potential(x,y):
	if (x > 0.79*Lx and x < 0.81*Lx) and ((y < 0.55*Ly or y > 0.56*Ly) and (y < 0.44*Ly or y > 0.45*Ly)):
		return 1
	else:
		return 0
potential = np.vectorize(potential)

def update(psi0, V, dt):
	A = np.exp(-1j*dt*V)
	B = np.exp(-1j*dt*(MX**2+MY**2)/2)

	psi1 = np.fft.ifft2(B*np.fft.fft2(A*psi0))
	return psi1

def wavepacket(x, y, sigma0, p0x, p0y):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-(x**2+y**2)/(4*sigma0**2) + 1j*(p0x*x+p0y*y))

def plane_wave(x, y, sigma0, p0x, p0y):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-(x**2)/(4*sigma0**2) + 1j*(p0x*x))

def plane_wave1(x,p0x):
	return np.exp(x+1j*(p0x*x))

V = potential(XX,YY)
#V += -1*(np.absolute(X-1)<0.005)
#V = 0*X#1.5*(np.sign(X-0.7)+1-(np.sign(X-0.71)+1))*0
psi = np.ndarray([Nt,Nx,Ny], dtype=complex)
#psi[0] = wavepacket(XX-0.5*Lx, YY-0.5*Ly, 0.1, -50, 0)
#psi[0] = plane_wave(XX-0.2*Lx, YY, 0.01, -50, 0)
psi[0] = plane_wave1(XX-0.2*Lx,-50)
for i in range(1,Nt):
	psi[i] = update(psi[i-1], V, dt)

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.square(np.absolute(psi[i])))
plt.show(block=False)
def init():
    return

def animate(i):
	Z = np.square(np.absolute(psi[i]))
	Z += V
	# wait for a second
	#time.sleep(1)
	# replace the image contents
	im.set_array(Z)
	# redraw the figure
	fig.canvas.draw()

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=1, blit=False)

#j = np.array(range(1,50,10))
#plt.plot(X,V,X,np.transpose(np.square(np.absolute(psi[j]))))
#plt.plot(X,V)
plt.show()
