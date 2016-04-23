import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from numba import jit

Lx = 2
Ly = 2
Nx = 256
Ny = 256
dt = 1
X = np.linspace(0, Lx, num=Nx)
Y = np.linspace(0, Ly, num=Ny)
momx = np.linspace(-np.pi/Lx, +np.pi/Lx, num=Nx)
momy = np.linspace(-np.pi/Ly, +np.pi/Ly, num=Ny)

XX, YY = np.meshgrid(X,Y)
MX, MY = np.meshgrid(momx,momy)

Vwall = 100000

wallX = 0.5*Lx
wallT = 0.01
slitY1 = 0.45*Ly
slitY2 = 0.55*Ly
slitT = 0.04

# make sure the wavefunction doesn't escape the box
def V_container(x, y, Vwall, Lx, Ly):
	return Vwall*(x<0.01*Lx or x>0.99*Lx or y<0.01*Ly or y>0.99*Ly)
V_container = np.vectorize(V_container)

# wall with two slits
def V_slit(x, y, Vwall, wallX, wallT, slitY1, slitY2, slitT):
	return Vwall*((x > wallX-wallT and x < wallX+wallT) and ((y < slitY1-slitT/2 or y > slitY1+slitT/2) and (y < slitY2-slitT/2 or y > slitY2+slitT/2)))
V_slit = np.vectorize(V_slit)

def V_geometry(x, y):
	return V_container(XX, YY, Vwall, Lx, Ly) + V_slit(XX, YY, Vwall, wallX, wallT, slitY1, slitY2, slitT)

def psi_wavepacket(x, y, sigma0, p0x, p0y):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-(x**2+y**2)/(4*sigma0**2) + 1j*(p0x*x+p0y*y))

def psi_plane_wave(x, y, kx, ky, w, t):
	return np.exp(1j*(x*kx+y*ky - w*t))

@jit
def update(psi0, V, dt):
	A = np.exp(-1j*dt*V)
	B = np.exp(-1j*dt*(MX**2+MY**2)/2)

	psi1 = np.fft.ifft2(B*np.fft.fft2(A*psi0))
	return psi1

V = V_geometry(XX, YY)

psi = np.ndarray([Nx,Ny], dtype=complex)
psi[:,Y<Ly/2-wallT/2] = psi_plane_wave(XX,YY,10,0,1,0)[:,Y<Ly/2-wallT/2]
#psi[:] = psi_wavepacket(XX-0.3*Lx, YY-0.5*Ly, 0.1*Lx, 10, 0)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.set_title('$|\Psi|^2$')
im1 = ax1.imshow(np.square(np.absolute(psi)))
ax2 = fig.add_subplot(122)
ax2.set_title('$Re(\Psi)^2$')
im2 = ax2.imshow(np.square(np.real(psi)))
t=0

def animate(i):
	global t
	t += dt
	#force a plane wave in the left side
	psi[:,Y<Ly/2-wallT/2] = psi_plane_wave(XX,YY,10,0,1,t)[:,Y<Ly/2-wallT/2]
	psi[:] = update(psi, V, dt)
	im1.set_array(np.square(np.absolute(psi))+V)
	im2.set_array(np.square(np.real(psi))+V)
	fig.canvas.draw()

anim = ani.FuncAnimation(fig, animate, interval=10, blit=False)
plt.show()
