import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from numba import jit

class Quantum2D:
	def __init__(self, Lx, Ly, Nx, Ny, dt, bc_func=None):
		self.t = 0
		self.dt = dt
		self.bc_func = bc_func
		self.X = np.linspace(0, Lx, num=Nx)
		self.Y = np.linspace(0, Ly, num=Ny)
		self.momx = np.linspace(-np.pi/Lx, +np.pi/Lx, num=Nx)
		self.momy = np.linspace(-np.pi/Ly, +np.pi/Ly, num=Ny)
		self.XX, self.YY = np.meshgrid(self.X,self.Y)
		self.MX, self.MY = np.meshgrid(self.momx,self.momy)
		self.V = np.ndarray([Nx,Ny], dtype=float)
		self.V[:,:] = 0
		self.psi = np.ndarray([Nx,Ny], dtype=complex)
		self.psi[:,:] = 0
		self.step_pos = np.exp(-1j*self.dt*self.V)
		self.step_mom = np.exp(-1j*dt*(self.MX**2+self.MY**2)/2)

	def set_V(self, V):
		self.V[:,:] = V
		self.step_pos = np.exp(-1j*self.dt*self.V)

	def set_psi(self, psi):
		self.psi[:,:] = psi

	@jit
	def update(self):
		if self.bc_func: # if a boundary condition is specified, enforce it now
			self.bc_func(self)
		self.psi[:,:] = np.fft.ifft2(self.step_mom*np.fft.fft2(self.step_pos*self.psi))
		self.t += self.dt

	def psi_wavepacket(self, x, sigma0, p0):
		return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-x**2/(4*sigma0**2) + 1j*p0*x)

	def plot(self):
		self.fig = plt.figure()
		self.ax1 = self.fig.add_subplot(121)
		self.ax1.set_title('$|\Psi|^2$')
		self.im1 = self.ax1.imshow(np.square(np.absolute(self.psi)))
		self.ax2 = self.fig.add_subplot(122)
		self.ax2.set_title('$Re(\Psi)^2$')
		self.im2 = self.ax2.imshow(np.square(np.real(self.psi)))
		anim = ani.FuncAnimation(self.fig, self._ani_update, init_func=self._ani_init)
		plt.show()

	def _ani_init(self):
		return

	def _ani_update(self,i):
		self.update()
		self.im1.set_array(np.square(np.absolute(self.psi))+V)
		self.im2.set_array(np.square(np.real(self.psi))+V)
		self.fig.canvas.draw()
		return

if __name__ == "__main__":
	# helper functions...
	# box to enclose the domain so the wavefunction doesn't escape
	def V_container(x, y, Vwall, Lx, Ly):
		return Vwall*(x<0.01*Lx or x>0.99*Lx or y<0.01*Ly or y>0.99*Ly)
	V_container = np.vectorize(V_container)

	# wall with two slits
	def V_slit(x, y, Vwall, wallX, wallT, slitY1, slitY2, slitT):
		return Vwall*((x > wallX-wallT and x < wallX+wallT) and ((y < slitY1-slitT/2 or y > slitY1+slitT/2) and (y < slitY2-slitT/2 or y > slitY2+slitT/2)))
	V_slit = np.vectorize(V_slit)

	# initial wave function for a wave packet
	def psi_wavepacket(x, y, sigma0, p0x, p0y):
		return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-(x**2+y**2)/(4*sigma0**2) + 1j*(p0x*x+p0y*y))

	# wave function for a plane wave, useful for double-slit experiment
	def psi_plane_wave(x, y, kx, ky, w, t):
		return np.exp(1j*(x*kx+y*ky - w*t))

	Lx = 2
	Ly = 2
	Nx = 256
	Ny = 256
	dt = 1

	# parameters for the geometry
	Vwall = 100000
	wallX = 0.5*Lx
	wallT = 0.01
	slitY1 = 0.45*Ly
	slitY2 = 0.55*Ly
	slitT = 0.04

	# pass a boundary condition that will be enforced after every time step
	def boundary_condition(q2d: Quantum2D):
		q2d.psi[:,q2d.Y<Ly/2-wallT/2] = psi_plane_wave(q2d.XX,q2d.YY,10,0,1,q2d.t)[:,q2d.Y<Ly/2-wallT/2]

	q2d = Quantum2D(Lx, Ly, Nx, Ny, dt, bc_func=boundary_condition)

	# potential consists of an enclosing box and a centered wall with a double slit
	V = V_container(q2d.XX, q2d.YY, Vwall, Lx, Ly) + V_slit(q2d.XX, q2d.YY, Vwall, wallX, wallT, slitY1, slitY2, slitT)
	q2d.set_V(V)

	# psi is a plane wave on the left side
	psi = np.ndarray([Nx,Ny], dtype=complex)
	psi[:,:] = 0
	psi[:,q2d.Y<Ly/2-wallT/2] = psi_plane_wave(q2d.XX,q2d.YY,10,0,1,0)[:,q2d.Y<Ly/2-wallT/2]
	#psi[:] = psi_wavepacket(XX-0.3*Lx, YY-0.5*Ly, 0.1*Lx, 10, 0)
	q2d.set_psi(psi)

	q2d.plot()
