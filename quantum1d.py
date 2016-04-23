import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from numba import jit

class Quantum1D:
	def __init__(self, Lx, Nx, dt):
		self.t = 0
		self.dt = dt
		self.X = np.linspace(0, Lx, num=Nx)
		self.mom = np.linspace(-np.pi/Lx, +np.pi/Lx, num=Nx)
		self.V = np.ndarray([Nx], dtype=float)
		self.V[:] = 0
		self.psi = np.ndarray([Nx], dtype=complex)
		self.psi[:] = 0
		self.step_pos = np.exp(-1j*self.dt*self.V)
		self.step_mom = np.exp(-1j*self.dt*self.mom**2/2)

	def set_V(self, V):
		self.V[:] = V
		self.step_pos = np.exp(-1j*self.dt*self.V)

	def set_psi(self, psi):
		self.psi[:] = psi

	@jit
	def update(self):
		self.psi[:] = np.fft.ifft(self.step_mom*np.fft.fft(self.step_pos*self.psi))
		self.t += self.dt

	def plot(self):
		self.fig = plt.figure()
		self.ax = plt.axes(xlim=(self.X[0], self.X[-1]), ylim=(-2, 10))
		self.ax.set_title('$|\Psi|^2$')
		self.psi_line, = self.ax.plot([], [], lw=2)
		plt.plot(self.X,self.V)
		anim = ani.FuncAnimation(self.fig, self._ani_update, init_func=self._ani_init, interval=10, blit=True)
		plt.show()

	def _ani_init(self):
		self.psi_line.set_data([], [])
		return self.psi_line,

	def _ani_update(self,i):
		self.update()
		self.psi_line.set_data(self.X,np.square(np.absolute(self.psi)))
		return self.psi_line,

if __name__ == "__main__":
	# helper function...
	def psi_wavepacket(x, sigma0, p0):
		return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-x**2/(4*sigma0**2) + 1j*p0*x)

	Lx = 2
	Nx = 1024
	dt = 1
	q1d = Quantum1D(Lx, Nx, dt)

	# parameter for the geometry
	Vwall = 100000

	# potential consists of walls to enclose the wave function ...
	V = 0*q1d.X
	V[q1d.X<0.05*Lx] = Vwall
	V[q1d.X>0.95*Lx] = Vwall
	# ... and a small well in the center
	V += -0.1*(np.absolute(q1d.X-1)<0.02)
	q1d.set_V(V)

	# initial wave function is a wave packet with some momentum
	psi = psi_wavepacket(q1d.X-0.2*Lx, 0.1, -10)
	q1d.set_psi(psi)

	q1d.plot()
