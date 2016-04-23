import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import time

Lx = 4
Ly = 4
Nx = 351
Ny = 351
Nt = 2500
dt = 2
X = np.linspace(0, Lx, num=Nx)
Y = np.linspace(0, Ly, num=Ny)
momx = np.linspace(-np.pi/Lx, np.pi/Lx, num=Nx)
momy = np.linspace(-np.pi/Ly, np.pi/Ly, num=Ny)

XX, YY = np.meshgrid(X,Y)
MX, MY = np.meshgrid(momx,momy)

T = np.arange(0, dt*Nt, step=dt, dtype=float)



xs = 0.5 #start position potential wall
xe = 0.54 # end position potential wall
xss = 0.4 # start smoothing potential
xw = 0.1 # percentge thickness of boundary wall
yw = 0.2 # percentage thickness of b.wall y-direction
ssd = 0.07 # slit slit distance
sw  = 0.025# slit width

def potential(x,y):
    
    V = np.zeros((len(x),len(y)))
    
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            if ((x[i,j] >= xs*Lx and x[i,j] <= xe*Lx) and (( y[i,j] >= (ssd/2+0.5+sw)*Ly or y[i,j] <= (ssd/2+0.5)*Ly) and (y[i,j] >= (-ssd/2+0.5)*Ly \
                                                                                                                 or y[i,j] <= (-ssd/2+0.5-sw)*Ly))):
                V[i,j] = 1
            # smoothing of potential
            if (y[i,j] - x[i,j] + (-yw+xss)*Lx < 0) and (x[i,j] > xss*Lx and x[i,j] < xs*Lx):
                V[i,j] = 1
            if (y[i,j] + x[i,j] - (-yw+xss)*Lx > Ly) and (x[i,j] > xss*Lx and x[i,j] < xs*Lx):
                V[i,j] = 1
    return V

def update(psi0, V, dt):
    A = np.exp(-1j*dt*V)
    B = np.exp(-1j*dt*(MX**2+MY**2)/2)

    psi1 = np.fft.ifft2(B*np.fft.fft2(A*psi0))
    return psi1

#def wavepacket(x, y, sigma0, p0x, p0y):
#	return np.exp(-(x**2+y**2)/(4*sigma0**2) + 1j*(p0x*x+p0y*y))

def wavepacket(x, y, sigma0, p0x, p0y):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-(x**2+y**2)/(4*sigma0**2) + 1j*(p0x*x+p0y*y))

def plane_wave(x, y, sigma0, p0x, p0y):
	return np.power(np.sqrt(2*np.pi)*sigma0,-0.5) * np.exp(-(x**2)/(4*sigma0**2) + 1j*(p0x*x))

def plane_wave1(x, y, Lx, Ly, p0x):
    psi = np.zeros((len(x),len(y)))
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            if (x[i,j] >= 0.1*Lx and x[i,j] <= 0.5*Lx) and (y[i,j] > yw*Ly and y[i,j] < (1-yw)*Ly):
                psi[i,j] = np.cos((12*x[i,j]))-1j*np.sin(p0x*x[i,j])
    return psi

def potential1(x, y, Lx, Ly, xw, yw):
    V = np.zeros((len(x),len(y)))
    for i in range(0,len(x)):
        for j in range(0,len(y)):
            if (x[i,j] >= 0 and x[i,j] <= xw*Lx):
                V[i,j] = 1
            if (x[i,j] >= (1-xw)*Lx):
                V[i,j] = 1
            if (y[i,j] >= 0 and y[i,j] <= yw*Ly):
                V[i,j] = 1
            if (y[i,j] >= (1-yw)*Ly):
                V[i,j] = 1
    return V
    

V = 10e15*potential(XX, YY)
V += 10e15*potential1(XX, YY, Lx, Ly, xw, yw)

psi = np.ndarray([2,Nx,Ny], dtype=complex)
print(psi.shape)
psi[0] = plane_wave1(XX, YY, Lx, Ly, 10)

Z = np.absolute(psi[0])

plt.imshow(Z)
plt.show()
plt.imshow(V)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
im = ax.imshow(np.square(np.absolute(psi[0])))#, norm=LogNorm(vmin=0.1, vmax=1))
plt.show(block=False)
def init():
    return

def animate(i):
    j = 1
    psi[j] = update(psi[j-1], V, dt)
    psi[0] = psi[1]
    Z = np.square(np.absolute((psi[0])))
    print(i)
	# wait for a second
	#time.sleep(1)
	# replace the image contents
    im.set_array(Z)
	# redraw the figure
    fig.canvas.draw()

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Nt, interval=1, blit=False)

plt.show()