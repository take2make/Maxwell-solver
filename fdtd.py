import numpy as np
import matplotlib.pyplot as plt
from constants import eps_0, mu_0, c_0


class solve_maxwell():
	def __init__(self, fs, Z, NT, Nz, Nz_source, Nz_bound, param):
		dz = Z/Nz
		dt = dz/c_0

		self.eps = np.ones(Nz)
		self.mu = np.ones(Nz)
		self.sigma = np.zeros(Nz)

		self.CM = dt/dz*1/(self.mu*mu_0)
		self.CE = dt/dz*1/(self.eps*eps_0)

		self.big_epsilon_area(Nz, Nz_bound, param)
		self.reconstruction_coefficients(dt)
		self.boundary_conditions(dz, dt, Nz, NT, Nz_source)
		self.solve_fdtd(Nz,NT)

	def big_epsilon_area(self, Nz, Nz_bound, param):
		for i in np.arange(1, Nz):
			if i >= Nz_bound:
				self.eps[i] = param[0]
				self.mu[i] = param[1]
				self.sigma[i] = param[2]
		pass

	def reconstruction_coefficients(self, dt):
		self.CM = self.CM/self.mu
		self.CE = self.CE/self.eps
		self.CE = self.CE/(1+self.sigma*dt/(self.eps*2*eps_0))
		self.CS = (1-self.sigma*dt/(self.eps*2*eps_0))/(1+self.sigma*dt/(self.eps*2*eps_0))
		pass

	def boundary_conditions(self, dz, dt, Nz, NT, Nz_source):
		z = np.arange(0,Nz*dz,dz)
		t = np.arange(0,NT*dt,dt)

		self.Ey = 10*np.exp(-((z-dz*Nz_source)/(100*dz))**2)
		self.Hx = np.zeros(Nz)
		self.Hx = np.exp(-((z-dz*Nz_source)/(5*dz))**2)
		self.E = np.zeros((Nz,NT))
		self.H = np.zeros((Nz,NT))
		self.I = np.zeros((Nz,NT))
		pass

	def solve_fdtd(self, Nz, NT):
		for i in np.arange(0, NT-1):

			self.Hx[0]=self.Hx[1]

			for j in np.arange(1,Nz-2):
				self.Hx[j] = self.Hx[j] - self.CM[j]*(self.Ey[j+1]-self.Ey[j])
				self.H[j,i] = self.Hx[j]

			self.Ey[Nz-1] = self.Ey[Nz-2]

			for j in np.arange(2,Nz-2):
				self.Ey[j] = self.CS[j]*self.Ey[j] - self.CE[j]*(self.Hx[j]-self.Hx[j-1])
				self.E[j,i] = self.Ey[j]

			self.I[:,i] = (self.E[:,i]**2+self.H[:,i]**2)


def plot_Intensity(Nz, I, it):
	plt.plot(np.arange(Nz), I[:,it])
	plt.show()


def test():
	fs = 5e9
	Z = 100
	NT = 1000
	Nz = 1000
	Nz_source = 100
	Nz_bound = 500
	param = [2, 1, 0.0003]

	solve = solve_maxwell(fs,Z,NT,Nz,Nz_source,Nz_bound, param)
	plot_Intensity(Nz, solve.I, 800)


if __name__ == "__main__":
    test()