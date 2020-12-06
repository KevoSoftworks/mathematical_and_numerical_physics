#!/usr/bin/python3
"""
Copyright © 2020 Kevin Vonk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the “Software”), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from functools import cached_property
import math
import copy

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve

class TridiagonalCoefficient:
	def __init__(self, size, bc=(0, 0)):
		self.coff = np.zeros(size)
		self.bc = bc

	def __getitem__(self, index):
		if index < 0:
			return self.bc[0]

		elif index >= len(self.coff):
			return self.bc[1]

		else:
			return self.coff[index]

	def __setitem__(self, index, value):
		self.coff[index] = value

	def reset(self):
		self.coff = np.zeros(len(self.coff))



class GeneralSolver1D:
	def __init__(self, xbounds, tbounds, nx, nt, ic, M=1, bc=(0, 0), bc_inner=True, nx_as_interval=False):
		if not callable(ic):
			raise TypeError("Initial condition must be a function or lambda expression with arguments x and t")

		if len(bc) % 2 != 0:
			raise ValueError("Boundary conditions must be an even number of elements")

		# Compensate for differences in definition
		if nx_as_interval:
			nx += 1

		self.x, self.dx = np.linspace(*xbounds, num=nx, retstep=True)
		self.t, self.dt = np.linspace(*tbounds, num=nt, retstep=True)

		self.solution = ic(self.x, self.t[0])

		if bc_inner:
			if len(bc) > 2:
				raise NotImplementedError("Inner boundary conditions not supported for more than 2 points")

			self.solution[0] = bc[0]
			self.solution[-1] = bc[1]
		
		self.M = M
		self.bc = bc
		self.bc_inner = bc_inner

		# Set the starting time index. Note that t=0 provides the initial conditions
		self.cur_t = 1

	def _keycheck(self, key):
		"""
			Verify the validity of the provided indexing key
		"""

		# Compute the length of the key, considering it can be an integer or a tuple
		keylen = None
		if isinstance(key, int):
			keylen = 1
		else:
			keylen = len(key)

		# If the key length does not match the expected dimensionality, raise an error
		if(keylen != 1):
			raise IndexError(f"Given dimension {keylen} is invalid, expected 1")

		# Check if slicing was used for indexing. If so, raise an error that it is not implemented
		if isinstance(key, slice):
			raise NotImplementedError(f"Index slicing not implemented")

		# Return success
		return True

	def __setitem__(self, key, value):
		self._keycheck(key)

		if key < 0 or key >= len(self.x):
			raise IndexError("Boundary conditions cannot be dynamically changed")

		self.solution[key] = value

		if self.bc_inner and (key == 0 or key == len(self.x) - 1):
			self.forceBC()
			pass

	def __getitem__(self, key):
		self._keycheck(key)
		
		# Force BC outside the range
		if key < 0 or key >= len(self.x):
			if key > 0:
				key %= len(self.x)

			index = int(len(self.bc)/2) + key
			return self.bc[index]
		else:
			return self.solution[key]

	
	@cached_property
	def alpha(self):
		# Computation of dx and dt is rather hacky, this means that grid spacing has to be constant. Too bad!
		return self.M * self.dt / (self.dx**2)

	def solve(self):
		raise NotImplementedError("GeneralSolver1D does not implement a solution strategy")

	def fullsolve(self):
		for _ in self.solve():
			pass

		return self.solution

	def forceBC(self):
		if self.bc_inner:
			self.solution[0] = self.bc[0]
			self.solution[-1] = self.bc[1]



class GeneralSolver2D:
	def __init__(self, xybounds, tbounds, nxy, nt, ic, M=1, bc=lambda x, y: 0*x + 0*y, nxy_as_interval=True):
		if not callable(ic):
			raise TypeError("Initial condition must be a function or lambda expression with arguments x, y and t")

		if not callable(bc):
			raise TypeError("Boundary condition must be a function or lambda expression with arguments x and y")

		# Compensate for differences in definition
		if nxy_as_interval:
			nxy = [i + 1 for i in nxy]

		self.x, self.dx = np.linspace(*xybounds[0], num=nxy[0], retstep=True)
		self.y, self.dy = np.linspace(*xybounds[1], num=nxy[1], retstep=True)
		self.t, self.dt = np.linspace(*tbounds, num=nt, retstep=True)

		self.meshx, self.meshy = np.meshgrid(self.x, self.y, indexing="ij")

		self.solution = ic(self.meshx, self.meshy, self.t[0])
		self.bc = bc
		self.forceBC()
		
		self.M = M
		self.bc = bc

		# Set the starting time index. Note that t=0 provides the initial conditions
		self.cur_t = 1

	def _keycheck(self, key):
		"""
			Verify the validity of the provided indexing key
		"""
		# We are expecting a tuple, so we must call as obj[a, b]
		if not isinstance(key, tuple):
			raise KeyError("Key must be a tuple")

		# Since we are working with a 2D space, we expect two elements in our tuple
		if len(key) != 2:
			raise IndexError(f"Expecting 2D index, got dimension {len(key)}")

		# Return success
		return True

	def __setitem__(self, key, value):
		self._keycheck(key)

		if key < (0, 0) or key >= (len(self.x), len(self.y)):
			raise IndexError(f"Index {key} out of bounds for x={(0, len(self.x)-1)}, y={(0, len(self.y)-1)}")

		self.solution[key] = value

		if key[0] == 0 or key[1] == 0 or key[0] == len(self.x) - 1 or key[1] == len(self.y) - 1:
			self.forceBC()

	def __getitem__(self, key):
		self._keycheck(key)

		if key < (0, 0) or key >= (len(self.x), len(self.y)):
			raise IndexError(f"Index {key} out of bounds for x={(0, len(self.x)-1)}, y={(0, len(self.y)-1)}")
		
		return self.solution[key]

	def solve(self):
		raise NotImplementedError("GeneralSolver2D does not implement a solution strategy")

	def fullsolve(self):
		for _ in self.solve():
			pass

		return self.solution

	def forceBC(self):
		# This can probably be done much neater, but I don't know how and I can't be bothered. Oh well!
		self.solution[:, 0] = self.bc(self.x, 0)
		self.solution[:, -1] = self.bc(self.x, self.y[-1])
		self.solution[0, :] = self.bc(0, self.y)
		self.solution[-1, :] = self.bc(self.x[-1], self.y)



class EulerForward1D(GeneralSolver1D):
	@staticmethod
	def stable_time(dx, m):
		return dx**2/(2*m)

	@classmethod
	def stable_time_steps(cls, xbounds, tbounds, nx, M, nx_as_interval=False):
		# TODO: This method should probably be generalised to GeneralSolver1D, since it is used multiple times
		# and only depends on the implementation of stable_time(), which can be implemented in each class
		# separately.

		if nx_as_interval:
			nx += 1

		x = xbounds[1] - xbounds[0]
		t = tbounds[1] - tbounds[0]

		# Compute dx, compensate for the fact that we need nx - 1 steps to go from x[0] to x[1]
		dx = x / (nx - 1)
		dt = cls.stable_time(dx, M)

		# Compute the time steps
		nt = t / dt + 1

		return math.ceil(nt)

	def solve(self):
		if self.alpha > 0.5:
			raise Exception(f"Von Neumann stability violated, got {self.alpha}, should be <= 0.5")

		while self.cur_t < len(self.t):
			self.solution = [\
				(1 - 2*self.alpha) * self[i] + self.alpha * (self[i + 1] + self[i - 1]) \
				for i in range(len(self.x))
			]

			self.forceBC()
			self.cur_t += 1

			yield self.solution



class EulerForward2D(GeneralSolver2D):
	@staticmethod
	def stable_time(dx, dy, m):
		return dx**2 * dy**2 / (2*m*(dx**2 + dy**2))

	@classmethod
	def stable_time_steps(cls, xybounds, tbounds, nxy, M, nxy_as_interval=False):
		if nxy_as_interval:
			nxy = [i + 1 for i in nxy]

		x = xybounds[0][1] - xybounds[0][0]
		y = xybounds[1][1] - xybounds[1][0]
		t = tbounds[1] - tbounds[0]

		dx = x / (nxy[0] - 1)
		dy = y / (nxy[1] - 1)

		dt = cls.stable_time(dx, dy, M)

		nt = t / dt + 1

		return math.ceil(nt)

	def solve(self):
		while self.cur_t < len(self.t):
			prev = copy.deepcopy(self.solution)

			for i in range(1, len(self.x) - 1):
				for j in range(1, len(self.y) - 1):
					self[i, j] = self.M * self.dt / (self.dx**2 * self.dy**2) * ( \
						self.dy**2 * (prev[i+1, j] - 2*prev[i, j] + prev[i-1, j]) + \
						self.dx**2 * (prev[i, j+1] - 2*prev[i, j] + prev[i, j-1]) \
					) + prev[i, j]

			self.forceBC()

			yield self.solution

			self.cur_t += 1



class DuFortFrankel1D(GeneralSolver1D):
	@staticmethod
	def stable_time(dx):
		return 1E-2 * dx

	@classmethod
	def stable_time_steps(cls, xbounds, tbounds, nx, nx_as_interval=False):
		# TODO: This method should probably be generalised to GeneralSolver1D, since it is used multiple times
		# and only depends on the implementation of stable_time(), which can be implemented in each class
		# separately.

		if nx_as_interval:
			nx += 1

		x = xbounds[1] - xbounds[0]
		t = tbounds[1] - tbounds[0]

		# Compute dx, compensate for the fact that we need nx - 1 steps to go from x[0] to x[1]
		dx = x / (nx - 1)
		dt = cls.stable_time(dx)

		# Compute the time steps
		nt = t / dt + 1

		return math.ceil(nt)


	def solve(self):
		c = 1 / (1 + 2*self.alpha)

		# In the inital run, assume that the past point is the same as the initial condition
		past = list(self.solution)

		while self.cur_t < len(self.t):
			new_past = list(self.solution)
			self.solution = [\
				c*(1 - 2*self.alpha) * past[i] + 2*self.alpha*c * (self[i + 1] + self[i - 1]) \
				for i in range(len(self.x))
			]
			past = new_past

			self.forceBC()
			self.cur_t += 1

			yield self.solution



class EulerBackward1D(GeneralSolver1D):
	def __setitem__(self, key, value):
		# We override the __setitem__ function because the self.forceBC() function provides
		# invalid results when updating the solution element-by-element. We need to call
		# self.forceBC() in solve() ourselves

		self._keycheck(key)

		if key < 0 or key >= len(self.x):
			raise IndexError("Boundary conditions cannot be dynamically changed")

		self.solution[key] = value

	@cached_property
	def A_raw(self):
		diag = (1 + 2*self.alpha) * np.ones(len(self.x))
		off_diag = -self.alpha * np.ones(len(self.x))

		return diag, off_diag

	@cached_property
	def A(self):
		diag, off_diag = self.A_raw
		
		return np.diag(diag, 0) + np.diag(off_diag, 1)[:-1, :-1] + np.diag(off_diag, -1)[:-1, :-1]

	def solve(self):
		diag, off_diag = self.A_raw
		off_diag = -1 * copy.deepcopy(off_diag)	# Since we solve for these to be negative specifically

		e = TridiagonalCoefficient(len(self.x))
		f = TridiagonalCoefficient(len(self.x))

		while self.cur_t < len(self.t):
			e.reset()
			f.reset()

			# Forward sweep
			for i in range(len(self.x)):
				e[i] = off_diag[i] / (diag[i] - off_diag[i]*e[i-1])
				f[i] = (self[i] + off_diag[i]*f[i-1]) / (diag[i] - off_diag[i]*e[i-1])

			# Back substitution
			for i in range(len(self.x) - 1, -1, -1):
				self[i] = f[i] + e[i] * self[i+1]

			self.forceBC()

			yield self.solution

			self.cur_t += 1



class CrankNicolson1D(EulerBackward1D):
	@cached_property
	def A_raw(self):
		diag = 2*(1 + self.alpha) * np.ones(len(self.x))
		off_diag = -self.alpha * np.ones(len(self.x))

		return diag, off_diag

	@cached_property
	def B_raw(self):
		diag = 2*(1 - self.alpha) * np.ones(len(self.x))
		off_diag = self.alpha * np.ones(len(self.x))

		return diag, off_diag

	@cached_property
	def B(self):
		diag, off_diag = self.B_raw
		
		return np.diag(diag, 0) + np.diag(off_diag, 1)[:-1, :-1] + np.diag(off_diag, -1)[:-1, :-1]

	def solve(self):
		diag, off_diag = self.A_raw
		off_diag = -1 * copy.deepcopy(off_diag)	# Since we solve for these to be negative specifically

		e = TridiagonalCoefficient(len(self.x))
		f = TridiagonalCoefficient(len(self.x))

		while self.cur_t < len(self.t):
			e.reset()
			f.reset()

			# Compute known side
			self.solution = self.B @ self.solution

			# Forward sweep
			for i in range(len(self.x)):
				e[i] = off_diag[i] / (diag[i] - off_diag[i]*e[i-1])
				f[i] = (self[i] + off_diag[i]*f[i-1]) / (diag[i] - off_diag[i]*e[i-1])

			# Back substitution
			for i in range(len(self.x) - 1, -1, -1):
				self[i] = f[i] + e[i] * self[i+1]

			self.forceBC()

			yield self.solution

			self.cur_t += 1


if __name__ == "__main__":
	print("This file can only be imported as a module.")