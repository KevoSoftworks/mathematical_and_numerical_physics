from __future__ import annotations

import itertools as it
from enum import Enum

import numpy as np
from scipy.constants import epsilon_0

class Direction(Enum):
	X = 0
	Y = 1
	Z = 2

class Const:
	INV_E0 = 1E9 / (4*np.pi*epsilon_0)

class Grid:
	def __init__(self, Nprime, a):
		self.Nprime = Nprime
		self.a = a

		self.lattice = { \
			i:UnitCell(self.index2coord(*i), self.a) \
			for i in it.product(*[range(2*Nprime) for _ in range(3)]) \
		}
	
	def index2coord(self, *indices):
		return [self.a * (i - self.Nprime) for i in indices]

	def molecules(self):
		for lat in self.lattice.values():
			for mol in lat.molecules:
				yield mol

	def polarisation_energy(self):
		res = 0

		for i in self.molecules():
			for a in Direction:
				e = [i.E0(d, self.molecules()) for d in Direction]
				res += e[a.value] * i.dipole(e)[a.value]

		return -0.5*res

class UnitCell:
	def __init__(self, origin, a):
		self.origin = origin
		self.a = a

		self._generate()

	def _generate(self):
		pos = [
			(self.origin[0], self.origin[1], self.origin[2]),
			(self.origin[0] + self.a / 2, self.origin[1] + self.a / 2, self.origin[2]),
			(self.origin[0] + self.a / 2, self.origin[1], self.origin[2] + self.a / 2),
			(self.origin[0], self.origin[1] + self.a / 2, self.origin[2] + self.a / 2)
		]
		self.molecules = [Molecule(p) for p in pos]

class Molecule:
	def __init__(self, pos):
		self.pos = np.array(pos)
		self.charge = 0

		# TODO: not hardcode this?
		if np.count_nonzero(self.pos) == 0:
			self.charge = 1

	def alpha(self, a: Direction, b: Direction):
		POL = 0.08	# TODO: Not hardcode this?

		if a == b:
			return POL
		return 0

	def dipole(self, E):
		return [np.sum([self.alpha(a, b) * E[b.value] for b in Direction]) for a in Direction]
	
	def E0(self, dir, mol):
		return -Const.INV_E0 \
			* np.sum([k.charge * (self.diff(k))[dir.value] / self.dist_to(k)**3 for k in mol if k.charge != 0])

	def Epol(self, dir, mol):
		pass
		#np.sum(\
		#	[np.sum( \
		#		[self.T(dir, j, c)*j.dipole(...) for c in Direction] \ #TODO
		#	) for j in mol if j not self] \
		#)
	
	def T(self, b, j, c):
		vec = self.diff(j)
		dist = self.dist_to(j)

		term = dist**2 if b == c else 0

		return Const.INV_E0 * (3*vec[b.value]*vec[c.value] - term) / dist**5

	def diff(self, mol: Molecule):
		return mol.pos - self.pos

	def dist_to(self, mol: Molecule):
		return np.linalg.norm(self.diff(mol))