from __future__ import annotations

import itertools as it
from functools import cached_property, lru_cache
from enum import Enum

import numpy as np
from scipy.constants import epsilon_0, elementary_charge

class Direction(Enum):
	X = 0
	Y = 1
	Z = 2

class Const:
	INV_E0 = elementary_charge * 1E9 / (4*np.pi*epsilon_0)

class Grid:
	def __init__(self, Nprime, a):
		self.Nprime = Nprime
		self.a = a

		self.lattice = { \
			i:UnitCell(self.index2coord(*i), self.a) \
			for i in it.product(*[range(2*Nprime) for _ in range(3)]) \
		}

	@cached_property
	def size(self):
		return len(list(self.molecules()))
	
	def index2coord(self, *indices):
		return [self.a * (i - self.Nprime) for i in indices]

	def molecules(self):
		for lat in self.lattice.values():
			for mol in lat.molecules:
				yield mol

	def get_charged_molecule(self):
		for i in self.molecules():
			if i.charge != 0:
				return i

	def flatten(self):
		charged = self.get_charged_molecule()
		return np.array([i.E0(a, charged) if i != charged else 0 for a in Direction for i in self.molecules()])

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
	
	@lru_cache
	def E0(self, dir, charged):
		return -Const.INV_E0 * charged.charge * self.diff(charged)[dir.value] / self.dist_to(charged)**3

	@lru_cache
	def T(self, b, j, c):
		vec = self.diff(j)
		dist = self.dist_to(j)

		term = dist**2 if b == c else 0

		return Const.INV_E0 * (3*vec[b.value]*vec[c.value] - term) / dist**5

	@lru_cache
	def diff(self, mol: Molecule):
		return mol.pos - self.pos

	@lru_cache
	def dist_to(self, mol: Molecule):
		return np.linalg.norm(self.diff(mol))