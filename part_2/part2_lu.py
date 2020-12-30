#!/usr/bin/python3
"""
Copyright © 2020-2021 Kevin Vonk

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

import numpy as np

class SparseTridiagonal:
	l = None
	u = None

	def decompose(self, *args):
		"""
			input sparse arrays

			2 args: diag and off diag
			3 args: left off diag, diag, right off diag

			returns sparse l, u
		"""
		if len(args) == 2:
			left = args[1]
			diag = args[0]
			right = args[1]
		elif len(args) == 3:
			left = args[0]
			diag = args[1]
			right = args[2]
		else:
			raise TypeError(f"Invalid amount of arguments provided, expected 2 or 3, got {len(args)}")

		n = len(diag)
		u = np.zeros((2, n))	# 0: diag, 1: off-diag
		l = np.zeros(n - 1)

		u[0, 0] = diag[0]

		for i in range(1, n):
			l[i - 1] = left[i - 1] / u[0, i - 1]
			u[1, i] = right[i - 1]
			u[0, i] = diag[i] - l[i - 1]*u[1, i]

		self.l, self.u = l, u
		return l, u

	def forward(self, b):
		if self.l is None or self.u is None:
			raise ValueError("No LU decomposition known. Execute `decompose()` first")

		n = len(b)

		if n != len(self.l) + 1:
			raise IndexError("b should have the length of the dimensionality from the decomposed array")

		y = np.zeros(n)
		y[0] = b[0]
		
		for i in range(1, n):
			y[i] = b[i] - self.l[i - 1] * y[i - 1]

		return y

	def backward(self, y):
		if self.l is None or self.u is None:
			raise ValueError("No LU decomposition known. Execute `decompose()` first")

		n = len(y)

		if n != len(self.l) + 1:
			raise IndexError("y should have the length of the dimensionality from the decomposed array")

		x = np.zeros(n)
		x[-1] = y[-1] / self.u[0, -1]

		for i in range(n - 1, 0, -1):
			x[i - 1] = (y[i - 1] - self.u[1, i] * x[i]) / self.u[0, i - 1]

		return x

	def solve(self, b):
		return self.backward(self.forward(b))



if __name__ == "__main__":
	print("This file can only be imported")