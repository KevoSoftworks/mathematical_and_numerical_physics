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

import re
import numpy as np

from part2_lu import SparseTridiagonal

def to_latex(arr, d=2, type="bmatrix"):
	string = str(arr.round(decimals=d))
	string = re.sub(r"^\s*\[*\s*", "", string)
	string = re.sub(r"\s*\]*\s*$", "", string)
	string = re.sub(r"\]\s*\[\s*", r"\\\\", string)
	string = re.sub(r"\s+", r"&", string)

	return r"\begin{" + type + "}" + string + r"\end{" + type + "}"



def eigen_power(A, b0, TOL=1E-8, ret_cycles=False):
	MAX_ITER = 200

	b0 = b0 / np.linalg.norm(b0)
	b1 = A @ b0
	R0 = b0.T @ b1

	epsilon = 999
	iter = 0

	while epsilon >= TOL and iter <= MAX_ITER:
		b0 = b1 / np.linalg.norm(b1)
		b1 = A @ b0
		R1 = b0.T @ b1

		epsilon = np.abs(R1 - R0)
		R0 = R1
		iter += 1

	if ret_cycles:
		return R0, b0, iter
	else:
		return R0, b0

def eigen_inv_power(A, *args, **kwargs):
	return eigen_power(np.linalg.inv(A), *args, **kwargs)

def eigen_shift_inv_power(A, b0, shift, TOL=1E-8, ret_cycles=False):
	MAX_ITER = 200

	mat = A - shift * np.diag(np.ones(len(A)))

	b0 = b0 / np.linalg.norm(b0)
	b1 = np.linalg.solve(mat, b0)
	S0 = b0.T @ b1

	epsilon = 999
	iter = 0

	while epsilon >= TOL and iter <= MAX_ITER:
		b0 = b1 / np.linalg.norm(b1)
		b1 = np.linalg.solve(mat, b0)
		S1 = b0.T @ b1

		epsilon = np.abs(1/S1 - 1/S0)
		S0 = S1
		iter += 1

	lambda_ = 1/S0 + shift

	if ret_cycles:
		return lambda_, b0, iter
	else:
		return lambda_, b0

def sparse_sipi(diag, off_diag, b0, shift, TOL=1E-8):
	MAX_ITER = 200

	mat = SparseTridiagonal()
	mat.decompose(diag - shift, off_diag)

	b0 = b0 / np.linalg.norm(b0)
	b1 = mat.solve(b0)
	S0 = b0.T @ b1

	epsilon = 999
	iter = 0

	while epsilon >= TOL and iter <= MAX_ITER:
		b0 = b1 / np.linalg.norm(b1)
		b1 = mat.solve(b0)
		S1 = b0.T @ b1

		epsilon = np.abs(1/S1 - 1/S0)
		S0 = S1
		iter += 1
		lambda_ = 1/S0 + shift

		yield lambda_, b0

def sparse_double_sipi(diag, off_diag, b0, bp0, shift, TOL=1E-8):
	MAX_ITER = 200

	mat = SparseTridiagonal()
	mat.decompose(diag - shift, off_diag)

	b0 = b0 / np.linalg.norm(b0)
	b1 = mat.solve(b0)
	S0 = b0.T @ b1
	b1 = b1 / np.linalg.norm(b1)

	bp0 = bp0 / np.linalg.norm(bp0)
	bp1 = mat.solve(bp0)
	bp1 = bp1 - b1*(b1.T @ bp1)
	Sp0 = bp0.T @ bp1
	bp1 = bp1 / np.linalg.norm(bp1)

	epsilon = False
	iter = 0

	while not epsilon and iter <= MAX_ITER:
		b0 = b1
		b1 = mat.solve(b0)
		S1 = b0.T @ b1
		b1 = b1 / np.linalg.norm(b1)

		bp0 = bp1
		bp1 = mat.solve(bp0)
		bp1 = bp1 - b1*(b1.T @ bp1)
		Sp1 = bp0.T @ bp1
		bp1 = bp1 / np.linalg.norm(bp1)

		epsilon = (np.abs(1/S1 - 1/S0) < TOL) and (np.abs(1/Sp1 - 1/Sp0) < TOL)
		S0 = S1
		Sp0 = Sp1
		iter += 1
		lambda_ = [1/S0 + shift, 1/Sp0 + shift]

		yield lambda_, [b0, bp0]


def wf_norm(wf, dx, t="norm"):
	if t == "integral":
		return wf / np.trapz(wf**2, dx=dx)
	elif t == "norm":
		return wf / (np.linalg.norm(wf) * np.sqrt(dx))
	else:
		raise ValueError(f"Type {t} is not supported")

def sparse_scf(h, f, *args, TOL_SCF=1E-6, **kwargs):
	MAX_ITER = 500
	iter = 0
	lambda_ = 999

	f = f[1:-1]

	n = np.abs(f)**2
	epsilon = False

	diag = 1/h**2 * (2 - 4*h**2 * n)
	off_diag = -1 * 1/h**2 * np.ones(len(diag) - 1)

	while not epsilon and iter <= MAX_ITER:
		diag = 1/h**2 * (2 - 4*h**2 * n)

		for val, f in sparse_sipi(diag, off_diag, f, *args, **kwargs):
			continue

		f = wf_norm([0, *f, 0], h)
		f = f[1:-1]

		n = np.abs(f)**2
		epsilon = np.abs(val - lambda_) < TOL_SCF
		lambda_ = val

		yield lambda_, f

		iter += 1

def bipolaron(grid, step, guess, spacing, beta, gamma, *args, TOL_SCF=1E-6, **kwargs):
	_W = lambda f: beta * np.abs(f)**2
	_S = lambda f1, f2: np.trapz(f1 * f2, dx=step)
	_n = lambda f1, f2: (f1**2 + f2**2 + 2*_S(f1, f2)*f1*f2) / (1 + _S(f1, f2)**2)
	_ls = lambda l, f1, f2: 2*l - beta * np.trapz(np.abs(f1)**2 * np.abs(f2)**2, dx=step)

	MAX_ITER = 200
	iter = 0
	lambda_ = 999
	epsilon = False

	f = guess
	f1, f2 = _create_image(grid, step, f, spacing)
	n = _n(f1, f2)

	diag = 1/step**2 * (2 - 4*step**2 * (n - _W(f2)))
	off_diag = -1 * 1/step**2 * np.ones(len(diag) - 1)

	while not epsilon and iter <= MAX_ITER:
		n = gamma * _n(f1, f2) + (1 - gamma) * n
		diag = 1/step**2 * (2 - 4*step**2 * (n - _W(f2)))

		for val, f in sparse_sipi(diag, off_diag, f, *args, **kwargs):
			continue

		f = wf_norm(f, step)
		f1, f2 = _create_image(grid, step, f, spacing)
		nlambda_ = _ls(val, f1, f2)

		epsilon = np.abs(nlambda_ - lambda_) < TOL_SCF
		lambda_ = nlambda_

		yield lambda_, f1, f2

		iter += 1

def _create_image(y, h, f, Y):
	ymax = y[np.argmax(f)]

	# Get middle value based on whether the element count in y is odd or even
	if y.size % 2 == 0:
		# Even array
		ymid = y[y.size // 2]
	else:
		# Odd array
		_i = y.size // 2
		ymid = (y[_i] + y[_i + 1]) / 2

	yshift = [ymid - ymax - Y/2, ymid - ymax + Y/2]
	nshift = [int(i // h) for i in yshift]

	f1, f2 = [np.roll(f, i) for i in nshift]

	return f1, f2

if __name__ == "__main__":
	print("This file can only be imported")