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

if __name__ == "__main__":
	print("This file can only be imported")