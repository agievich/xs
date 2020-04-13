import sys
import numpy as np
from sympy import Matrix

def det2(m):
	return round(np.linalg.det(m)) % 2

def int_to_bitlist(num, digits):
    return np.array([int(bool(num & (1<<n))) for n in range(digits)])

def inv2(m):
	if m.shape[0] != m.shape[1]:
		raise TypeError
	d = np.linalg.det(m)
	if round(d) % 2 != 1:
		raise ZeroDivisionError
	m = np.round(d * np.linalg.inv(m)) % 2
	return m

def dot2(u, v):
	return np.round(np.dot(u, v)) % 2

def frobenius_cell(B):
	B_ = Matrix(B)
	n = len(B)
	# вычисление характеристического многочлена матрицы используя пакет sympy
	B_ = Matrix(B)
	b = np.array(B_.charpoly().as_list()[::-1])[:-1].reshape((n,1))%2
	for i in range(1, 2**n):
		p1 = int_to_bitlist(i, n).reshape(n,1)
		P = v = p1
		for i in range(n-1):
			v = np.dot(B, v)
			P = np.hstack((P,v))
		if(det2(P) == 1):
			frob_B = (dot2(dot2(inv2(P), B), P)).astype(int)
			return frob_B, P