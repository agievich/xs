#******************************************************************************
# \file gf2.py
# \project XS [XS-circuits into block ciphers]
# \brief Arithmetic over GF(2)
# \author Sergey Agieivich [agievich@{bsu.by|gmail.com}]
# \author Egor Lawrenov
# \withhelp Mark Dickinson [https://stackoverflow.com/q/56856378]
# \created 2020.05.08
# \version 2020.06.30
# \license Public domain
#******************************************************************************

import numpy as np

#******************************************************************************
# Matrices
#******************************************************************************

def det(a):
	return round(np.linalg.det(a)) % 2

def zeros(n):
	return np.zeros(n, dtype=int)

def inv(a):
	if a.shape[0] != a.shape[1]:
		raise TypeError
	d = np.linalg.det(a)
	if round(d) % 2 != 1:
		raise ZeroDivisionError
	a = np.round(d * np.linalg.inv(a)) % 2
	return a

def dot(a, b):
	return np.round(np.dot(a, b)) % 2

def add(a, b):
	return (a + b) % 2

def eye(n):
	return np.eye(n, dtype=int)

def rank(a):
	# represent rows as non-negative integers
	rows = []
	for i in range(0, a.shape[0]):
		row = 0
		for j in range(0, a.shape[1]):
			row = 2 * row + int(a[i, j])
		rows.append(row)
	# linear algebra
	r = 0
	while rows:
		pivot_row = rows.pop()
		if pivot_row:
			r += 1
			lsb = pivot_row & -pivot_row
			for i, row in enumerate(rows):
				if row & lsb:
					rows[i] = row ^ pivot_row
	return r
