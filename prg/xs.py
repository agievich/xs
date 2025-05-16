#******************************************************************************
# \file xs.py
# \project XS [XS-circuits into block ciphers]
# \brief Characteristics of XS-circuits
# \usage: xs path_to_the_circuit
# \author Sergey Agieivich
# \author Egor Lawrenov
# \withhelp Svetlana Mironovich
# \withhelp Nikita Lukyanov
# \created 2017.05.05
# \version 2025.05.16
# \license Public domain
#******************************************************************************

import functools as ft
import math
import numpy as np
import sys
import gf2

#******************************************************************************
# Class XS
#******************************************************************************

class XS:
	def __init__(self, a, B, c):
		self.n = len(a)
		self.a = np.array(a, dtype=int)
		self.B = np.array(B, dtype=int)
		self.c = np.array(c, dtype=int)

	@staticmethod
	def read_from_file(input_filename, sep):
		with open(input_filename, 'r') as inp:
			lines = inp.readlines()
			lines = [line.strip() for line in lines\
            	if line.strip() and not line.startswith('#')]
			M = np.array([list(map(int, line.split(sep))) for line in lines],\
				dtype=bool)
		if M.shape[0] != M.shape[1] or M[-1, -1] != 0:
			raise IOError("Bad format of (a,B,c)")
		M = M.astype(int)
		a = M[:-1, -1]
		B = M[:-1, :-1]
		c = M[-1, :-1]
		return XS(a, B, c)

	def save_to_file(self, output_filename, sep):
		np.savetxt(fname=output_filename, X=self.M(), fmt='%d', delimiter=sep)

	def M(self):
		M = np.ndarray(shape=(self.n + 1, self.n + 1), dtype=int)
		M[:-1, -1] = self.a
		M[:-1, :-1] = self.B
		M[-1, :-1] = self.c
		M[self.n, self.n] = 0
		return M

	def aBc(self):
		return self.a, self.B, self.c

	def is_invertible(self):
		if gf2.det(self.B) == 0:
			return gf2.det(self.M()) == 1
		return gf2.dot(gf2.dot(self.c, gf2.inv(self.B)), self.a) == 0

	def get_type(self):
		assert(self.is_invertible())
		if gf2.det(self.B) == 1:
			return 1
		else:
			return 2

	def inv(self):
		assert(self.is_invertible())
		if self.get_type() == 1:
			B1 = gf2.inv(self.B)
			a1 = gf2.dot(B1, self.a)
			c1 = gf2.dot(self.c, B1)
		else:
			M = gf2.inv(self.M())
			a1 = M[:-1, -1]
			B1 = M[:-1, :-1]
			c1 = M[-1, :-1]
		return XS(a1, B1, c1)

	def dual(self):
		return XS(self.c, self.B.T, self.a)

	def C(self):
		m = v = self.c
		for i in range(1, self.n):
			v = gf2.dot(v, self.B)
			m = np.vstack((v, m))
		return m

	def is_transitive(self):
		return gf2.det(self.C()) == 1

	def A(self):
		m = v = self.a
		for i in range(1, self.n):
			v = gf2.dot(v, self.B.T)
			m = np.vstack((m, v))
		return m.T

	def is_weak2transitive(self):
		return gf2.det(self.A()) == 1

	def is_regular(self):
		return self.is_transitive() and self.is_weak2transitive()

	def get_lag(self):
		l = 1
		v = self.c
		while l <= self.n and gf2.dot(v, self.a) == 0:
			l = l + 1
			v = gf2.dot(v, self.B)
		return l

	def get_profile(self, len):
		profile = np.empty(len, dtype=int)
		v = self.c
		for i in range(len):
			profile[i] = gf2.dot(v, self.a)
			v = gf2.dot(v, self.B)
		return profile

	def is_strong_regular(self):
		if self.is_regular() == False:
			return False
		l = self.get_lag()
		if l == 1:
			return True
		Bl = self.B
		for i in range(1,l):
			Bl = gf2.dot(Bl, self.B)
		m = v = self.c
		for i in range(1, self.n):
			v = gf2.dot(v, Bl)
			m = np.vstack((m, v))
		return gf2.det(m) == 1

	def rho2(self):
		v = self.c
		gamma = np.zeros(self.n)
		for i in range (0, self.n):
			gamma[i] = gf2.dot(v, self.a)
			v = gf2.dot(v, self.B)
		ret = 0
		A1 = gf2.inv(self.A())
		for r in range(0, self.n):
			y = np.zeros(self.n)
			y[r] = 1
			y[r + 1:] = gamma[:self.n - 1 - r]
			y = gf2.dot(y, A1)
			for i in range (0, self.n):
				y = gf2.dot(y, self.B)
			t = 0
			while True:
				t = t + 1
				if gf2.dot(y, self.a) != gamma[self.n - r + t - 2]:
					break
				y = gf2.dot(y, self.B)
			if t > ret:
				ret = t
		return self.n + ret

	# 1st canonical form: c0 = (0,0,...,0,1)
	def CF1(self):
		B = self.B
		c0 = np.zeros(self.n, dtype=int)
		c0[-1] = 1
		# bring B to the Frobenius form
		P = self.A()
		B = gf2.dot(gf2.dot(gf2.inv(P), B), P)
		a = gf2.dot(gf2.inv(P), self.a)
		c = gf2.dot(self.c, P)
        # find P = P(c): P.B.P^{-1} = B and c.P^{-1} = c0
		M = gf2.eye(self.n)
		P = gf2.dot(c, M)
		b = B[:, -1]
		for i in range(self.n - 1, 0, -1):
			M = gf2.dot(B, M)
			M = gf2.add(M, b[i] * gf2.eye(self.n))
			P = np.vstack((gf2.dot(c, M), P))
		return XS(gf2.dot(P, a), B, c0)

	# 2nd canonical form: a0 = (1,0,0,...,0)^T
	def CF2(self):
		B = self.B
		a0 = np.zeros(self.n, dtype=int)
		a0[0] = 1
		# bring B to the Frobenius form
		P = self.A()
		B = gf2.dot(gf2.dot(gf2.inv(P), B), P)
		a = gf2.dot(gf2.inv(P), self.a)
		c = gf2.dot(self.c, P)
		A = XS(a, B, c).A()
        # use the facts: A^{-1}.B.A = B and A^{-1}.a = a0
		return XS(a0, B, gf2.dot(c, A))

	def is_dense(self):
		assert(self.is_regular())
		cf2 = circ.CF2()
		b = cf2.B[:, -1]
		c = cf2.c
		r = []
		for i in range(0, circ.n):
			if b[circ.n - 1 - i] == 1 or c[i] == 1:
				r.append(i + 1)
		assert(r)
		return ft.reduce(math.gcd, r) == 1

	def describe(self):
		# invertibility
		if self.is_invertible() != True:
			print ("    - invertible")
			return 
		print("    %d type" % self.get_type())
		# transitivity
		if self.is_transitive():
			print("    + transitivity")
		else:
			print("    - transitivity")
		if self.is_weak2transitive():
			print("    + weak 2-transitivity")
		else:
			print("    - weak 2-transitivity")
		# is regular?
		if self.is_regular():
			# lag
			print("    %d lag" % self.get_lag())
			# rho2
			print("    %d \\rho2" % self.rho2())
			# strong regularity
			if self.is_strong_regular():
				print("    + strong regularity")
			else:
				print("    - strong regularity")
			# profile
			print("    profile =", self.get_profile(4 * self.n))
			# is dense? 
			if self.is_dense():
				print("      + dense")
			else:
				print("      - dense")
			# CFs
			a, B, c = self.CF1().aBc()
			print("    CF.b =", B[:,-1].T)
			print("      CF1.a =", a, "CF1.c =", c)
			a, B, c = self.CF2().aBc()
			print("      CF2.a =", a, "CF2.c =", c)

if __name__ == '__main__':
	circ_filename = sys.argv[1]
	circ = XS.read_from_file(circ_filename, ' ')
	print("circuit = %s:" % circ_filename)
	circ.describe()
	if circ.is_invertible():
		print("  inv(circuit):")
		circ.inv().describe()
		print("  dual(circuit):")
		circ.dual().describe()
