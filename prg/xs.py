#******************************************************************************
# \file xs.py
# \project XS [XS-circuits into block ciphers]
# \brief Characteristics of XS-circuits
# \author Sergey Agieivich [agievich@{bsu.by|gmail.com}]
# \withhelp Svetlana Mironovich
# \created 2017.05.05
# \version 2018.06.08
# \license Public domain.
#******************************************************************************

import sys
import numpy as np

class XS:
	def __init__(self, a, B, c):
		self.a, self.B, self.c = np.array(a), np.array(B), np.array(c)
		self.n = len(a)

	@staticmethod
	def read_from_file(input_filename, sep):
		with open(input_filename, 'r') as inp:
			lines = inp.readlines()
			lines = [line.strip() for line in lines\
            	if line.strip() and not line.startswith('#')]
			M = np.array([list(map(int, line.split(sep))) for line in lines],\
				dtype = bool)
		if M.shape[0] != M.shape[1] or M[-1, -1] != 0:
			raise IOError("Bad format of (a,B,c)")
		M = M.astype(float)
		a = M[:-1, -1]
		B = M[:-1, :-1]
		c = M[-1, :-1]
		return XS(a, B, c)

	@staticmethod
	def inv2(m):
		if m.shape[0] != m.shape[1]:
			raise TypeError
		d = np.linalg.det(m)
		if round(d) % 2 != 1:
			raise ZeroDivisionError
		m = np.round(d * np.linalg.inv(m)) % 2
		return m

	def is_invertible(self):
		if np.linalg.matrix_rank(self.B) == self.n:
			return round(np.dot(np.dot(self.c, XS.inv2(self.B)), self.a)) % 2 == 0
		elif np.linalg.matrix_rank(self.B) == self.n - 1:
			Bc = np.vstack((self.B, self.c))
			Ba = np.vstack((self.B.T, self.a))
			return np.linalg.matrix_rank(Bc) == self.n and\
				np.linalg.matrix_rank(Ba) == self.n 
			return True
		else:
			return False

	def get_type(self):
		if np.linalg.matrix_rank(self.B) == self.n:
			return 1
		else:
			return 2

	def inv(self):
		if np.linalg.matrix_rank(self.B) == self.n:
			B1 = XS.inv2(self.B)
			a1 = np.round(np.dot(B1, self.a)) % 2
			c1 = np.round(np.dot(self.c, B1)) % 2
		else:
			M = np.ndarray(shape = (self.n + 1, self.n + 1))
			M[:-1, -1] = self.a
			M[:-1, :-1] = self.B
			M[-1, :-1] = self.c
			M = XS.inv2(M)
			a1 = M[:-1, -1]
			B1 = M[:-1, :-1]
			c1 = M[-1, :-1]
		return XS(a1, B1, c1)

	def dual(self):
		return XS(self.c, self.B.T, self.a)

	def C(self):
		m = v = self.c
		for i in range(1, self.n):
			v = np.round(np.dot(v, self.B)) % 2
			m = np.vstack((v, m))
		return m

	def is_transitive(self):
		return np.linalg.matrix_rank(self.C()) == self.n

	def A(self):
		m = v = self.a
		for i in range(1, self.n):
			v = np.round(np.dot(v, self.B.T)) % 2
			m = np.vstack((m, v))
		return m.T

	def is_weak2transitive(self):
		return np.linalg.matrix_rank(self.A()) == self.n

	def is_regular(self):
		return self.is_transitive() and self.is_weak2transitive()

	def get_lag(self):
		l = 1
		v = self.c
		while l <= self.n and round(np.dot(v, self.a)) % 2 == 0:
			l = l + 1
			v = np.dot(v, self.B)
		return l

	def is_strong_regular(self):
		if self.is_regular() == False:
			return False
		l = self.get_lag()
		if l == 1:
			return True
		Bl = self.B
		for i in range(1,l):
			Bl = np.round(np.dot(Bl, self.B)) % 2
		m = v = self.c
		for i in range(1, self.n):
			v = np.round(np.dot(v, Bl)) % 2
			m = np.vstack((m, v))
		return np.linalg.matrix_rank(m) == self.n

	def rho2(self):
		v = self.c
		gamma = np.zeros(self.n)
		for i in range (0, self.n):
			gamma[i] = round(np.dot(v, self.a)) % 2
			v = np.round(np.dot(v, self.B)) % 2
		ret = 0
		A1 = XS.inv2(self.A())
		for r in range(0, self.n):
			y = np.zeros(self.n)
			y[r] = 1
			y[r + 1:] = gamma[:self.n - 1 - r]
			y = np.round(np.dot(y, A1)) % 2
			for i in range (0, self.n):
				y = np.round(np.dot(y, self.B)) % 2
			t = 0
			while True:
				t = t + 1
				if round(np.dot(y, self.a)) % 2 != gamma[self.n - r + t - 2]:
					break
				y = np.round(np.dot(y, self.B)) % 2
			if t > ret:
				ret = t
		return self.n + ret

	def describe(self):
		if self.is_invertible() != True:
			print ("    - invertible")
			return 
		print("    %d type" % circ.get_type())
		if self.is_transitive():
			print("    + transitivity")
		else:
			print("    - transitivity")
		if self.is_weak2transitive():
			print("    + weak 2-transitivity")
		else:
			print("    - weak 2-transitivity")

		if self.is_regular():
			print("    %d lag" % self.get_lag())
			print("    %d \\rho2" % circ.rho2())
			if self.is_strong_regular():
				print("    + strong regularity")
			else:
				print("    - strong regularity")

if __name__ == '__main__':
	circ_filename = sys.argv[1]
	circ = XS.read_from_file(circ_filename, ' ')
	print("circuit = %s:" % circ_filename)
	circ.describe()
	print("  inv(circuit):")
	circ.inv().describe()
	print("  dual(circuit):")
	circ.dual().describe()
