#******************************************************************************
# \file xs.py
# \project XS [XS-circuits into block ciphers]
# \brief Characteristics of XS-circuits
# \author Sergey Agieivich [agievich@{bsu.by|gmail.com}]
# \withhelp Svetlana Mironovich
# \created 2017.05.05
# \version 2018.06.21
# \license Public domain.
#******************************************************************************

import sys
import numpy as np

class XS:
	def __init__(self, a, B, c):
		self.a, self.B, self.c = np.array(a, dtype=int), np.array(B, dtype=int), np.array(c, dtype=int)
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

	def save_to_file(self, output_filename, sep):
		np.savetxt(fname = output_filename, X = self.M(), fmt='%d', delimiter=sep)

	@staticmethod
	def det2(m):
		return round(np.linalg.det(m)) % 2

	@staticmethod
	def inv2(m):
		if m.shape[0] != m.shape[1]:
			raise TypeError
		d = np.linalg.det(m)
		if round(d) % 2 != 1:
			raise ZeroDivisionError
		m = np.round(d * np.linalg.inv(m)) % 2
		return m

	@staticmethod
	def dot2(u, v):
		return np.round(np.dot(u, v)) % 2

	def M(self):
		M = np.ndarray(shape = (self.n + 1, self.n + 1), dtype=int)
		M[:-1, -1] = self.a
		M[:-1, :-1] = self.B
		M[-1, :-1] = self.c
		M[self.n, self.n] = 0
		return M

	def is_invertible(self):
		if XS.det2(self.B) == 0:
			return XS.det2(self.M()) == 1
		return XS.dot2(XS.dot2(self.c, XS.inv2(self.B)), self.a) == 0

	def get_type(self):
		assert(self.is_invertible())
		if XS.det2(self.B) == 1:
			return 1
		else:
			return 2

	def inv(self):
		assert(self.is_invertible())
		if self.get_type() == 1:
			B1 = XS.inv2(self.B)
			a1 = XS.dot2(B1, self.a)
			c1 = XS.dot2(self.c, B1)
		else:
			M = XS.inv2(self.M())
			a1 = M[:-1, -1]
			B1 = M[:-1, :-1]
			c1 = M[-1, :-1]
		return XS(a1, B1, c1)

	def dual(self):
		return XS(self.c, self.B.T, self.a)

	def C(self):
		m = v = self.c
		for i in range(1, self.n):
			v = XS.dot2(v, self.B)
			m = np.vstack((v, m))
		return m

	def is_transitive(self):
		return XS.det2(self.C()) == 1

	def A(self):
		m = v = self.a
		for i in range(1, self.n):
			v = XS.dot2(v, self.B.T)
			m = np.vstack((m, v))
		return m.T

	def is_weak2transitive(self):
		return XS.det2(self.A()) == 1

	def is_regular(self):
		return self.is_transitive() and self.is_weak2transitive()

	def get_lag(self):
		l = 1
		v = self.c
		while l <= self.n and XS.dot2(v, self.a) == 0:
			l = l + 1
			v = XS.dot2(v, self.B)
		return l

	def is_strong_regular(self):
		if self.is_regular() == False:
			return False
		l = self.get_lag()
		if l == 1:
			return True
		Bl = self.B
		for i in range(1,l):
			Bl = XS.dot2(Bl, self.B)
		m = v = self.c
		for i in range(1, self.n):
			v = XS.dot2(v, Bl)
			m = np.vstack((m, v))
		return XS.det2(m) == 1

	def rho2(self):
		v = self.c
		gamma = np.zeros(self.n)
		for i in range (0, self.n):
			gamma[i] = XS.dot2(v, self.a)
			v = XS.dot2(v, self.B)
		ret = 0
		A1 = XS.inv2(self.A())
		for r in range(0, self.n):
			y = np.zeros(self.n)
			y[r] = 1
			y[r + 1:] = gamma[:self.n - 1 - r]
			y = XS.dot2(y, A1)
			for i in range (0, self.n):
				y = XS.dot2(y, self.B)
			t = 0
			while True:
				t = t + 1
				if XS.dot2(y, self.a) != gamma[self.n - r + t - 2]:
					break
				y = XS.dot2(y, self.B)
			if t > ret:
				ret = t
		return self.n + ret

	def to_canon_1(self):
		B = self.B
		mr = np.eye(self.n, dtype=int)

		for i in range(0, self.n-1):
			m = np.eye(self.n, dtype = int)
			pl = np.eye(self.n, dtype=int)
			if(B[i+1, i] == 0):
				for k in range(i+2, self.n):
					if(B[k, i] == 1):
						pl[i+1], pl[k] = pl[k], pl[i+1].copy()
						pr = pl.transpose()
						B = XS.dot2(XS.dot2(pl, B), pr)
						break
			m[:,i+1] = B[:,i]
			mr = XS.dot2(mr,m)
			B = XS.dot2(m, np.dot(B, m))

		a = XS.dot2(XS.inv2(mr), self.a)
		c = XS.dot2(self.c, mr)
		right_c = np.zeros(self.n, dtype=int)
		right_c[self.n-1]=1
		if((c == right_c).all()):
			return XS(a,B,c)
		a = XS.dot2(self.C(), self.a)
		return XS(a,B,right_c)

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
	circ.to_canon_1().save_to_file(sys.argv[1][:-4] + "_frobenius.txt", sep=' ')
