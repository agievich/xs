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
import gf2

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

	def M(self):
		M = np.ndarray(shape = (self.n + 1, self.n + 1), dtype=int)
		M[:-1, -1] = self.a
		M[:-1, :-1] = self.B
		M[-1, :-1] = self.c
		M[self.n, self.n] = 0
		return M

	def aBc(self):
		return self.a, self.B, self.c

	def is_invertible(self):
		if gf2.det2(self.B) == 0:
			return gf2.det2(self.M()) == 1
		return gf2.dot2(gf2.dot2(self.c, gf2.inv2(self.B)), self.a) == 0

	def get_type(self):
		assert(self.is_invertible())
		if gf2.det2(self.B) == 1:
			return 1
		else:
			return 2

	def inv(self):
		assert(self.is_invertible())
		if self.get_type() == 1:
			B1 = gf2.inv2(self.B)
			a1 = gf2.dot2(B1, self.a)
			c1 = gf2.dot2(self.c, B1)
		else:
			M = gf2.inv2(self.M())
			a1 = M[:-1, -1]
			B1 = M[:-1, :-1]
			c1 = M[-1, :-1]
		return XS(a1, B1, c1)

	def dual(self):
		return XS(self.c, self.B.T, self.a)

	def C(self):
		m = v = self.c
		for i in range(1, self.n):
			v = gf2.dot2(v, self.B)
			m = np.vstack((v, m))
		return m

	def is_transitive(self):
		return gf2.det2(self.C()) == 1

	def A(self):
		m = v = self.a
		for i in range(1, self.n):
			v = gf2.dot2(v, self.B.T)
			m = np.vstack((m, v))
		return m.T

	def is_weak2transitive(self):
		return gf2.det2(self.A()) == 1

	def is_regular(self):
		return self.is_transitive() and self.is_weak2transitive()

	def get_lag(self):
		l = 1
		v = self.c
		while l <= self.n and gf2.dot2(v, self.a) == 0:
			l = l + 1
			v = gf2.dot2(v, self.B)
		return l

	def is_strong_regular(self):
		if self.is_regular() == False:
			return False
		l = self.get_lag()
		if l == 1:
			return True
		Bl = self.B
		for i in range(1,l):
			Bl = gf2.dot2(Bl, self.B)
		m = v = self.c
		for i in range(1, self.n):
			v = gf2.dot2(v, Bl)
			m = np.vstack((m, v))
		return gf2.det2(m) == 1

	def rho2(self):
		v = self.c
		gamma = np.zeros(self.n)
		for i in range (0, self.n):
			gamma[i] = gf2.dot2(v, self.a)
			v = gf2.dot2(v, self.B)
		ret = 0
		A1 = gf2.inv2(self.A())
		for r in range(0, self.n):
			y = np.zeros(self.n)
			y[r] = 1
			y[r + 1:] = gamma[:self.n - 1 - r]
			y = gf2.dot2(y, A1)
			for i in range (0, self.n):
				y = gf2.dot2(y, self.B)
			t = 0
			while True:
				t = t + 1
				if gf2.dot2(y, self.a) != gamma[self.n - r + t - 2]:
					break
				y = gf2.dot2(y, self.B)
			if t > ret:
				ret = t
		return self.n + ret

	#первая каноническая форма, где c = (0,0,0,...,1)
	def form_one(self):
		B = self.B
		B, P = gf2.frobenius_cell(B)

		a = gf2.dot2(gf2.inv2(P), self.a)
		c = gf2.dot2(self.c, P)
		right_c = np.zeros(self.n, dtype=int)
		right_c[self.n-1]=1
		if((c == right_c).all()):
			return XS(a,B,c)

		M_n = np.eye(self.n, dtype=int)
		P_c = gf2.dot2(c,M_n)
		b = B[:,self.n-1]
		for i in range(self.n-1, 0, -1):
			M_n = gf2.dot2(B, M_n)
			M_n = (M_n + b[i] * np.eye(self.n, dtype=int))%2
			P_c = np.vstack((P_c, (gf2.dot2(c,M_n))))
		P_c = P_c[::-1]
		a = gf2.dot2(P_c, self.a)
		return XS(a,B,right_c)

	#вторая каноническая форма, где a = (1,0,0,...,0)^T
	def form_two(self):
		B = self.B
		B, P = gf2.frobenius_cell(B)

		a = gf2.dot2(gf2.inv2(P), self.a)
		c = gf2.dot2(self.c, P)
		right_a = np.zeros((self.n,1), dtype=int)
		right_a[0]=1
		P = self.A()
		c = gf2.dot2(P, c)
		return XS(right_a, B, c)

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
			return True

if __name__ == '__main__':
	circ_filename = sys.argv[1]
	circ = XS.read_from_file(circ_filename, ' ')
	print("circuit = %s:" % circ_filename)
	regular_flag = circ.describe()
	print("  inv(circuit):")
	circ.inv().describe()
	print("  dual(circuit):")
	circ.dual().describe()
	if (regular_flag):
		form_1 = circ.form_one()
		form1_a, form1_B, form1_c = form_1.aBc()
		form_2 = circ.form_two()
		form2_a, form2_B, form2_c = form_2.aBc()
		n = len(form1_a)
		out_info = "canonical forms:\n {}\n {} and {} or\n {} and {}".format(
			form1_B[:,-1].transpose(), form1_a.reshape(n), form1_c, form2_a.reshape(n), form2_c)
		print(out_info)
		form_1 = circ.dual().form_one()
		form1_a, form1_B, form1_c = form_1.aBc()
		form_2 = circ.dual().form_two()
		form2_a, form2_B, form2_c = form_2.aBc()
		out_info = "dual(canonical forms):\n {}\n {} and {} or\n {} and {}".format(
			form1_B[:,-1].transpose(), form1_a.reshape(n), form1_c, form2_a.reshape(n), form2_c)
		print(out_info)