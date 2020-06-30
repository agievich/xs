#******************************************************************************
# \file gna.py
# \project XS [XS-circuits into block ciphers]
# \brief Calculating the GNA (Guaranteed Number of Activations)
# \usage: gna [--lrs] path_to_the_circuit_descr rounds
#   [--lrs means that the approximate LRS-bound is calculated]
# \author Sergey Agieivich [agievich@{bsu.by|gmail.com}]
# \author Egor Lawrenov
# \created 2020.06.08
# \version 2020.06.30
# \license Public domain
#******************************************************************************

import sys
import itertools
import numpy as np
from xs import XS
import gf2

def GNA2(circ, t):
	a, B, c = circ.aBc()
	if t < circ.n:
		return 0
	if t == circ.n:
		return 1
	f = gf2.add(a, B[:, -1])
	min_d = t
	for s0 in itertools.product([0, 1], repeat=circ.n):
		d = np.sum(s0)
		if d == 0:
			continue
		s = list(s0)
		for i in range(t - circ.n):
			s = np.append(s[1:], gf2.dot(s, f))
			d = d + s[-1]
		if d < min_d:
			min_d = d
	return min_d

def GNA(circ, t):
	a, B, c = circ.aBc()
	if t < circ.n:
		return 0
	if t == circ.n:
		return 1
	# build (transposed) G 
	b1 = np.concatenate((B[:, -1], [1]))
	G = np.empty((2 * t, circ.n + t), dtype=int)
	for i in range(0, t):
		G[2 * i] = np.concatenate((gf2.zeros(i), a, gf2.zeros(t - i)))
		G[2 * i + 1] = np.concatenate((gf2.zeros(i), b1, gf2.zeros(t - i - 1)))
	# run over k
	k = t - GNA2(circ, t)
	feasible_partition = True
	while k < t and feasible_partition:
		# iterate over partitions (G0, G1)
		for combination in itertools.combinations(range(0, t), k + 1):
			partition = []
			for i in combination:
				partition.extend([2 * i, 2 * i + 1])
			G0 = G[partition]
			feasible_partition = True
			r = gf2.rank(G0)
			if r >= t + circ.n - 1:
				feasible_partition = False	
				continue
			for i in range(0, 2 * t):
				if i in partition:
					continue	
				if gf2.rank(np.vstack((G0, G[i]))) == r:
					feasible_partition = False
					break
			if not feasible_partition:
				continue
			# increase k
			k = k + 1
			break
	# finish
	return t - k

if __name__ == '__main__':
    # number of args
	if len(sys.argv) < 3 or len(sys.argv) > 4 or\
		len(sys.argv) == 4 and sys.argv[1] != "--lrs":
		raise IOError("Usage: gna [---lrs] path_to_the_circuit_descr rounds")
	# circuit
	circ_fname = sys.argv[-2]
	circ = XS.read_from_file(circ_fname, ' ')
	print("circuit = %s" % circ_fname)
	if not circ.is_regular:
		raise IOError("Irregular circuit")
	circ = circ.CF1()
	# rounds
	t = int(sys.argv[-1])
	if t <= 0:
		raise IOError("Zero or negative number of rounds")
	# GNA
	if len(sys.argv) == 4:
		print("GNA[lrs](circuit, rounds=%d) = %d" % (t, GNA2(circ, t)))
	else:
		print("GNA(circuit, rounds=%d) = %d" % (t, GNA(circ, t)))
