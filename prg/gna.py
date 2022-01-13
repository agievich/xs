#******************************************************************************
# \file gna.py
# \project XS [XS-circuits into block ciphers]
# \brief Calculating the GNA (Guaranteed Number of Activations)
# - GNA_RI -- reference implementation (https://eprint.iacr.org/2020/850);
# - GNA -- fast implementation based on branch-and-bound technique 
#   (see https://bmm.mca.nsu.ru/download/mca_o_cypher/Note.pdf for details).
# \usage: gna [--F2] path_to_the_circuit rounds
# [--F2 means that the approximate LRS-bound for the case F = F2 is calculated]
# \author Sergey Agieivich, Egor Lawrenov
# \author Denis Parfenov, Alexandr Bakharev
# \created 2020.06.08
# \version 2022.01.13
# \license Public domain
#******************************************************************************

import sys
import itertools
import numpy as np
from xs import XS
import gf2

def GNAF2(circ, t):
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

def GNA_RI(circ, t):
	a, B, c = circ.aBc()
	if t < circ.n:
		return 0
	if t == circ.n:
		return 1
	# build G (transposed)
	b1 = np.concatenate((B[:, -1], [1]))
	G = np.empty((2 * t, circ.n + t), dtype=int)
	for i in range(0, t):
		G[2 * i] = np.concatenate((gf2.zeros(i), a, gf2.zeros(t - i)))
		G[2 * i + 1] = np.concatenate((gf2.zeros(i), b1, gf2.zeros(t - i - 1)))
	# run over k
	k = t - GNAF2(circ, t)
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

def GNA(circ, t):
	"""
	GNA computation using branch and bound method
	Designed and implemented by Denis Parfenov and Alexandr Bakharev
	"""
	# this part is the same as in the reference implementation
	a, B, c = circ.aBc()
	if t < circ.n:
		return 0
	if t == circ.n:
		return 1
	# build G (transposed)
	b1 = np.concatenate((B[:, -1], [1]))
	G = np.empty((2 * t, circ.n + t), dtype=int)
	for i in range(0, t):
		G[2 * i] = np.concatenate((gf2.zeros(i), a, gf2.zeros(t - i)))
		G[2 * i + 1] = np.concatenate((gf2.zeros(i), b1, gf2.zeros(t - i - 1)))
	# run over k
	k = t - GNAF2(circ, t)

	# implicitly traverse a binary tree that encodes partitions of G:
	# - a node of level d is a binary string of length d: s = s1 s2 ... sd;
	# - if the i-th pair (of columns) of G belongs to G0, then si = 1;
	# - if the i-th pair of G belongs to G1, then si = 0.
	partitions = [[0], [1]]
	# keep the partitions with weight > k + 1 in overwieght (they are not 
	# feasible with the current k but can become feasible as k grows)
	overweight = []
	while partitions:
		partition = partitions.pop()
		wt = np.sum(partition)
		# skip the partition if it is not possible to achieve enough 1's
		if t - len(partition) < k + 1 - wt:
			continue
		# handle zero weight
		if wt == 0:
			partitions += [partition + [0], partition + [1]]
			continue
		# flag indicating whether G0 contains n - 1 consecutive pairs
		consec_n_1_ones = False
		trail_ones = 0
		if wt >= circ.n - 1:
			consec_n_1_ones = True
			for i in range(1, circ.n):
				if partition[-i] == 0:
					consec_n_1_ones = False
					break
				trail_ones += 1
		# check whether the partition can be continued so that 
		# it does not contain n consecutive pairs in G0
		need_zeros = (k + 1 - wt) - (circ.n - 1 - trail_ones)
		need_zeros = (need_zeros + circ.n - 2) // (circ.n - 1)
		if t - len(partition) < k + 1 - wt + need_zeros:
			continue
		if wt <= k + 1:
			# pairs TAKEN to G0
			taken = []
			# pairs WASTED from G0 and then taken to G1
			wasted = []
			for i, val in enumerate(partition):
				if val:
					taken += [2 * i, 2 * i + 1]
				else:
					wasted += [2 * i, 2 * i + 1]

			G0 = G[taken]
			feasible = True
			r = gf2.rank(G0)

			# if during the last step a pair was added to G0, then all columns 
			# of G1 should be checked for linear dependence with updated G0
			if partition[-1] == 1:
				# if rank(G0) >= t + n - 1
				if r >= t + circ.n - 1:
					# then the partition is not feasible
					continue
				# check all columns of G1
				for i in wasted:
					# if there is a linearly dependent column in G1
					if gf2.rank(np.vstack((G0, G[i]))) == r:
						# then the partition is not feasible
						feasible = False
						break
			# if during the last step a pair was added to G1, then only 
			# this pair should be checked for linear dependence
			else:
				last = [wasted[-1], wasted[-2]]
				# check the last pair of columns of G1
				for i in last:
					# if one of these columns is linearly dependent with G0
					if gf2.rank(np.vstack((G0, G[i]))) == r:
						# then the partition is not feasible
						feasible = False
						break

			if not feasible:
				continue
			if len(partition) < t:
				# if there are n - 1 consecutive pairs in G0
				if consec_n_1_ones:
					# then the next pair has to be in G1
					partitions += [partition + [0]]
				else:
					# enqueue both of the following partitions
					partitions += [partition + [0], partition + [1]]
			else:
				# increase k and enqueue overweight partitions
				k += 1
				partitions += overweight
				overweight.clear()
		else:
			overweight.append(partition)
	return t - k

if __name__ == '__main__':
	# number of args
	if len(sys.argv) < 3 or len(sys.argv) > 4 or\
		len(sys.argv) == 4 and sys.argv[1] != "--F2":
		raise IOError("Usage: gna [--F2] path_to_the_circuit rounds")
	# circuit
	circ_fname = sys.argv[-2]
	circ = XS.read_from_file(circ_fname, ' ')
	if not circ.is_regular:
		raise IOError("Irregular circuit")
	circ = circ.CF1()
	# rounds
	t = int(sys.argv[-1])
	if t <= 0:
		raise IOError("Zero or negative number of rounds")
	# GNA
	if len(sys.argv) == 4:
		print("GNA[F2](circuit=%s, rounds=%d) = %d" %\
			(circ_fname, t, GNAF2(circ, t)))
	else:
		print("GNA(circuit=%s, rounds=%d) = %d" %\
			(circ_fname, t, GNA(circ, t)))
