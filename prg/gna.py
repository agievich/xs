#******************************************************************************
# \file gna.py
# \project XS [XS-circuits into block ciphers]
# \brief Calculating the GNA (Guaranteed Number of Activations)
# \usage: gna [--F2] path_to_the_circuit rounds
# [--F2 means that the approximate LRS-bound for the case F = F2 is calculated]
# \author Sergey Agieivich [agievich@{bsu.by|gmail.com}]
# \author Egor Lawrenov
# \created 2020.06.08
# \version 2021.10.12
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

def GNA(circ, t):
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

def GNA_branch_and_bound(circ, t):
	"""
	GNA computation using branch and bound method
	Designed and implemented by Denis Parfenov and Alexandr Bakharev
	"""
	# This part is common with default GNA implementation
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

	# implicitly traverse tree of matrix partitions
	# 0 on i-th position means that i-th pair belongs to G0
	# 1 on i-th position means that i-th pair belongs to G1
	st = [[0], [1]]
	# save ways with weight(way) > (k + 1) for the current step (they
	# have too many '1' and as a result define partition with more then
	# k+1 column in G0)
	overweight = []
	while st:
		cur_pair = st.pop()
		wt = np.sum(cur_pair)
		# go to the next partition if it is not possible
		# to get enough 1's
		if t - len(cur_pair) < k + 1 - wt:
			continue
		# handle zero pair
		if wt == 0:
			st += [cur_pair + [0], cur_pair + [1]]
			continue
		# flag for consecutive columns
		# True if current partition contains n consecutive pairs of columns
		consec_flag = False
		trail_consec_n = 0
		if wt >= circ.n-1:
			consec_flag = True
			for i in range(1, circ.n):
				if cur_pair[-i] == 0:
					consec_flag = False
					break
				trail_consec_n += 1
		# check whether it is possible to choose k+1 pairs from remaining
		# and it would not lead to n consecutive pairs
		spa = ((k + 1 - wt) - (circ.n - 1 - trail_consec_n) + circ.n - 2) // (circ.n - 1)
		if t - len(cur_pair) < k + 1 - wt + spa:
			continue
		if wt <= k + 1:
			# columns TAKEN to G0
			taken = []
			# columns WASTED from G0 and then taken to G1
			wasted = []
			for i, val in enumerate(cur_pair):
				if val:
					taken += [2 * i, 2 * i + 1]
				else:
					wasted += [2 * i, 2 * i + 1]

			G0 = G[taken]
			feasible = True
			r = gf2.rank(G0)

			if cur_pair[-1] == 1:
				# during the last step one pair was added to G0
				# and all columns from G1 should be checked for
				# linear dependence with new state of G0
				if r >= t + circ.n - 1:
					# if rank >= t + n - 1 then partition is not feasible
					continue
				# check all columns from G1
				for i in wasted:
					if gf2.rank(np.vstack((G0, G[i]))) == r:
						# if there is linear dependent column in G1
						# partition is not feasible
						feasible = False
						break
			else:
				# during the last step one pair was added to G1
				# and only this pair should be checked for linear dependence
				last = [wasted[-1], wasted[-2]]
				# check only last column from G1
				for i in last:
					if gf2.rank(np.vstack((G0, G[i]))) == r:
						# if one of this columns is linear dependent with G0
						# partition is not feasible
						feasible = False
						break

			if not feasible:
				continue
			if len(cur_pair) < t:
				if consec_flag:
					# if there are n consecutive pairs in G0
					# add the next column to G1 only
					st += [cur_pair + [0]]
				else:
					# enqueue both following partitions
					st += [cur_pair + [0], cur_pair + [1]]
			else:
				# increase k and enqueue previously overweight partitions
				k += 1
				st += overweight
				overweight.clear()
		else:
			overweight.append(cur_pair)
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
		print("GNA_branch_and_bound(circuit=%s, rounds=%d) = %d" %\
			(circ_fname, t, GNA_branch_and_bound(circ, t)))
		print("GNA(circuit=%s, rounds=%d) = %d" % \
			(circ_fname, t, GNA(circ, t)))
