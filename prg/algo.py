import sys
import numpy as np
from xs import XS


def get_bool(num, n):
		res = []
		while num > 0:
			res.append(num % 2)
			num //= 2
		res += [0] * (n - len(res))
		return np.array(res[::-1])

def get_int(a):
	return a

def rotate(l, n):
	return l[n:] + l[:n]

def next_combination(n, a):
	k = len(a)
	for i in range(k-1, -1, -1):
		if(a[i] < n - k + i):
			a[i]+=1
			for j in range(i+1, k):
				a[j] = a[j-1] + 1
			return True
	return False

def gf2_rank(rows):
	rank = 0
	while rows:
		pivot_row = rows.pop()
		if pivot_row:
			rank += 1
			lsb = pivot_row & -pivot_row
			for index, row in enumerate(rows):
				if row & lsb:
					rows[index] = row ^ pivot_row
	return rank

def d(scheme, dF2, t):
	k = t - dF2 +1
	tmp1 = np.zeros((scheme.n + t, 1), dtype=int)
	tmp2 = np.zeros((scheme.n + t, 1), dtype=int)
	first[:scheme.n,:] = scheme.a
	b = scheme.B[:,-1]
	second[:scheme.n,:] = b
	second[scheme.n:scheme.n+1,:] = 1
	G = np.hstack((first, second))
	for i in range(1,t+1):
		first = rotate(first, i)
		second = rotate(second,i)
		G = np.hstack((G,first))
		G = np.hstack((G,second))
		combination = np.arange(k)
		while True:
			rank_flag = False
			g1_flag = False
			G_0 = G.transpose()[combination]
			rows = [get_int(x) for x in G_0]
			rank_g0 = gf2_rank(rows)
			if(rank_g0==t+scheme.n):
				rank_flag = True
			if(rank_flag==False):
				for col in G.transpose():
					rows.append(get_int(col))
					rank_g1 = gf2_rank(rows)
					rows.pop()
					if(rank_g1==rank_g0):
						g1_flag = True
			if(g1_flag==True or rank_g0==True):
				k+=1
			else:
				return t - k
			if (next_combination(circ.n, combination)==False):
				break

def df2(scheme):
	b = scheme.B[:,-1]
	for t in range(2*scheme.n, 6*scheme.n):
		#cnts = {}
		cnts = []
		for i in range(1,2**scheme.n):
			coeff = (scheme.a + b)%2
			bit_arr = scheme.get_bool(i)
			for k in range(t - scheme.n):
				new_item = XS.dot2(bit_arr[k:k+scheme.n], coeff)
				bit_arr = np.append(bit_arr, new_item)
			#activations = np.sum(bit_arr)
			#cnts.setdefault(activations, []).append(list(bit_arr))
			cnts.append(np.sum(bit_arr))
		min_activations = min(cnts)
		print("activations on {} tacts = {}".format(t, min_activations))
		#print("activations on " + str(min_activations))

if __name__ == '__main__':
	circ_filename = sys.argv[1]
	circ = XS.read_from_file(circ_filename, ' ')
	k = 3
	print(circ.n)
	combination = np.arange(0,3)
	while True:
		print(combination)
		if (next_combination(circ.n, combination)==False):
			break

