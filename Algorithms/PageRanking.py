#CS641 Homework-1
#Author: Sai Nikhil Reddy  Mettupally
#Date: 02/07/2018

import numpy as np

def pageranking(G):
	a = np.sum(mat,axis=0)
	c = G/a 	#transition matrix
	print(c)
	d = np.sum(c,axis=0)
	f = c
	i = 1

	while d.all() == 1:	#as long as the sum of columns is 1
		print("Iteration",i)	
		dot_mul = np.dot(f,c)
		q = np.abs(f-dot_mul)	#error calculation
		m = np.around(q,decimals = 8)
		#print(m)	#debug statement
		if m.all() < 0.00001:	#break if error less than threshold
			break
		print("dot_mul:\n",dot_mul)	#matrix in each iteration
		f = dot_mul
		i += 1
		g = dot_mul[:,0]	#eigen vector
	return g

#testing
mat = np.random.random((5,5))
output = pageranking(mat)
print(output)
print(np.sum(output))
