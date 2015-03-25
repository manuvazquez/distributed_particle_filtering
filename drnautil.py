import numpy as np

def supremumUpperBound(M,c,q,epsilon):
	
	return (c**q) / M**(q-epsilon)

f = lambda c,z,K,M: c/(M**z*np.sqrt(K))

def error_fit(Ms,f_Ms,K,step = 0.01,n_iter = 1000000):
	
	# initial values for c,z
	c,z = np.random.rand(2)
	
	for i in range(n_iter):

		# so that computations are reused
		Ms_to_the_z = Ms**z
		Ms_to_the_z_times_sqrt_K = Ms_to_the_z*np.sqrt(K)
		error = f_Ms - c/Ms_to_the_z_times_sqrt_K
		
		# derivative respect to c
		grad_c = np.sum(2 * error * (-1/Ms_to_the_z_times_sqrt_K))
		
		# derivative respect to z
		grad_z = np.sum(2* error * c/np.sqrt(K)*np.log(Ms)*(1/Ms_to_the_z))
		
		c -= step*grad_c
		z -= step*grad_z
		
		print('c = {}, z = {}'.format(c,z))
	
	return c,z