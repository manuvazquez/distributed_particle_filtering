import numpy as np


def log_rbf_kernel_matrix(x, y, sigma):

	norms_x = np.sum(x**2, axis=1)
	norms_y = np.sum(y**2, axis=1)

	return -sigma*(norms_x[:, np.newaxis] + norms_y[np.newaxis, :] - 2*x.dot(y.T))


def find_weiszfeld_median(subset_atoms, sigma, maxit, tol, small_number=1e-6):

	n_subsets = len(subset_atoms)

	n_atoms = np.empty(n_subsets, dtype=int)
	kernel_plus_weights_subset = [None]*n_subsets
	subset_probs = [None]*n_subsets
	old_norms = np.zeros(n_subsets)

	for i, this_subset_atoms in enumerate(subset_atoms):

		n_atoms[i] = len(this_subset_atoms)
		subset_probs[i] = np.full((n_atoms[i], 1), 1.0/n_atoms[i])

		# all these are matrices
		kernel_matrix_subset = log_rbf_kernel_matrix(this_subset_atoms, this_subset_atoms, sigma)
		weights_subset = np.log(np.outer(subset_probs[i], subset_probs[i]))
		kernel_plus_weights_subset[i] = kernel_matrix_subset + weights_subset

	median_empirical_measure_atoms = np.vstack(subset_atoms)
	median_empirical_measure_probs = np.repeat(1/(n_subsets*n_atoms), n_atoms)[:, np.newaxis]

	# distances between atoms for the median posterior
	kernel_matrix_median = log_rbf_kernel_matrix(median_empirical_measure_atoms, median_empirical_measure_atoms, sigma)

	for jj in range(maxit):

		# if jj % 10 == 0:
		#
		# 	print('Weiszfeld iteration {}'.format(jj+1))

		# --------------------------------
		weights_median = np.log(np.outer(median_empirical_measure_probs, median_empirical_measure_probs))
		kernel_plus_weights_median = kernel_matrix_median + weights_median

		norms = np.zeros(n_subsets)

		for i, (this_subset_atoms, this_subset_probs, this_subset_kernel_plus_weights) in enumerate(zip(
				subset_atoms, subset_probs, kernel_plus_weights_subset)):

			kernel_matrix_subset_median = log_rbf_kernel_matrix(this_subset_atoms, median_empirical_measure_atoms, sigma)
			weights_subset_median = np.log(np.outer(this_subset_probs, median_empirical_measure_probs))

			kernel_plus_weights_subset_median = kernel_matrix_subset_median + weights_subset_median

			# in order to avoid very small numbers
			max_exponent = max(
				this_subset_kernel_plus_weights.max(),
				kernel_plus_weights_median.max(),
				kernel_plus_weights_subset_median.max())

			norms[i] = np.exp(max_exponent) * (
				np.exp(this_subset_kernel_plus_weights - max_exponent).sum() +
				np.exp(kernel_plus_weights_median - max_exponent).sum() -
				2 * np.exp(kernel_plus_weights_subset_median - max_exponent).sum())

			if norms[i] < small_number:

				# print('norm is small...truncating')
				norms[i] = small_number

		sqrt_norms = np.sqrt(norms)
		weiszfeld_weights = 1/sqrt_norms
		weiszfeld_weights /= sum(weiszfeld_weights)

		# histWts[:, jj] = weiszfeld_weights

		median_empirical_measure_probs = np.repeat(weiszfeld_weights/n_atoms, n_atoms)

		if (abs(norms - old_norms).sum() / n_subsets < tol) and jj > 10:

			# print('converged a iteration {}'.format(jj + 1))
			break

		old_norms = norms

	# if jj < maxit:
	#
	# 	histWts = histWts[:, :jj]

	return median_empirical_measure_atoms.T, median_empirical_measure_probs
