import sys
import os
import numpy as np
import scipy.stats
import typing

# sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util


def normal_parameters_from_lognormal(mean, var):

	aux = 1 + var / mean ** 2
	var = np.log(aux)
	mean = np.log(mean) - np.log(np.sqrt(aux))

	return mean, var


def loglikelihood(pf, observations, log_tx_power, log_min_power, log_path_loss_exp):

	# the "inner" PF (for approximating the likelihood) is initialized
	pf.initialize()

	# the parameters of the sensors *within* the PF are set accordingly
	for s in pf.sensors:

		s.set_parameters(
			tx_power=np.exp(log_tx_power), minimum_amount_of_power=np.exp(log_min_power),
			path_loss_exponent=np.exp(log_path_loss_exp))

	# this is needed to rebuild the sensors with the new parameters
	pf.reset_sensors_array()

	res = 0.

	# the collection of observations is processed
	for obs in observations:

		# ...a step is taken
		pf.step(obs)

		# the loglikelihoods computed by the "inner" bootstrap filter
		loglikes = pf.last_unnormalized_loglikelihoods

		# logarithm of the average
		res += manu.smc.util.log_sum_from_individual_logs(loglikes) - np.log(len(loglikes))

	return res


class ProposalUpdater:

	# @staticmethod
	def update_proposal(self, particle_filter):

		# np.ma.average(particle_filter._samples, axis=1, weights=particle_filter._weights).data
		particle_filter._mean = particle_filter._weights @ particle_filter._samples
		particle_filter._covar = np.cov(particle_filter._samples.T, ddof=0, aweights=particle_filter._weights)

		# if the covariance matrix is "essentially" all-zeros, it may happen that samples drawn thereof have zero
		# density (mathematically preposterous but possible due to finite precision issues); in order to avoid this...
		# if all the coefficients in the covariance matrix are *close* to zero...
		if np.allclose(particle_filter._covar, 0):

			# ...the covariance matrix is set equal to the prior covariance matrix
			particle_filter._covar = particle_filter._prior_covar

			return

		# ...still, we make sure it is possible to evaluate the density of a sample with the above covariance...
		try:
			# ...we evaluate the density at the mean
			scipy.stats.multivariate_normal.pdf(x=particle_filter._mean, mean=particle_filter._mean, cov=particle_filter._covar)
		# if it is not possible...
		except (np.linalg.linalg.LinAlgError, ValueError):
			# ...the covariance matrix is set equal to the prior covariance matrix
			particle_filter._covar = particle_filter._prior_covar

	def initialize(self):

		# no need here
		pass


class ClippingProposalUpdater(ProposalUpdater):

	def __init__(self, n_clipped: int) -> None:

		super().__init__()

		self._n_clipped = n_clipped

	def update_proposal(self, particle_filter):

		# weights before clipping are saved since they are returned by the "weights" property of the "particle_filter"
		# (to be used when marking down the largest weight and also when computing the effective sample size)
		particle_filter._unclipped_weights = particle_filter._weights.copy()

		# indices of the samples whose weight is to be clipped
		i_clipped = np.argpartition(particle_filter._unnormalized_log_weights, -self._n_clipped)[-self._n_clipped:]

		# minimum (unnormalized) weight among those to be clipped
		clipping_threshold = particle_filter._unnormalized_log_weights[i_clipped[0]]

		particle_filter._unnormalized_log_weights[i_clipped] = clipping_threshold

		particle_filter._weights = manu.smc.util.normalize_from_logs(particle_filter._unnormalized_log_weights)

		particle_filter._mean = particle_filter._weights @ particle_filter._samples
		particle_filter._covar = np.cov(particle_filter._samples.T, ddof=0, aweights=particle_filter._weights)


class ClippedCovarianceProposalUpdater(ClippingProposalUpdater):

	def update_proposal(self, particle_filter):

		super().update_proposal(particle_filter)

		# mean is recomputed using the unclipped weights
		particle_filter._mean = particle_filter._unclipped_weights @ particle_filter._samples


class VaryingClippedNumberClippingProposalUpdater(ClippingProposalUpdater):

	def __init__(self, n_clipped: typing.Sequence[int]) -> None:

		# the whole list is kept
		self._n_clipped_list = n_clipped

		self._i_n_clipped = 0

	def update_proposal(self, particle_filter):

		# the appropriate "n_clipped" (M_T) is picked
		self._n_clipped = self._n_clipped_list[self._i_n_clipped]

		super().update_proposal(particle_filter)

		# in the next iteration, if any, the following "n_clipped" element in the list should be picked
		self._i_n_clipped += 1

	def initialize(self):

		# reset
		self._i_n_clipped = 0
