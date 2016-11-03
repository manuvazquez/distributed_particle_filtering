import sys
import os
import numpy as np

import colorama

from . import util
import smc.particle_filter.particle_filter

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util


class PopulationMonteCarlo(smc.particle_filter.particle_filter.ParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, prng, name=None):

		super().__init__(n_particles, resampling_algorithm, resampling_criterion, name=name)

		# self._pf = copy.deepcopy(pf)
		self._pf = pf
		self._prior_mean = prior_mean
		self._prior_covar = prior_covar
		self._prng = prng

		# these are "global" attributes
		self._samples = None
		self._loglikelihoods = None
		self._mean = None
		self._covar = None
		self._weights = None

	@property
	def weights(self):

		return self._weights

	def initialize(self):

		# the initial mean and covariance are given by the prior
		self._mean = self._prior_mean
		self._covar = self._prior_covar

	def step(self, observations):

		# samples are drawn from the mean and covariance
		self._samples = self._prng.multivariate_normal(self._mean, self._covar, size=self._n_particles)

		self._loglikelihoods = np.zeros(self._n_particles)

		for i_sample, (tx_power, min_power, path_loss_exp) in enumerate(self._samples):

			self._loglikelihoods[i_sample] = util.loglikelihood(self._pf, observations, tx_power, min_power, path_loss_exp)

		self._weights = manu.smc.util.normalize_from_logs(self._loglikelihoods)

		self.update_proposal()

		adjusted_mean = self._mean.copy()
		adjusted_mean[:2] = np.exp(adjusted_mean[:2])

		print('mean:\n', self._mean)
		print('covar:\n', self._covar)
		print('adjusted mean:\n', colorama.Fore.LIGHTWHITE_EX + '{}'.format(adjusted_mean) + colorama.Style.RESET_ALL)

	def update_proposal(self):

		# np.ma.average(self._samples, axis=1, weights=self._weights).data
		self._mean = self._weights @ self._samples
		self._covar = np.cov(self._samples.T, ddof=0, aweights=self._weights)


class NonLinearPopulationMonteCarlo(PopulationMonteCarlo):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, M_T, prng, name=None):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, prng, name=name)

		self._M_T = M_T

		self._unclipped_weights = None

	@property
	def weights(self):

		return self._unclipped_weights

	def update_proposal(self):

		# this is saved because it's returned by the above property
		self._unclipped_weights = self._weights.copy()

		# indices of the samples whose weight is to be clipped
		i_clipped = np.argpartition(self._loglikelihoods, -self._M_T)[-self._M_T:]

		# minimum (unnormalized) weight among those to be clipped
		clipping_threshold = self._loglikelihoods[i_clipped[0]]

		self._loglikelihoods[i_clipped] = clipping_threshold

		self._weights = manu.smc.util.normalize_from_logs(self._loglikelihoods)

		self._mean = self._weights @ self._samples
		self._covar = np.cov(self._samples.T, ddof=0, aweights=self._weights)


class NonLinearPopulationMonteCarloCovarOnly(NonLinearPopulationMonteCarlo):

	def update_proposal(self):

		super().update_proposal()

		# mean is recomputed using the unclipped weights
		self._mean = self._unclipped_weights @ self._samples