import sys
import os
import numpy as np
import scipy.stats

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
		self._unnormalized_log_weights = None

		# the smallest representable positive in this machine and for this data type
		self._machine_eps = np.finfo(prior_covar.dtype).eps

		# a "ProposalUpdater" is instantiated (to be decided by the children classes)
		self._proposal_updater = self.build_proposal_updater()

	@property
	def weights(self):

		return self._weights

	def build_proposal_updater(self):

		# the "vanilla" "ProposalUpdater" is used
		return util.ProposalUpdater()

	def initialize(self):

		# the initial mean and covariance are given by the prior
		self._mean = self._prior_mean
		self._covar = self._prior_covar

		# this may be needed for some "ProposalUpdater"s
		self._proposal_updater.initialize()

	def step(self, observations):

		# samples are drawn from the mean and covariance
		self._samples = self._prng.multivariate_normal(self._mean, self._covar, size=self._n_particles)

		self._loglikelihoods = np.zeros(self._n_particles)

		for i_sample, (tx_power, min_power, path_loss_exp) in enumerate(self._samples):

			self._loglikelihoods[i_sample] = util.loglikelihood(self._pf, observations, tx_power, min_power, path_loss_exp)

		log_prior = np.log(scipy.stats.multivariate_normal.pdf(
			x=self._samples, mean=self._prior_mean, cov=self._prior_covar))

		log_proposal = np.log(scipy.stats.multivariate_normal.pdf(x=self._samples, mean=self._mean, cov=self._covar))

		# NOTE: the first time this is called, "log_prior" should be equal to "log_proposal"
		self._unnormalized_log_weights = self._loglikelihoods + log_prior - log_proposal

		self._weights = manu.smc.util.normalize_from_logs(self._unnormalized_log_weights)

		# self.update_proposal()
		self._proposal_updater.update_proposal(self)

		adjusted_mean = self._mean.copy()
		adjusted_mean[:2] = np.exp(adjusted_mean[:2])

		print('mean:\n', self._mean)
		print('covar:\n', self._covar)
		print('adjusted mean:\n', colorama.Fore.LIGHTWHITE_EX + '{}'.format(adjusted_mean) + colorama.Style.RESET_ALL)


class NonLinearPopulationMonteCarlo(PopulationMonteCarlo):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, M_T, prng, name=None):

		# "build_proposal_updater" (below) called by the parent needs this parameter
		self._M_T = M_T

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, prng, name=name)

		self._unclipped_weights = None

	@property
	def weights(self):

		return self._unclipped_weights

	@property
	def clipped_weights(self):

		return self._weights

	def build_proposal_updater(self):

		# the *Clipping* "ProposalUpdater" is used
		return util.ClippingProposalUpdater(self._M_T)


class NonLinearPopulationMonteCarloCovarOnly(NonLinearPopulationMonteCarlo):

	def build_proposal_updater(self):

		return util.ClippedCovarianceProposalUpdater(self._M_T)
