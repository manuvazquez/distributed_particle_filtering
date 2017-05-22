import sys
import os
import numpy as np
import scipy.stats

import colorama

from . import util
from . import pmc

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util


class AdaptiveMultipleImportanceSampling(pmc.PopulationMonteCarlo):

	def __init__(
			self, n_particles, n_iterations, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar,
			prng, name=None):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, prng, name)

		# "n_particles" should now be a list and the number of iterations is inferred therefrom
		self._n_iterations = n_iterations

		self._means_list = None
		self._covars_list = None

		# the index for the current iteration
		self._i_iter = None

		self._structured_samples = None

		self._unnormalized_log_weights = None
		self._structured_log_weights = None

		self._structured_log_targets = None
		self._weighted_proposals = None

	def initialize(self):

		super().initialize()

		# memory for the lists storing the means and covariances is reserved
		# NOTE: this is done here so that the object can be reused by simply calling this method
		self._means_list = [None]*self._n_iterations
		self._covars_list = [None]*self._n_iterations

		self._i_iter = 0

		# the samples drawn at every iteration must be kept
		self._structured_samples = np.zeros((self._n_particles, len(self._prior_mean), self._n_iterations))

		self._structured_log_weights = np.zeros((self._n_particles, self._n_iterations))
		self._structured_log_targets = np.zeros((self._n_particles, self._n_iterations))

		# for every particle and every time instant, the linear combination of all the proposals
		self._weighted_proposals = np.zeros((self._n_particles, self._n_iterations))

	def step(self, observations):

		# mean and covariance matrix to be used in this iteration are saved
		self._means_list[self._i_iter] = self._mean.copy()
		self._covars_list[self._i_iter] = self._covar.copy()

		# samples are drawn from the mean and covariance
		self._structured_samples[..., self._i_iter] = self._prng.multivariate_normal(
			self._mean, self._covar, size=self._n_particles)

		# for the sake of convenience
		samples = self._structured_samples[..., self._i_iter]

		# this will be "filled in" in the loop below
		self._loglikelihoods = np.zeros(self._n_particles)

		for i_sample, (tx_power, min_power, path_loss_exp) in enumerate(samples):

			self._loglikelihoods[i_sample] = util.loglikelihood(self._pf, observations, tx_power, min_power, path_loss_exp)

		log_prior = np.log(scipy.stats.multivariate_normal.pdf(
			x=samples, mean=self._prior_mean, cov=self._prior_covar))

		# NOTE: the first time this is called, "log_prior" should be equal to "log_proposal"

		# the previous *proposal functions* are also accounted for (along with the current one at index "self._i_iter")
		for proposal_mean, proposal_covar in zip(self._means_list[:self._i_iter+1], self._covars_list[:self._i_iter+1]):

			self._weighted_proposals[:, self._i_iter] += self._n_particles * scipy.stats.multivariate_normal.pdf(
				x=samples, mean=proposal_mean, cov=proposal_covar)

		# the *target* pdf (likelihood times prior) is stored for later reuse
		self._structured_log_targets[..., self._i_iter] = self._loglikelihoods + log_prior

		# the logarithms of the weights are computed and stored in the corresponding slice
		self._structured_log_weights[:, self._i_iter] =\
			self._structured_log_targets[..., self._i_iter] - np.log(self._weighted_proposals[:, self._i_iter]) + np.log(
				(self._i_iter+1)*self._n_particles)

		# the weights of the previously drawn samples also need to be updated
		for i_iter in range(self._i_iter):

			# the old particles are evaluated at the new proposal...
			self._weighted_proposals[:, i_iter] += self._n_particles * scipy.stats.multivariate_normal.pdf(
				x=self._structured_samples[..., i_iter], mean=self._mean, cov=self._covar)

			# ...and the weights recomputed using that and the previously stored value for the *target*
			self._structured_log_weights[:, self._i_iter] = \
				self._structured_log_targets[..., self._i_iter] - np.log(self._weighted_proposals[:, i_iter])

		# the methods in the superclass expect the samples in the usual form

		# the samples from every *past* iteration are stacked one upon another
		self._samples = np.moveaxis(self._structured_samples[..., :self._i_iter+1], 1, 2).\
			reshape((-1, samples.shape[1]), order='F')

		# this is actually *required* since the ProposalUpdater is counting on it
		self._unnormalized_log_weights = self._structured_log_weights[..., :self._i_iter + 1].ravel(order='F')

		# the (log)weights are normalized (above have already been shaped up as a row vector)
		self._weights = manu.smc.util.normalize_from_logs(self._unnormalized_log_weights)

		# self.update_proposal()
		self._proposal_updater.update_proposal(self)

		adjusted_mean = np.exp(self._mean.copy())

		# the number of iteration is increased
		self._i_iter += 1

		print('mean:\n', self._mean)
		print('covar:\n', self._covar)
		print('adjusted mean:\n', colorama.Fore.LIGHTWHITE_EX + '{}'.format(adjusted_mean) + colorama.Style.RESET_ALL)


class NonLinearAdaptiveMultipleImportanceSampling(AdaptiveMultipleImportanceSampling):

	def __init__(
			self, n_particles, n_iterations, resampling_algorithm, resampling_criterion, pf, prior_mean,
			prior_covar, M_T, prng, name=None):

		# "build_proposal_updater" (below) called by the parent needs this parameter
		self._M_T = M_T

		# call the __init__ method of the parent
		super().__init__(
			n_particles, n_iterations, resampling_algorithm, resampling_criterion, pf, prior_mean,prior_covar,
			prng, name)

		self._unclipped_weights = None

	@property
	def weights(self):

		return self._unclipped_weights

	def build_proposal_updater(self):

		# the "vanilla" "ProposalUpdater" is used
		return util.ClippingProposalUpdater(self._M_T)


class VaryingClippedNumberNonLinearAdaptiveMultipleImportanceSampling(NonLinearAdaptiveMultipleImportanceSampling):

	def __init__(
			self, n_particles, n_iterations, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar,
			M_T, prng, name=None):

		super().__init__(
			n_particles, n_iterations, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, M_T,
			prng, name)

		# an "M_T" for every iteration must have been received
		assert n_iterations == len(M_T)

	def build_proposal_updater(self):

		# the "vanilla" "ProposalUpdater" is used
		return util.VaryingClippedNumberClippingProposalUpdater(self._M_T)