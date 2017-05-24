import numpy as np
import scipy.stats

from . import util


class MetropolisHastings:

	def __init__(self, n_samples, pf, prior, prng, burn_in_period, proposal, name=None):

		# otherwise, the number of samples is not comparable to that of the IS algorithms
		assert burn_in_period == 0

		self.name = name

		self._n_samples = n_samples
		self._pf = pf
		self._prior = prior
		self._prng = prng

		self._burn_in_period = burn_in_period

		# number of samples + number of samples for burn-in period + initial sample
		self._chain = np.empty((self._prior.sample_len, n_samples + burn_in_period))

		# self._kernel_covar = prior_covar*covar_ratio
		self._proposal = proposal

		self._i_sample = 0

		self._last_posterior = None

	def run(self, observations):

		# initialization
		self._chain[:, self._i_sample] = self._prior.draw_sample()

		# the logarithm of the prior for the sample just generated
		log_prior = np.log(self._prior.pdf(self._chain[:, self._i_sample]))

		# the posterior (up to a proportionality constant) of the last sample
		self._last_posterior = util.loglikelihood(self._pf, observations, *self._chain[:, self._i_sample]) + log_prior

		# another sample is to be drawn
		self._i_sample += 1

		self.extend_chain(observations, self._burn_in_period + self._n_samples - 1)

		# reset (so that run can be called again)
		self._i_sample = 0

	@property
	def chain(self):

		return self._chain

	def compute_mean(self, n_samples):

		return self._chain[:, self._burn_in_period: self._burn_in_period + n_samples].mean(axis=1)

	def compute_half_sample_mean(self, n_samples):

		return self._chain[:, n_samples//2:  n_samples].mean(axis=1)

	def extend_chain(self, observations, n):

		# for every new sample to be added to the Markov chain
		for i in range(n):

			# a candidate sample is generated...
			candidate = self._proposal.draw_sample(1, {'mean': self._chain[:, self._i_sample-1]})

			# the log of the prior evaluated at the candidate
			log_prior = np.log(self._prior.pdf(candidate))

			# ...and its posterior (up to a proportionality constant) are computed
			# NOTE: "candidate" is actually a 2D (numpy) array with a single row
			candidate_posterior = util.loglikelihood(self._pf, observations, *candidate[0]) + log_prior

			# if the likelihood of the candidate is larger...
			if candidate_posterior > self._last_posterior:

				# the sample is accepted
				self._chain[:, self._i_sample] = candidate

				# the new posterior is kept
				self._last_posterior = candidate_posterior

			else:

				# the threshold does not depend on the proposal due the symmetry thereof
				threshold = np.exp(candidate_posterior - self._last_posterior)

				# if the threshold is essentially 0 or the random sample is larger
				if np.isclose(threshold, 0.) or (self._prng.random_sample() > threshold):

					# the new sample is equal to the previous
					self._chain[:, self._i_sample] = self._chain[:, self._i_sample-1]

				else:

					# the sample is accepted
					self._chain[:, self._i_sample] = candidate

					# the new posterior is kept
					self._last_posterior = candidate_posterior

			self._i_sample += 1

		# import code
		# code.interact(local=dict(globals(), **locals()))