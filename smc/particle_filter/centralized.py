import numpy as np

import state
from smc.particle_filter.particle_filter import ParticleFilter


class TargetTrackingParticleFilter(ParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight=1.0):

		super().__init__(n_particles, resampling_algorithm, resampling_criterion)

		# a vector with the weights is created...but not initialized (that must be done by the "initialize" method)
		self._log_weights = np.empty(n_particles)

		# the state equation is encoded in the transition kernel
		self._state_transition_kernel = state_transition_kernel

		# the prior is needed to initialize the state
		self._prior = prior

		# the sensors are kept
		self._sensors = sensors

		# EVERY time, this PF is initialized, the aggregated weight is set to this value
		self._initial_aggregated_weight = aggregated_weight

		# these will get set as soon as the "initialize" method gets called
		self._state = None
		self._aggregated_weight = None

	def initialize(self):

		# initial samples...
		self._state = self._prior.sample(self._n_particles)

		# the weights are assigned equal probabilities
		self._log_weights.fill(np.log(self._initial_aggregated_weight) - np.log(self._n_particles))

		# this variable just keeps tabs on the sum of all the weights
		self._aggregated_weight = self._initial_aggregated_weight

	def step(self, observations):

		assert len(observations) == len(self._sensors)

		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._state_transition_kernel.next_state(self._state[:, i:i + 1]) for i in range(self._n_particles)])

		# TODO: this may cause a "divide by zero" warning when a likelihood is very small
		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.to_position(self._state)) for sensor, obs in zip(self._sensors, observations)]))

		# for each particle, we compute the product of the likelihoods for all the sensors
		log_likelihoods_product = loglikelihoods.sum(axis=0)

		# the weights are updated
		self._log_weights += log_likelihoods_product

		# the aggregated weight is kept up to date at all times
		self.update_aggregated_weight()

		# whatever is required (it depends on the algorithm) to avoid weights degeneracy...
		self.avoid_weight_degeneracy()

	# FIXME: this method is useless given the property below
	def get_state(self):

		return self._state

	def resample(self, normalized_log_weights):

		# the weights need to be converted to "natural" units
		normalized_weights = np.exp(normalized_log_weights)

		# we check whether a resampling step is actually needed or not
		if self._resampling_criterion.is_resampling_needed(normalized_weights):

			try:
				# the resampling algorithm is used to decide which particles to keep
				i_particles_to_be_kept = self._resampling_algorithm.get_indexes(normalized_weights)

			except ValueError:

				# this should mean the normalized weights don't add up EXACTLY to one...we renormalize them...
				normalized_weights /= normalized_weights.sum()

				# ...and try again
				i_particles_to_be_kept = self._resampling_algorithm.get_indexes(normalized_weights)

			# the above indexes are used to update the state
			self._state = self._state[:, i_particles_to_be_kept]

			# note that if the weights have been normalized ("standard" centralized particle filter),
			# then "self._aggregated_weight" is equal to 1
			self._log_weights.fill(np.log(self._aggregated_weight) - np.log(self._n_particles))

	def get_particle(self, index):

		return self._state[:, index:index+1].copy(), self._log_weights[index]

	def get_samples_at(self, indexes):

		"""Obtain (just) the samples at certain given indexes.

		This yields a "view" of the data, rather than a copy.

		Parameters
		----------
		indexes: 1-D ndarray
			The indexes of the requested particles

		Returns
		-------
		samples: 2-D ndarray
			The selected samples
		"""

		return self._state[:, indexes]

	@property
	def samples(self):

		return self._state

	@samples.setter
	def samples(self, value):

		if value.shape == self._state.shape:

			self._state = value

		else:

			raise Exception('the number and/or dimensions of the samples are not equal to the current ones')

	def set_particle(self, index, particle):

		self._state[:, index:index+1] = particle[0]
		self._log_weights[index] = particle[1]

		# the sum of the weights might have changed...
		self.update_aggregated_weight()

	def update_aggregated_weight(self):

		# the aggregated weight is simply the sum of the non-normalized weights
		self._aggregated_weight = np.exp(self._log_weights).sum()

	def compute_mean(self):

		# if all the weights in this PF/PE are zero...
		if self._aggregated_weight == 0:

			# ...then an all-zeros estimate is returned...though any should do since this estimate must contribute zero
			# return np.zeros((state.n_elements, 1))
			return np.full((state.n_elements, 1), np.pi)

		normalized_log_weights = self._log_weights - np.log(self._aggregated_weight)

		# element-wise multiplication of the state vectors and their correspondent weights,
		# followed by addition => weighted mean
		return (self._state*np.exp(normalized_log_weights)[np.newaxis, :]).sum(axis=1)[:, np.newaxis]

	def normalize_weights(self):

		# if all the weights are zero...
		if self._aggregated_weight == 0:

			# ...then normalization makes no sense and we just initialize the weights again
			self._log_weights.fill(-np.log(self._n_particles))

		else:

			self._log_weights -= np.log(self._aggregated_weight)

		# we forced this above
		self._aggregated_weight = 1.0

	# this methods encapsulates the parts within the code of "step" which are different in this class and its children
	def avoid_weight_degeneracy(self):

		self.normalize_weights()

		# the normalized weights are used to resample
		self.resample(self._log_weights)

	@property
	def log_weights(self):

		return self._log_weights

	@log_weights.setter
	def log_weights(self, value):

		if self._log_weights.shape == value.shape:

			self._log_weights = value

		else:

			raise Exception('the number of weights does not match the number of particles')

	@property
	def weights(self):

		return np.exp(self._log_weights)

	@weights.setter
	def weights(self, value):

		if self._log_weights.shape == value.shape:

			self._log_weights = np.log(value)

		else:

			raise Exception('the number of weights does not match the number of particles')

# =========================================================================================================


class EmbeddedTargetTrackingParticleFilter(TargetTrackingParticleFilter):

	@property
	def aggregated_weight(self):

		return self._aggregated_weight

	def divide_weights(self, factor):

		self._log_weights -= np.log(factor)
		self._aggregated_weight /= factor

	# NOTE: using np.close may yield quite different results
	def avoid_weight_degeneracy(self):

		# if all the weights are zero...
		if self._aggregated_weight == 0:

			# ...there is nothing we can do
			return

		else:
			# the normalized weights are used to resample
			self.resample(self._log_weights - np.log(self._aggregated_weight))

# =========================================================================================================


class TargetTrackingParticleFilterWithConsensusCapabilities(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight=1.0):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight)

		# the noise covariance matrix is built from the individual variances of each sensor
		self._noiseCovariance = np.diag([s.noiseVar for s in sensors])

		# the position of every sensor
		self._sensorsPositions = np.hstack([s.position for s in sensors])

	def set_polynomial_approximation_constants(self, M, R_p, r_a_tuples, r_a, r_d_tuples, r_d, rs_gamma):

		self._M = M
		self._R_p = R_p
		self._r_a_tuples = r_a_tuples
		self._r_a = r_a
		self._r_d_tuples = r_d_tuples
		self._r_d = r_d
		self._rs_gamma = rs_gamma

	def likelihood_mean(self, positions):

		# each row gives the distances from a sensor to ALL the positions
		distances = np.linalg.norm(positions[:, :, np.newaxis] - self._sensorsPositions[:, np.newaxis, :], axis=0).T

		return np.vstack([s.likelihoodMean(d) for d, s in zip(distances, self._sensors)])

	def step(self, observations):

		# the exponents of the monomials and their associated coefficients are extracted from the "consensed" beta's
		exponents = np.array(list(self.betaConsensus.keys()))
		betas = np.array(list(self.betaConsensus.values()))

		# for the sake of convenience
		x = state.to_position(self._state)

		# a matrix containing the monomials evaluated for the all the x's
		phi = (x[:, :, np.newaxis]**exponents.T[:, np.newaxis, :]).prod(axis=0)

		# the exponent of the Joint Likelihood Function (JLF) for every particle (x), as computed in this PE
		S = (phi * betas[np.newaxis, :]).sum(axis=1)

		# S contains exponents...and hence subtracting a constant is tantamount to scaling the power (the JLF)
		shifted_S = S - max(S)

		# the weights should be multiplied by e^shifted_S and divided by the sum thereof, and when taking
		# the logarithm this yields
		self._log_weights += shifted_S - np.log(np.exp(shifted_S).sum())

		# the aggregated weight is kept up to date at all times
		self.update_aggregated_weight()

		# whatever is required (it depends on the algorithm) to avoid weights degeneracy...
		self.avoid_weight_degeneracy()

	def pre_consensus_step(self, observations):

		assert len(observations) == len(self._sensors)

		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._state_transition_kernel.next_state(self._state[:, i:i + 1]) for i in range(self._n_particles)])

		self.polynomial_approximation(observations)

	def polynomial_approximation(self, observations):

		x = state.to_position(self._state)

		# in the first matrix, we just replicate the samples matrix (<component within sample>,<number of sample>)
		# along the third dimension; in the second matrix, the third dimension gives the number of monomial
		phi = (x[:, :, np.newaxis]**self._r_a.T[:, np.newaxis, :]).prod(axis=0)

		# the true values for the function to be approximated
		A = self.likelihood_mean(x).T

		# the coefficients of the approximation (each row corresponds to one coefficient)
		Y = np.linalg.pinv(phi).dot(A)

		# the solution is stored in a dictionary for easy access
		alpha = dict(zip(self._r_a_tuples, Y))

		# a dictionary (indexed by the elements in r_d_tuples) with the computed coefficients gamma
		gamma = {}

		for r, possible_combinations in zip(self._r_d, self._rs_gamma):

			accum = 0

			# we sum over all the possible combinations (each one gives rise to one term)
			for t in possible_combinations:

				# alpha * covariance * alpha^T (alpha has been stored as a row vector)
				accum += alpha[t[:self._M]].dot(self._noiseCovariance).dot(alpha[t[self._M:]][:, np.newaxis])

			accum /= 2

			# the computed value is added to the dictionary
			gamma[tuple(r)] = accum.item(0)

		# this term is independent of the indices
		b = self._noiseCovariance.dot(observations)

		# a dictionary to store the beta component associated to every vector of indices
		self.beta = {}

		for r in self._r_d_tuples:

			deg = sum(r)

			if deg <= self._R_p:

				self.beta[r] = alpha[r].dot(b) - gamma[r]

			elif deg <= (2*self._R_p):

				self.beta[r] = - gamma[r]

			else:

				raise Exception('coefficient for this combination of exponents not found!!')

# =========================================================================================================


class TargetTrackingParticleFilterWithFusionCenter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			i_fusion_center_neighbour, aggregated_weight=1.0):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors, aggregated_weight)

		self._i_fusion_center_neighbour = i_fusion_center_neighbour

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		distances = processing_elements_topology.distances_between_processing_elements[self._i_fusion_center_neighbour] + 1

		return sum([d*len(each_processing_element_connected_sensors) for d, c in zip(
			distances, each_processing_element_connected_sensors)])

# =========================================================================================================


class TargetTrackingGaussianParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight=1.0):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight)

		# they will be set later
		self._Q = None
		self._nu = None

	def step(self, observations):

		# TODO: this should be estimated!!
		self._estimated_n_PEs = 2

		assert len(observations) == len(self._sensors)

		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._state_transition_kernel.next_state(self._state[:, i:i + 1]) for i in range(self._n_particles)])

		# TODO: this may cause a "divide by zero" warning when a likelihood is very small
		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.to_position(self._state)) for sensor, obs in
			 zip(self._sensors, observations)]))

		# for each particle, we compute the product of the likelihoods for all the sensors
		log_likelihoods_product = loglikelihoods.sum(axis=0)

		# every likelihood is exponentiated by the estimate of the number of PEs
		self._log_weights = log_likelihoods_product*self._estimated_n_PEs

		# the aggregated weight is kept up to date at all times
		self.update_aggregated_weight()

		# normalization of the weights (that will be later used in "step")
		self.normalize_weights()

		# -------------------------

		# we compute the mean...
		mean = (self.weights[np.newaxis, :] * self.samples).sum(axis=1)

		# ...and (weighted) covariance
		covariance = np.cov(self.samples, ddof=0, aweights=self.weights)

		# the inverse of the covariance
		inv_covariance = np.linalg.inv(covariance)

		# Q and nu are updated
		self._Q, self._nu = inv_covariance, inv_covariance @ mean

		# import code
		# code.interact(local=dict(globals(), **locals()))
