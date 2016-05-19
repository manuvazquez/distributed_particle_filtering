import sys
import numpy as np
import scipy.stats

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
		self._state = self._state_transition_kernel.next_state(self._state)

		# TODO: this may cause a "divide by zero" warning when a likelihood is very small
		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.to_position(self._state)) for sensor, obs in zip(self._sensors, observations)]))

		# for each particle, we compute the product of the likelihoods for all the sensors
		loglikelihoods_product = loglikelihoods.sum(axis=0)

		# the weights are updated
		self._log_weights += loglikelihoods_product

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
			M, R_p, r_a_tuples, r_a, r_d_tuples, r_d, rs_gamma):

		super().__init__(n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors)

		# the noise covariance matrix is built from the individual variances of each sensor
		self._noiseCovariance = np.diag([s.noiseVar for s in sensors])

		# the position of every sensor
		self._sensorsPositions = np.hstack([s.position for s in sensors])

		# M, R_p, r_a_tuples, r_a, r_d_tuples, r_d, rs_gamma

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
		self._state = self._state_transition_kernel.next_state(self._state)

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
			initial_size_estimate, aggregated_weight=1.0):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight)

		self.estimated_n_PEs = initial_size_estimate

		# they will be set later
		self._Q = None
		self._nu = None

		# value of the elements in the diagonal of a covariance matrix that became 0
		self._default_variance = 20

		self._fake_random_state = self.FakeRandomState()

	class FakeRandomState:

		def __init__(self):
			pass

		def normal(self, mean, variance, size):
			return np.zeros(size)

	def step(self, observations):

		assert len(observations) == len(self._sensors)

		# the particles are propagated with a *fake* RandomState that always returns 0's
		predictions = self._state_transition_kernel.next_state(self._state)

		# for each sensor, we compute the likelihood of EVERY predicted particle (position)
		predictions_likelihoods = np.array(
			[sensor.likelihood(obs, state.to_position(predictions)) for sensor, obs in
			 zip(self._sensors, observations)])

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# + 1e-200 in order to avoid division by zero
		predictions_likelihoods_product = predictions_likelihoods.prod(axis=0) + 1e-200

		sampling_weights = predictions_likelihoods_product / predictions_likelihoods_product.sum()

		i_particles_resampled = self._resampling_algorithm.get_indexes(sampling_weights)

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# every particle is updated (previous state is not stored...)
		self._state = self._state_transition_kernel.next_state(self._state[:, i_particles_resampled])

		# TODO: this may cause a "divide by zero" warning when a likelihood is very small
		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.to_position(self._state)) for sensor, obs in
			 zip(self._sensors, observations)]))

		# for each particle, we compute the product of the likelihoods for all the sensors
		loglikelihoods_product = loglikelihoods.sum(axis=0)

		# every likelihood is exponentiated by the estimate of the number of PEs
		self._log_weights = loglikelihoods_product*self.estimated_n_PEs - np.log(predictions_likelihoods_product[i_particles_resampled])

		# the aggregated weight is kept up to date at all times
		self.update_aggregated_weight()

		# normalization of the weights (that will be later used in "step")
		self.normalize_weights()

		# -------------------------

		# we compute the mean...
		mean = (self.weights[np.newaxis, :] * self.samples).sum(axis=1)

		# ...and (weighted) covariance
		covariance = np.cov(self.samples, ddof=0, aweights=self.weights)

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# if the matrix is singular (this happens when a single particle accumulates most of the weight)
		if np.isnan(np.linalg.cond(covariance)) or  np.linalg.cond(covariance) > (1 / sys.float_info.epsilon):

			# the (weighted) covariance matrix being zero does NOT mean there is no uncertainty about the random vector.
			# On the contrary, it *most likely* means that ALL of the likelihoods are really small, which results in a
			# single (arbitrary, due to numeric precision) weight concentrating all the mass (corresponding to the
			# biggest likelihood that may be 10^-12...)
			covariance += np.identity(state.n_elements)*self._default_variance

		# we try...
		try:

			# ...to invert the covariance matrix
			inv_covariance = np.linalg.inv(covariance)

		# it it's not possible (singular)...
		except np.linalg.linalg.LinAlgError:

			import pdb; pdb.set_trace()

		# Q and nu are updated
		self._Q, self._nu = inv_covariance, inv_covariance @ mean

# =========================================================================================================


class TargetTrackingGaussianMixtureParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight=1.0):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			aggregated_weight)

		self.previous_weights = None

	# the same as in the parent class with no resampling
	def avoid_weight_degeneracy(self):

		self.normalize_weights()

	def step(self, observations):

		self.previous_weights = self.weights

		super().step(observations)

# =========================================================================================================


class TargetTrackingSetMembershipConstrainedParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors, L,
			alpha, beta, n_repeats_rejection_sampling, RS_PRNG):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors)

		self._L = L
		self._alpha = alpha
		self._beta = beta
		self._n_repeats_rejection_sampling = n_repeats_rejection_sampling
		self._RS_PRNG = RS_PRNG

		self.bounding_box_min = None
		self.bounding_box_max = None
		self.loglikelihoods = None
		self.norm_constants = None

		self._alpha_beta_ratio = self._beta/self._alpha

	def belongs(self, samples):

		return np.logical_and(
			(samples.T > self.bounding_box_min).all(axis=1),
			(samples.T < self.bounding_box_max).all(axis=1)
		)

	def step(self, observations):

		# oversampling (of the previous probability measure)
		i_oversampled_particles = self._resampling_algorithm.get_indexes(self.weights, self.n_particles*self._L)

		# oversampled particles are updated
		oversampled_particles = self._state_transition_kernel.next_state(self._state[:, i_oversampled_particles])

		# for each sensor, we compute the likelihood of EVERY particle (position)
		likelihoods = np.prod([sensor.likelihood(obs, state.to_position(oversampled_particles))
		                       for sensor, obs in zip(self._sensors, observations)], axis=0)

		# in order to avoid dividing by zero
		likelihoods += 1e-200

		# normalization
		weights = likelihoods/likelihoods.sum()

		# indexes of the particles...
		i_sampled_particles = self._resampling_algorithm.get_indexes(weights, self.n_particles)

		# ...that determine the bounding box
		particles_bounding_box = oversampled_particles[:, i_sampled_particles]

		self.bounding_box_min = particles_bounding_box.min(axis=1)
		self.bounding_box_max = particles_bounding_box.max(axis=1)

	def rejection_sampling(self, samples):

		# in order to avoid loops, each particle is replicated as many times as
		n_replicas = self._n_repeats_rejection_sampling

		# all the replicas of all the particles are propagated
		n_fold_propagated_samples = self._state_transition_kernel.next_state(np.repeat(samples, n_replicas, axis=1))

		# are particles within bounds?
		within_bounds = self.belongs(n_fold_propagated_samples).reshape(-1, n_replicas)

		# there is a small chance they will be accepted anyway
		accept_anyway = scipy.stats.bernoulli.rvs(
			self._alpha_beta_ratio, size=(self.n_particles, n_replicas), random_state=self._RS_PRNG).astype(np.bool)

		# the index for the first valid replica of each particle
		i_first_valid = [np.where(sample_fold)[0][0] for sample_fold in (within_bounds | accept_anyway)]

		# we need to choose the right particle from the manifold of replicas
		i_valid = np.array(i_first_valid) + n_replicas*np.arange(self.n_particles)

		propagated_particles = n_fold_propagated_samples[:, i_valid]

		# a rough way of computing the normalization constant for every particle
		# ===========================

		# for every particle, (an approximation of) the probability of drawing a sample within the region
		prob_within = within_bounds.sum(axis=1)/n_replicas

		# for every particle, (an approximation of) the probability of drawing a sample outside the region, and accepting
		# it anyway
		# prob_outside_but_accepted = (~within_bounds & accept_anyway).sum(axis=1)/n_replicas

		# another (not exactly equivalent) way; this should be better because the above probability will usually be high,
		# and hence easier to estimate
		prob_outside_but_accepted = (1-prob_within)*self._beta

		# the (approximation of the) normalization constant is the sum of the two of them
		norm_constants = self._alpha*prob_within + prob_outside_but_accepted

		return propagated_particles, norm_constants

	def actual_sampling_step(self, observations):

		# resampling of the particles from the previous time instant (with a RandomState object that should be
		# synchronized across different PEs
		i_sampled_particles = self._resampling_algorithm.get_indexes(self.weights)

		# ...the resulting particles
		self._state, self.norm_constants = self.rejection_sampling(self._state[:, i_sampled_particles])

		likelihoods = np.prod([sensor.likelihood(obs, state.to_position(self._state))
		                       for sensor, obs in zip(self._sensors, observations)], axis=0)

		# in order to avoid numerical precision problems
		likelihoods += 1e-200

		self.loglikelihoods = np.log(likelihoods)

	def weight_update_step(self):

		# particles are in the right region?
		belongs =self.belongs(self._state)

		# the last term in the weights update equation
		tentative_term = np.log(self._alpha * belongs.astype(float) + self._beta * (~belongs).astype(float))

		# the new log-weights
		self.log_weights = self.loglikelihoods + np.log(self.norm_constants) - tentative_term

		# the aggregated weight is updated...
		self.update_aggregated_weight()

		# ...to perform normalization
		self.normalize_weights()


# =========================================================================================================


class TargetTrackingSelectiveGossipParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			n_PEs):

		super().__init__(n_particles, resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors)

		self._n_PEs = n_PEs

		# for later use
		self.gamma = None
		self.ro = None

	def step(self, observations):

		# every particle is propagated for computing the first-stage weights
		auxiliar_state = self._state_transition_kernel.next_state(self._state)

		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.to_position(auxiliar_state)) for sensor, obs in zip(self._sensors, observations)]))

		# for each particle we store the product of the likelihoods for all the sensors multiplied by the number of PEs
		self.gamma = self._n_PEs*loglikelihoods.sum(axis=0)

	def actual_sampling_step(self, observations):

		ro = np.exp(self.gamma_postgossip)

		# the significant samples according to the consensus algorithm...
		significant_samples = self._state[:, self.i_significant]

		# ...and their corresponding weights, which are at the same time updated using the first-stage weights
		updated_significant_weights = self.weights[self.i_significant]*ro

		# in order to avoid numerical problems...
		updated_significant_weights += 1e-200

		# ...when normalizing
		updated_significant_weights /= updated_significant_weights.sum()

		# resampling of the particles from the previous step according to the first-stage weights
		i_resampled = self._resampling_algorithm.get_indexes(updated_significant_weights, self.n_particles)
		resampled = significant_samples[:, i_resampled]
		self.ro = ro[i_resampled]

		self._state = self._state_transition_kernel.next_state(resampled)

		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.to_position(self._state)) for sensor, obs in zip(self._sensors, observations)]))

		# for each particle we store the product of the likelihoods for all the sensors multiplied by the number of PEs
		self.gamma = self._n_PEs*loglikelihoods.sum(axis=0)

	def weights_update_step(self):

		state = self._state[:, self.i_significant]
		weights = np.exp(self.gamma_postgossip)/self.ro[self.i_significant]

		weights /= weights.sum()

		# we resample from the set of *significant* particles
		i_resampling = self._resampling_algorithm.get_indexes(weights, self.n_particles)

		self._state = state[:, i_resampling]
		self.weights = np.full(self.n_particles, 1/self.n_particles)
