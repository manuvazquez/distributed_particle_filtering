import numpy as np
import abc
import itertools
import scipy.misc

import state
import smc.estimator

# this is required (due to a bug?) for import rpy2
import readline

import rpy2.robjects as robjects

# in order to load an R package
from rpy2.robjects.packages import importr

# for automatic conversion from numpy arrays to R data types
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


class ParticleFilter(metaclass=abc.ABCMeta):
	
	def __init__(self, n_particles, resampling_algorithm, resampling_criterion):
		
		self._nParticles = n_particles
		
		self._resamplingAlgorithm = resampling_algorithm
		self._resamplingCriterion = resampling_criterion
	
	@abc.abstractmethod
	def initialize(self):

		pass
	
	@abc.abstractmethod
	def step(self, observations):
		
		pass
	
	@abc.abstractmethod
	def get_state(self):
		
		pass

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		# to indicate it has not been computed
		return -1

# =========================================================================================================


class CentralizedTargetTrackingParticleFilter(ParticleFilter):
	
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
		self._state = self._prior.sample(self._nParticles)
		
		# the weights are assigned equal probabilities
		self._log_weights.fill(np.log(self._initial_aggregated_weight)-np.log(self._nParticles))

		# this variable just keeps tabs on the sum of all the weights
		self._aggregated_weight = self._initial_aggregated_weight

	def step(self, observations):
		
		assert len(observations) == len(self._sensors)
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._state_transition_kernel.nextState(self._state[:, i:i+1]) for i in range(self._nParticles)])
		
		# TODO: this may cause a "divide by zero" warning when a likelihood is very small
		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array(
			[sensor.likelihood(obs, state.position(self._state)) for sensor, obs in zip(self._sensors, observations)]))
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		log_likelihoods_product = loglikelihoods.sum(axis=0)
		
		# the weights are updated
		self._log_weights += log_likelihoods_product
		
		# the aggregated weight is kept up to date at all times
		self.update_aggregated_weight()
		
		# whatever is required (it depends on the algorithm) to avoid weights degeneracy...
		self.avoid_weight_degeneracy()

	def get_state(self):
		
		return self._state

	def resample(self, normalized_log_weights):
		
		# the weights need to be converted to "natural" units
		normalized_weights = np.exp(normalized_log_weights)
		
		# we check whether a resampling step is actually needed or not
		if self._resamplingCriterion.isResamplingNeeded(normalized_weights):
			
			try:
				# the resampling algorithm is used to decide which particles to keep
				i_particles_to_be_kept = self._resamplingAlgorithm.getIndexes(normalized_weights)
				
			except ValueError:
				
				# this should mean the normalized weights don't add up EXACTLY to one...we renormalize them...
				normalized_weights /= normalized_weights.sum()
				
				# ...and try again
				i_particles_to_be_kept = self._resamplingAlgorithm.getIndexes(normalized_weights)
				
			# the above indexes are used to update the state
			self._state = self._state[:, i_particles_to_be_kept]
			
			# note that if the weights have been normalized ("standard" centralized particle filter),
			# then "self._aggregated_weight" is equal to 1
			self._log_weights.fill(np.log(self._aggregated_weight)-np.log(self._nParticles))
		
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
	def n_particles(self):

		return len(self._log_weights)

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
			# return np.zeros((state.nElements, 1))
			return np.full((state.nElements, 1), np.pi)
		
		normalized_log_weights = self._log_weights - np.log(self._aggregated_weight)

		# element-wise multiplication of the state vectors and their correspondent weights,
		# followed by addition => weighted mean
		return (self._state*np.exp(normalized_log_weights)[np.newaxis, :]).sum(axis=1)[:, np.newaxis]

	# this methods encapsulates the parts within the code of "step" which are different in this class and its children
	def avoid_weight_degeneracy(self):
		
		# if all the weights are zero...
		if self._aggregated_weight == 0:
			
			# ...then normalization makes no sense and we just initialize the weights again
			self._log_weights.fill(-np.log(self._nParticles))

		else:
		
			self._log_weights -= np.log(self._aggregated_weight)
			
		# we forced this above
		self._aggregated_weight = 1.0
		
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
			
# =========================================================================================================


class EmbeddedTargetTrackingParticleFilter(CentralizedTargetTrackingParticleFilter):

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


class CentralizedTargetTrackingParticleFilterWithConsensusCapabilities(CentralizedTargetTrackingParticleFilter):
	
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
		x = state.position(self._state)
		
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
			[self._state_transition_kernel.nextState(self._state[:, i:i+1]) for i in range(self._nParticles)])
		
		self.polynomial_approximation(observations)

	def polynomial_approximation(self, observations):
		
		x = state.position(self._state)
		
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


class DistributedTargetTrackingParticleFilter(ParticleFilter):
	
	def __init__(
			self, n_processing_elements, n_particles_per_processing_element, resampling_algorithm, resampling_criterion,
			prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			particle_filters_class=CentralizedTargetTrackingParticleFilter, particle_filters_initial_aggregated_weight=1.0):
		
		super().__init__(n_processing_elements*n_particles_per_processing_element, resampling_algorithm, resampling_criterion)
		
		# it is handy to keep the number of PEs in a variable...
		self._nPEs = n_processing_elements

		# ...and the same for the overall number of sensors
		self._n_sensors = len(sensors)
		
		# a list of lists, the first one containing the indices of the sensors "seen" by the first PE...and so on
		self._each_processing_element_required_sensors = each_processing_element_required_sensors

		# number of particles per Pe
		self._nParticlesPerPE = n_particles_per_processing_element
		
		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [particle_filters_class(
			n_particles_per_processing_element, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			[sensors[iSensor] for iSensor in connections], aggregated_weight=particle_filters_initial_aggregated_weight
		) for connections in each_processing_element_required_sensors]

	@property
	def n_PEs(self):

		return self._nPEs

	def initialize(self):
		
		# all the PFs are initialized
		for PE in self._PEs:
			
			PE.initialize()

		# we keep track of the elapsed (discreet) time instants
		self._n = 0

	def step(self, observations):
		
		# a step is taken in every PF (ideally, this would occur concurrently); notice that every PE always accesses the
		# sensors it needs (whatever the cost in communication messages)
		for PE, sensorsConnections in zip(self._PEs, self._each_processing_element_required_sensors):
			
			# only the appropriate observations are passed to this PE
			# NOTE: it is assumed that the order in which the observations are passed is the same as that of the sensors
			# when building the PF
			PE.step(observations[sensorsConnections])
			
		# a new time instant has elapsed
		self._n += 1
	
	def get_state(self):
		
		# the state from every PE is gathered together
		return np.hstack([PE.get_state() for PE in self._PEs])

	def messages_observations_propagation(self, processing_elements_topology, each_processing_element_connected_sensors):

		i_observation_to_i_processing_element = np.empty(self._n_sensors)

		# in order to find out to which PE is associated each observation
		for i_PE, i_sensors in enumerate(each_processing_element_connected_sensors):

			for i in i_sensors:

				i_observation_to_i_processing_element[i] = i_PE

		# the distance in hops between each pair of PEs
		distances = processing_elements_topology.distances_between_processing_elements

		n_messages = 0

		# we loop through the observations "required" by each PE
		for i_PE, i_sensors in enumerate(self._each_processing_element_required_sensors):

			# for every observation required by the PE...
			for i in i_sensors:

				# ...if it doesn't have access to it...
				if i not in each_processing_element_connected_sensors[i_PE]:

					# ...it must be sent from the corresponding PE
					n_messages += distances[i_PE,i_observation_to_i_processing_element[i]]

		return n_messages

# =========================================================================================================


class LikelihoodConsensusDistributedTargetTrackingParticleFilter(DistributedTargetTrackingParticleFilter):
	
	def __init__(
			self, exchange_recipe, n_processing_elements, n_particles_per_processing_element, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			polynomial_degree, particle_filters_class=CentralizedTargetTrackingParticleFilterWithConsensusCapabilities):
		
		super().__init__(
			n_processing_elements, n_particles_per_processing_element, resampling_algorithm, resampling_criterion, prior,
			state_transition_kernel, sensors, each_processing_element_required_sensors,
			particle_filters_class=particle_filters_class)
		
		# the exchange recipe is kept
		self.exchange_recipe = exchange_recipe
		
		# for the sake of convenience when following the pseudocode in
		# "Likelihood Consensus and its Application to Distributed Particle Filtering":
		# -------------------
		
		# the size of the state
		self._M = 2
		
		# the chosen degree for the polynomial approximation
		self._R_p = polynomial_degree
		
		# a list of tuples, each one representing a combination of possible exponents for a monomial
		self._r_a_tuples = list(itertools.filterfalse(
			lambda x: sum(x) > self._R_p, itertools.product(range(self._R_p+1), repeat=self._M)))
		
		# each row contains the exponents for a monomial
		self._r_a = np.array(self._r_a_tuples)
		
		# theoretical number of monomials in the approximation
		R_a = scipy.misc.comb(self._R_p + self._M, self._R_p, exact=True)
		
		assert(R_a == len(self._r_a_tuples))
		
		# exponents for computing d
		self._r_d_tuples = list(itertools.filterfalse(
			lambda x: sum(x)>(2*self._R_p), itertools.product(range(2*self._R_p+1), repeat=self._M)))
		self._r_d = np.array(self._r_d_tuples)
		
		# we generate the *two* vectors of exponents (r' and r'' in the paper) jointly,
		# and then drop those combinations that don't satisfy the required constraints
		self._rs_gamma = [list(itertools.filterfalse(
			lambda x:
			not np.allclose((np.array(x)[:self._M] + x[self._M:]),r) or sum(x[:self._M]) > self._R_p or sum(x[self._M:]) > self._R_p,
			itertools.product(range(self._R_p+1), repeat=2*self._M))) for r in self._r_d]
	
		# theoretically, this is the number of beta components that should result
		N_c = scipy.misc.comb(2*self._R_p + self._M, 2*self._R_p, exact=True)
		
		assert(N_c == len(self._r_d_tuples))

	def initialize(self):

		super().initialize()
		
		# the constant values required by every PE to carry out the polynomial approximation ared passed to each PE
		for PE in self._PEs:
			
			PE.set_polynomial_approximation_constants(
				self._M, self._R_p, self._r_a_tuples, self._r_a, self._r_d_tuples, self._r_d, self._rs_gamma)

	def step(self, observations):
		
		# each PE initializes its local state
		for PE, sensors_connections in zip(self._PEs, self._each_processing_element_required_sensors):
			
			PE.pre_consensus_step(observations[sensors_connections])
		
		# consensus
		self.exchange_recipe.perform_exchange(self)
		
		# a step is taken in every PF (ideally, this would occur concurrently)
		for PE, sensors_connections in zip(self._PEs, self._each_processing_element_required_sensors):
			
			# only the appropriate observations are passed to this PE. Note that it is assumed that the order in which
			# the observations are passed is the same as that of the sensors when building the PF
			PE.step(observations[sensors_connections])

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		messages_observations_propagation = super().messages_observations_propagation(
			processing_elements_topology, each_processing_element_connected_sensors)

		# in LC-based DPF, observations are not transmitted between PEs
		assert(messages_observations_propagation) == 0

		return messages_observations_propagation + self.exchange_recipe.messages()


# =========================================================================================================


class TargetTrackingParticleFilterWithDRNA(DistributedTargetTrackingParticleFilter):
	
	def __init__(
			self, exchange_period, exchange_recipe, n_particles_per_processing_element, normalization_period,
			resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			each_processing_element_required_sensors, particle_filters_class=EmbeddedTargetTrackingParticleFilter):
		
		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_processing_element, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			particle_filters_class=particle_filters_class,
			particle_filters_initial_aggregated_weight=1.0/exchange_recipe.n_processing_elements)
		
		# a exchange of particles among PEs will happen every...
		self._exchange_period = exchange_period

		# ...and this object is responsible
		self.exchange_recipe = exchange_recipe

		# period for the normalization of the aggregated weights
		self._normalization_period = normalization_period
		
		self._estimator = smc.estimator.WeightedMean(self)
		
	def step(self, observations):
		
		super().step(observations)
		
		# if it is exchanging particles time
		if self._n % self._exchange_period == 0:
			
			self.exchange_recipe.perform_exchange(self)
			
			# after the exchange, the aggregated weight of every PE must be updated
			for PE in self._PEs:
				
				PE.update_aggregated_weight()
		
		# needed to perform the normalization below
		aggregated_weights_sum = self.aggregated_weights.sum()
		
		# if every aggregated weight is zero...
		if np.isclose(aggregated_weights_sum, 0):
			
			# ...we reinitialize the weights for all the particles of all the PEs
			self.reset_weights()
			
			# ...and skip the normalization code below
			return
		
		# the aggregated weights must be normalized every now and then to avoid computer precision issues
		if self._n % self._normalization_period == 0:
			
			# ...to scale all the weights within ALL the PEs
			for PE in self._PEs:
				
				PE.divide_weights(aggregated_weights_sum)

	@property
	def aggregated_weights(self):
		
		return np.array([PE.aggregated_weight for PE in self._PEs])

	def reset_weights(self):
		
		"""It sets every weight of every PE to the same value.
		"""
		
		# every PE will be assigned the same aggregated weight:
		aggregated_weight = 1.0/self._nPEs
		
		# for every PE in this DPF...
		for PE in self._PEs:
			
			# the aggregated weight is set...
			PE._aggregated_weight = aggregated_weight
			
			# ...along with the individual weights within the PE
			PE.log_weights = np.full(PE._nParticles, -np.log(self._nPEs)-np.log(PE._nParticles))

	def compute_mean(self):
		
		return self._estimator.estimate()

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		messages_observations_propagation = super().messages_observations_propagation(
			processing_elements_topology, each_processing_element_connected_sensors)

		return messages_observations_propagation + self.exchange_recipe.messages()/self._exchange_period


# =========================================================================================================

class DistributedTargetTrackingParticleFilterWithMposterior(DistributedTargetTrackingParticleFilter):

	def __init__(
			self, exchange_recipe, n_particles_per_processing_element, resampling_algorithm, resampling_criterion,
			prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			find_weiszfeld_median_parameters, sharing_period,
			particle_filters_class=CentralizedTargetTrackingParticleFilter):
		
		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_processing_element, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			particle_filters_class=particle_filters_class)
		
		self._sharingPeriod = sharing_period
		self.exchange_recipe = exchange_recipe
		
		# the (R) Mposterior package is imported...
		self._Mposterior = importr('Mposterior')
		
		# ...and the parameters to be passed to the required function are kept
		self._findWeiszfeldMedianParameters = find_weiszfeld_median_parameters
		
	def Mposterior(self, posterior_distributions):
		
		"""Applies the Mposterior algorithm to weight the samples of a list of "subset posterior distribution"s.
		
		Parameters
		----------
		posterior_distributions: list of tuples
			A list in which each element is a tuple representing a "subset posterior distribution": the first element is
			the samples, and the second the associated weights
		
		Returns
		-------
		samples: tuple
			The first element is a 2-D ndarray with all the samples, and the second the corresponding weights.
		"""
		
		# the samples of all the "subset posterior distribution"s are extracted
		samples = [posterior[0] for posterior in posterior_distributions]
		
		# an R function implementing the "M posterior" algorithm is called
		weiszfeld_median = self._Mposterior.findWeiszfeldMedian(samples, **self._findWeiszfeldMedianParameters)

		# the weights assigned by the algorithm to each "subset posterior distribution"
		weiszfeld_weights = np.array(weiszfeld_median[1])
		
		# a numpy array containing all the particles (coming from all the PEs)
		joint_particles = np.array(weiszfeld_median[3]).T
		
		# the weight of each PE is scaled according to the "Weiszfeld_weights" and, all of them are stacked together
		joint_weights =	np.hstack([posterior[1]*weight for posterior, weight in zip(posterior_distributions, weiszfeld_weights)])
		
		return joint_particles, joint_weights

	def step(self, observations):
		
		super().step(observations)
		
		# if it is sharing particles time
		if self._n % self._sharingPeriod == 0:
			
			self.exchange_recipe.perform_exchange(self)

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		messages_observations_propagation = super().messages_observations_propagation(
			processing_elements_topology, each_processing_element_connected_sensors)

		return messages_observations_propagation + self.exchange_recipe.messages()/self._sharingPeriod


class CentralizedTargetTrackingParticleFilterWithFusionCenter(CentralizedTargetTrackingParticleFilter):

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

