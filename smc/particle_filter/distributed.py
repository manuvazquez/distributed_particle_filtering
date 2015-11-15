import numpy as np
import scipy.misc
import itertools

# this is required (due to a bug?) for import rpy2
import readline

import rpy2.robjects as robjects

# in order to load an R package
from rpy2.robjects.packages import importr

# for automatic conversion from numpy arrays to R data types
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import smc.estimator
from .particle_filter import ParticleFilter
from . import centralized


class TargetTrackingParticleFilter(ParticleFilter):

	def __init__(
			self, n_processing_elements, n_particles_per_processing_element, resampling_algorithm, resampling_criterion,
			prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			particle_filters_class=centralized.TargetTrackingParticleFilter, particle_filters_initial_aggregated_weight=1.0):

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
					n_messages += distances[i_PE, i_observation_to_i_processing_element[i]]

		return n_messages

# =========================================================================================================


class LikelihoodConsensusTargetTrackingParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, exchange_recipe, n_processing_elements, n_particles_per_processing_element, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			polynomial_degree, particle_filters_class=centralized.TargetTrackingParticleFilterWithConsensusCapabilities):

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
			lambda x: sum(x) > (2*self._R_p), itertools.product(range(2*self._R_p+1), repeat=self._M)))
		self._r_d = np.array(self._r_d_tuples)

		# we generate the *two* vectors of exponents (r' and r'' in the paper) jointly,
		# and then drop those combinations that don't satisfy the required constraints
		self._rs_gamma = [list(itertools.filterfalse(
			lambda x:
			not np.allclose((np.array(x)[:self._M] + x[self._M:]), r) or sum(x[:self._M]) > self._R_p or sum(x[self._M:]) > self._R_p,
			itertools.product(range(self._R_p+1), repeat=2*self._M))) for r in self._r_d]

		# theoretically, this is the number of beta components that should result
		N_c = scipy.misc.comb(2*self._R_p + self._M, 2*self._R_p, exact=True)

		assert(N_c == len(self._r_d_tuples))

	def initialize(self):

		super().initialize()

		# the constant values required by every PE to carry out the polynomial approximation are passed to each PE
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
		assert messages_observations_propagation == 0

		return messages_observations_propagation + self.exchange_recipe.messages()


# =========================================================================================================


class TargetTrackingParticleFilterWithDRNA(TargetTrackingParticleFilter):

	def __init__(
			self, exchange_period, exchange_recipe, n_particles_per_processing_element, normalization_period,
			resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			each_processing_element_required_sensors, particle_filters_class=centralized.EmbeddedTargetTrackingParticleFilter):

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

class TargetTrackingParticleFilterWithMposterior(TargetTrackingParticleFilter):

	def __init__(
			self, exchange_recipe, n_particles_per_processing_element, resampling_algorithm, resampling_criterion,
			prior, state_transition_kernel, sensors, each_processing_element_required_sensors,
			find_weiszfeld_median_parameters, sharing_period,
			particle_filters_class=centralized.TargetTrackingParticleFilter):

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

		# no messages should be used in this algorithm to transmit observation between PEs
		assert messages_observations_propagation==0

		return messages_observations_propagation + self.exchange_recipe.messages()/self._sharingPeriod