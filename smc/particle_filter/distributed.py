import numpy as np
import scipy.misc
import itertools
import abc
import copy

import smc.estimator
from .particle_filter import ParticleFilter
from . import centralized


class TargetTrackingParticleFilter(ParticleFilter, metaclass=abc.ABCMeta):

	def __init__(
			self, n_PEs, n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			sensors, each_PE_required_sensors, pf_initial_aggregated_weight=1.0):

		super().__init__(n_PEs * n_particles_per_PE, resampling_algorithm, resampling_criterion)

		# it is handy to keep the number of PEs in a variable...
		self._n_PEs = n_PEs

		# ...and the same for the overall number of sensors
		self._n_sensors = len(sensors)

		# a list of lists, the first one containing the indices of the sensors "seen" by the first PE...and so on
		self._each_PE_required_sensors = each_PE_required_sensors

	@property
	def n_PEs(self):

		return self._n_PEs

	@property
	def PEs(self):

		return self._PEs

	def initialize(self):

		# all the PFs are initialized
		for PE in self._PEs:

			PE.initialize()

		# we keep track of the elapsed (discreet) time instants
		self._n = 0

	def step(self, observations):

		# a step is taken in every PF (ideally, this would occur concurrently); notice that every PE always accesses the
		# sensors it needs (whatever the cost in communication messages)
		for PE, sensors_connections in zip(self._PEs, self._each_PE_required_sensors):

			# only the appropriate observations are passed to this PE
			# NOTE: it is assumed that the order in which the observations are passed is the same as that of the sensors
			# when building the PF
			PE.step(observations[sensors_connections])

		# a new time instant has elapsed
		self._n += 1

	def get_state(self):

		# the state from every PE is gathered together
		return np.hstack([PE.get_state() for PE in self._PEs])

	def messages_observations_propagation(self, PEs_topology, each_processing_element_connected_sensors):

		i_observation_to_i_processing_element = np.empty(self._n_sensors)

		# in order to find out to which PE is associated each observation
		for i_PE, i_sensors in enumerate(each_processing_element_connected_sensors):

			for i in i_sensors:

				i_observation_to_i_processing_element[i] = i_PE

		# the distance in hops between each pair of PEs
		distances = PEs_topology.distances_between_processing_elements

		n_messages = 0

		# we loop through the observations "required" by each PE
		for i_PE, i_sensors in enumerate(self._each_PE_required_sensors):

			# for every observation required by the PE...
			for i in i_sensors:

				# ...if it doesn't have access to it...
				if i not in each_processing_element_connected_sensors[i_PE]:

					# ...it must be sent from the corresponding PE
					n_messages += distances[i_PE, i_observation_to_i_processing_element[i]]

		return n_messages

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		return np.NaN

# =========================================================================================================


class LikelihoodConsensusTargetTrackingParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, exchange_recipe, n_PEs, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors,
			polynomial_degree):

		super().__init__(
			n_PEs, n_particles_per_PE, resampling_algorithm, resampling_criterion, prior,
			state_transition_kernel, sensors, each_PE_required_sensors)

		# the exchange recipe is kept
		self.exchange_recipe = exchange_recipe

		# for the sake of convenience when following the pseudocode in
		# "Likelihood Consensus and its Application to Distributed Particle Filtering":
		# -------------------

		# the size of the state
		M = 2

		# the chosen degree for the polynomial approximation
		R_p = polynomial_degree

		# a list of tuples, each one representing a combination of possible exponents for a monomial
		r_a_tuples = list(itertools.filterfalse(
			lambda x: sum(x) > R_p, itertools.product(range(R_p+1), repeat=M)))

		# each row contains the exponents for a monomial
		r_a = np.array(r_a_tuples)

		# theoretical number of monomials in the approximation
		R_a = scipy.misc.comb(R_p + M, R_p, exact=True)

		assert(R_a == len(r_a_tuples))

		# exponents for computing d
		self._r_d_tuples = list(itertools.filterfalse(
			lambda x: sum(x) > (2*R_p), itertools.product(range(2*R_p+1), repeat=M)))
		r_d = np.array(self._r_d_tuples)

		# we generate the *two* vectors of exponents (r' and r'' in the paper) jointly,
		# and then drop those combinations that don't satisfy the required constraints
		rs_gamma = [list(itertools.filterfalse(
			lambda x:
			not np.allclose((np.array(x)[:M] + x[M:]), r) or sum(x[:M]) > R_p or sum(x[M:]) > R_p,
			itertools.product(range(R_p+1), repeat=2*M))) for r in r_d]

		# theoretically, this is the number of beta components that should result
		N_c = scipy.misc.comb(2*R_p + M, 2*R_p, exact=True)

		assert(N_c == len(self._r_d_tuples))

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.TargetTrackingParticleFilterWithConsensusCapabilities(
			n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			[sensors[iSensor] for iSensor in connections], M, R_p, r_a_tuples, r_a, self._r_d_tuples, r_d, rs_gamma
		) for connections in each_PE_required_sensors]
		
	def step(self, observations):

		# each PE initializes its local state
		for PE, sensors_connections in zip(self._PEs, self._each_PE_required_sensors):

			PE.pre_consensus_step(observations[sensors_connections])

		# consensus
		self.exchange_recipe.perform_exchange(self)

		# a step is taken in every PF (ideally, this would occur concurrently)
		for PE, sensors_connections in zip(self._PEs, self._each_PE_required_sensors):

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
			self, exchange_period, exchange_recipe, n_particles_per_PE, normalization_period,
			resampling_algorithm, resampling_criterion, prior, state_transition_kernel, sensors,
			each_PE_required_sensors):

		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors,
			pf_initial_aggregated_weight=1.0 / exchange_recipe.n_processing_elements)

		# a exchange of particles among PEs will happen every...
		self._exchange_period = exchange_period

		# ...and this object is responsible
		self.exchange_recipe = exchange_recipe

		# period for the normalization of the aggregated weights
		self._normalization_period = normalization_period

		self._estimator = smc.estimator.WeightedMean(self)

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.EmbeddedTargetTrackingParticleFilter(
			n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			[sensors[iSensor] for iSensor in connections], aggregated_weight=1.0 / exchange_recipe.n_processing_elements
		) for connections in each_PE_required_sensors]

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
		aggregated_weight = 1.0/self._n_PEs

		# for every PE in this DPF...
		for PE in self._PEs:

			# the aggregated weight is set...
			PE._aggregated_weight = aggregated_weight

			# ...along with the individual weights within the PE
			PE.log_weights = np.full(PE._n_particles, -np.log(self._n_PEs) - np.log(PE._n_particles))

	def compute_mean(self):

		return self._estimator.estimate()

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		messages_observations_propagation = super().messages_observations_propagation(
			processing_elements_topology, each_processing_element_connected_sensors)

		return messages_observations_propagation + self.exchange_recipe.messages()/self._exchange_period


# =========================================================================================================

class TargetTrackingParticleFilterWithMposterior(TargetTrackingParticleFilter):

	def __init__(
			self, exchange_recipe, n_particles_per_PE, resampling_algorithm, resampling_criterion,
			prior, state_transition_kernel, sensors, each_PE_required_sensors, sharing_period):

		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors)

		self._sharing_period = sharing_period
		self.exchange_recipe = exchange_recipe

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.TargetTrackingParticleFilter(
			n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			[sensors[iSensor] for iSensor in connections]
		) for connections in each_PE_required_sensors]

	def step(self, observations):

		super().step(observations)

		# if it is sharing particles time
		if self._n % self._sharing_period == 0:

			self.exchange_recipe.perform_exchange(self)

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		messages_observations_propagation = super().messages_observations_propagation(
			processing_elements_topology, each_processing_element_connected_sensors)

		# no messages should be used in this algorithm to transmit observation between PEs
		assert messages_observations_propagation==0

		return messages_observations_propagation + self.exchange_recipe.messages()/self._sharing_period

# =========================================================================================================


class TargetTrackingGaussianParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			sensors, each_PE_required_sensors, exchange_recipe, ad_hoc_parameters, PRNG):

		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors)

		self.exchange_recipe = exchange_recipe
		self._PRNG = PRNG

		# if no initial estimate is passed through the parameters file...
		if "initial_size_estimate" not in ad_hoc_parameters:

			# ...the initial estimate is assumed to be half the number of PEs
			initial_size_estimate = self._n_PEs//2

		else:

			initial_size_estimate = ad_hoc_parameters["initial_size_estimate"]

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.TargetTrackingGaussianParticleFilter(
			n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			[sensors[iSensor] for iSensor in connections], initial_size_estimate
		) for connections in each_PE_required_sensors]

		# self._n_particles_per_PE = n_particles_per_PE

	def initialize(self):

		super().initialize()

		# the size is estimated only once at the beginning
		self.exchange_recipe.size_estimation(self)

	def step(self, observations):

		super().step(observations)

		# Q = np.stack([PE._Q for PE in self.PEs], axis=2).mean(axis=2)
		# nu = np.stack([PE._nu for PE in self.PEs], axis=0).mean(axis=0)

		# import code
		# code.interact(local=dict(globals(), **locals()))

		self.exchange_recipe.perform_exchange(self)

		# self._PEs[20]._Q
		# self._PEs[20]._nu
		#
		# import code
		# code.interact(local=dict(globals(), **locals()))
		# np.array([PE.samples.mean(axis=1) for PE in self._PEs])

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# for PE in self._PEs:
		#
		# 	covariance = np.linalg.inv(Q)
		#
		# 	mean = np.dot(covariance, nu)
		#
		# 	PE.samples = self.exchange_recipe.truncate_samples(np.random.multivariate_normal(mean, covariance, size=self._n_particles_per_PE).T)

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		return np.NaN

# =========================================================================================================


class TargetTrackingGaussianMixtureParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			sensors, each_PE_required_sensors, exchange_recipe, ad_hoc_parameters, PRNG):

		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors)

		self.exchange_recipe = exchange_recipe
		self._PRNG = PRNG

		self._C = ad_hoc_parameters["number_of_components"]

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.TargetTrackingGaussianMixtureParticleFilter(
			n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			[sensors[iSensor] for iSensor in connections]
		) for connections in each_PE_required_sensors]

	def step(self, observations):

		super().step(observations)

		self.exchange_recipe.perform_exchange(self)

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		return np.NaN

# =========================================================================================================


class TargetTrackingSetMembershipConstrainedParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			sensors, each_PE_required_sensors, exchange_recipe, ad_hoc_parameters, RS_PRNG):

		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors)

		self.exchange_recipe = exchange_recipe

		L = ad_hoc_parameters['over-sampling factor']
		alpha = ad_hoc_parameters['alpha_k']
		beta = ad_hoc_parameters['beta_k']
		# self._n_iterations_global_set = ad_hoc_parameters["iterations for global set determination"]

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.TargetTrackingSetMembershipConstrainedParticleFilter(
			n_particles_per_PE, copy.deepcopy(resampling_algorithm), resampling_criterion, copy.deepcopy(prior),
			copy.deepcopy(state_transition_kernel), [sensors[iSensor] for iSensor in connections], L, alpha, beta,
			copy.deepcopy(RS_PRNG)
		) for connections in each_PE_required_sensors]

	def step(self, observations):

		# neither particles nor weights are modified yet
		super().step(observations)

		# min = np.min([PE.bounding_box_min for PE in self._PEs], axis=0)
		# max = np.max([PE.bounding_box_max for PE in self._PEs], axis=0)

		# import code
		# code.interact(local=dict(globals(), **locals()))

		self.exchange_recipe.global_set_determination(self)

		for PE in self._PEs:

			PE.post_bounding_box_step()

		# import code
		# code.interact(local=dict(globals(), **locals()))

# =========================================================================================================


class TargetTrackingSelectiveGossipParticleFilter(TargetTrackingParticleFilter):

	def __init__(
			self, n_particles_per_PE, resampling_algorithm, resampling_criterion, prior, state_transition_kernel,
			sensors, each_PE_required_sensors, exchange_recipe, ad_hoc_parameters):

		super().__init__(
			exchange_recipe.n_processing_elements, n_particles_per_PE, resampling_algorithm,
			resampling_criterion, prior, state_transition_kernel, sensors, each_PE_required_sensors)

		self.exchange_recipe = exchange_recipe

		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [centralized.TargetTrackingSelectiveGossipParticleFilter(
			n_particles_per_PE, copy.deepcopy(resampling_algorithm), resampling_criterion, copy.deepcopy(prior),
			copy.deepcopy(state_transition_kernel), [sensors[iSensor] for iSensor in connections], self._n_PEs
		) for connections in each_PE_required_sensors]

	def step(self, observations):

		# this simply calls the step method within every PE (neither particles nor weights are modified yet)
		super().step(observations)

		# selective gossip followed by max gossip *for the first-stage weights*
		self.exchange_recipe.selective_and_max_gossip(self)

		for PE, sensors_connections in zip(self._PEs, self._each_PE_required_sensors):

			PE.actual_sampling_step(observations[sensors_connections])

		# np.array([PE.samples for PE in self._PEs])

		# import code
		# code.interact(local=dict(globals(), **locals()))
		#
		# print(np.all(self._PEs[0].samples == self._PEs[1].samples))
		# print(np.all(self._PEs[10].samples == self._PEs[21].samples))
		# print(np.all(self._PEs[5].samples == self._PEs[9].samples))

		# selective gossip followed by max gossip *for the local likelihoods*
		self.exchange_recipe.selective_and_max_gossip(self)

		for PE in self._PEs:

			PE.weights_update_step()

		# print(np.all(self._PEs[0].samples == self._PEs[1].samples))
		# print(np.all(self._PEs[10].samples == self._PEs[21].samples))
		# print(np.all(self._PEs[5].samples == self._PEs[9].samples))
		#
		# import code
		# code.interact(local=dict(globals(), **locals()))
