import collections
import abc
import colorama
import numpy as np
import scipy
import sklearn.mixture
import scipy.stats

import state
import mposterior

# a named tuple for a more intuitive access to a "exchange tuple"
ExchangeTuple = collections.namedtuple(
	'ExchangeTuple', ['i_PE', 'i_particle_within_PE', 'i_neighbour', 'i_particle_within_neighbour'])

# a named tuple for a more intuitive access to a "exchange tuple"
NeighbourParticlesTuple = collections.namedtuple(
	'NeighbourParticlesTuple', ['i_neighbour', 'i_particles'])


class ExchangeRecipe(metaclass=abc.ABCMeta):

	def __init__(self, processing_elements_topology):

		self._PEs_topology = processing_elements_topology

		# for the sake of convenience, we keep the number of PEs...
		self._n_PEs = processing_elements_topology.n_processing_elements

		# the "radius" of the network is half the maximum number of hops between any two PEs
		self._network_diameter = processing_elements_topology.distances_between_processing_elements.max()

	def randomized_wakeup(self, n, PRNG):

		# time elapsed between ticks of the PEs' clocks (each row is the tick of the corresponding PE)
		time_elapsed_between_ticks = PRNG.exponential(size=(self._n_PEs, n))

		# waking times for every PE (as many as the number of iterations so that, in principle, any PE can *always* be
		# the chosen one)
		ticks_absolute_time = time_elapsed_between_ticks.cumsum(axis=1)

		# these are the indexes of the PEs that will wake up to exchange statistics with a neighbour (notice that a
		# certain PE can show up several times)
		# REMARK I: [0] is because we only care about the index of the waking PE (and not the instant)
		# REMARK II: [:self.n_iterations] is because we only consider the "self.n_iterations" earliest wakings
		i_waking_PEs = np.unravel_index(np.argsort(ticks_absolute_time, axis=None), (self._n_PEs, n))[0][:n]

		return i_waking_PEs

	def perform_exchange(self, DPF):

		pass

	def messages(self):

		return np.NaN

	@property
	def n_processing_elements(self):

		return self._n_PEs

	@property
	def PEs_topology(self):

		return self._PEs_topology


# a decorator
class IteratedExchangeRecipe(ExchangeRecipe):

	def __init__(self, exchange_recipe, n_iterations):

		self._exchange_recipe = exchange_recipe
		self._n_iterations = n_iterations

	def messages(self):

		return self._exchange_recipe.messages()*self._n_iterations

	def perform_exchange(self, DPF):

		for _ in range(self._n_iterations):

			self._exchange_recipe.perform_exchange(DPF)

	@property
	def n_processing_elements(self):

		return self._exchange_recipe.n_processing_elements


class ParticlesBasedExchangeRecipe(ExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles):

			super().__init__(processing_elements_topology)

			# the "contacts" of each PE are the PEs it is going to exchange/share particles with
			self.processing_elements_contacts = self.get_PEs_contacts()

			# the number of particles that are to be exchanged between a couple of neighbours is computed (or set)
			if type(exchanged_particles) is int:

				self.n_particles_exchanged_between_neighbours = exchanged_particles

			elif type(exchanged_particles) is float:

				# it is computed accounting for the maximum number of neighbours a given PE can have
				self.n_particles_exchanged_between_neighbours = int(
					(n_particles_per_processing_element * exchanged_particles) // max(
						[len(neighbourhood) for neighbourhood in self.processing_elements_contacts]))

			else:

				raise Exception('type of "exchanged_particles" is not valid')

			if self.n_particles_exchanged_between_neighbours is 0:

				raise Exception('no particles are to be shared by a PE with its processing_elements_contacts')

	def perform_exchange(self, DPF):

		# first, we gather all the particles that are going to be exchanged in an auxiliar variable
		aux = []
		for exchangeTuple in self.exchange_tuples:
			aux.append(
				(DPF.PEs[exchangeTuple.i_PE].get_particle(exchangeTuple.i_particle_within_PE),
				DPF.PEs[exchangeTuple.i_neighbour].get_particle(exchangeTuple.i_particle_within_neighbour)))

		# afterwards, we loop through all the exchange tuples performing the real exchange
		for (exchangeTuple, particles) in zip(self.exchange_tuples, aux):
			DPF.PEs[exchangeTuple.i_PE].set_particle(exchangeTuple.i_particle_within_PE, particles[1])
			DPF.PEs[exchangeTuple.i_neighbour].set_particle(exchangeTuple.i_particle_within_neighbour, particles[0])

	def messages(self):

		# the number of hops between each pair of PEs
		distances = self._PEs_topology.distances_between_processing_elements

		# overall number of messages sent/received in an exchange step
		n_messages = 0

		# for every PE (index) along with its list of neighbours
		for i_processing_element, neighbours_list in enumerate(self.neighbours_particles):

			# each element of the list is a tuple (<index neighbour>,<indexes of the particles exchanged with that neighbour>)
			for i_neighbour, i_particles in neighbours_list:

				# the number of messages required to send the samples
				n_messages += distances[i_processing_element, i_neighbour]*len(i_particles)*state.n_elements

			# we also need to send the aggregated weight to each neighbour
			n_messages += len(neighbours_list)

		return n_messages

	@abc.abstractmethod
	def get_PEs_contacts(self):

		pass

	@abc.abstractproperty
	def exchange_tuples(self):

		pass

	@abc.abstractproperty
	def neighbours_particles(self):

		pass


class DRNAExchangeRecipe(ParticlesBasedExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		super().__init__(processing_elements_topology, n_particles_per_processing_element, exchanged_particles)

		# indexes of the particles...just for the sake of efficiency (this array will be used many times)
		i_particles = np.arange(n_particles_per_processing_element)

		# an array to keep tabs on pairs of PEs already processed
		already_processed_PEs = np.zeros((self._n_PEs, self._n_PEs), dtype=bool)

		# in order to keep tabs on which particles a given PE has already "promised" to exchange
		particles_not_swapped_yet = np.ones((self._n_PEs, n_particles_per_processing_element), dtype=bool)

		# all the elements in this array will be "true" across all the iterations of the loop below
		candidate_particles_all_true = particles_not_swapped_yet.copy()

		if allow_exchange_one_particle_more_than_once:

			# the "reference" below is set to a fixed all-true array
			candidate_particles = candidate_particles_all_true

		else:

			# the "reference" is set to the (varying) "particles_not_swapped_yet"
			candidate_particles = particles_not_swapped_yet

		# named tuples as defined above, each representing an exchange
		self._exchangeTuples = []

		# a list in which the i-th element is also a list containing tuples of the form (<neighbour index>,<numpy array>
		#  with the indices of particles to be exchanged with that neighbour>)
		self._neighbours_particles = [[] for _ in range(self._n_PEs)]

		for iPE, i_this_PE_neighbours in enumerate(self.processing_elements_contacts):

			for iNeighbour in i_this_PE_neighbours:

				if not already_processed_PEs[iPE, iNeighbour]:

					# the particles to be exchanged are chosen randomly (with no replacement) for both, this PE...
					i_exchanged_particles_within_PE = PRNG.choice(
						i_particles[candidate_particles[iPE, :]],
						size=self.n_particles_exchanged_between_neighbours, replace=False)

					# ...and the corresponding neighbour
					i_exchanged_particles_within_neighbour = PRNG.choice(
						i_particles[candidate_particles[iNeighbour, :]],
						size=self.n_particles_exchanged_between_neighbours, replace=False)

					# new "exchange tuple"s are generated
					self._exchangeTuples.extend([ExchangeTuple(
						i_PE=iPE, i_particle_within_PE=iParticleWithinPE, i_neighbour=iNeighbour,
						i_particle_within_neighbour=iParticleWithinNeighbour
					) for iParticleWithinPE, iParticleWithinNeighbour in zip(
						i_exchanged_particles_within_PE, i_exchanged_particles_within_neighbour)])

					# these PEs (the one considered in the main loop and the neighbour being processed) should not
					# exchange the selected particles (different in each case) with other PEs
					particles_not_swapped_yet[iPE, i_exchanged_particles_within_PE] = False
					particles_not_swapped_yet[iNeighbour, i_exchanged_particles_within_neighbour] = False

					# we "mark" this pair of PEs as already processed; despite the symmetry,
					# only "already_processed_PEs[iNeighbour, iPe]" should be accessed later on
					already_processed_PEs[iNeighbour, iPE] = already_processed_PEs[iPE, iNeighbour] = True

					# each tuple specifies a neighbor, and the particles THE LATTER exchanges with it (rather than
					# the other way around)
					self._neighbours_particles[iPE].append(
						NeighbourParticlesTuple(iNeighbour, i_exchanged_particles_within_neighbour))

					self._neighbours_particles[iNeighbour].append(
						NeighbourParticlesTuple(iPE, i_exchanged_particles_within_PE))

	@property
	def exchange_tuples(self):

		return self._exchangeTuples

	# this is only meant to be used by subclasses (specifically, Mposterior-related ones)
	@property
	def neighbours_particles(self):

		"""Particles received from each neighbour.

		Returns
		-------
		neighbours, particles : list of lists
			Every individual list contains tuples of the form (<index neighbour>, <indexes particles within
			that neighbour>) for the corresponding PE
		"""

		return self._neighbours_particles

	def get_PEs_contacts(self):

		return self._PEs_topology.get_neighbours()


class MposteriorExchangeRecipe(DRNAExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			weiszfeld_parameters, PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		super().__init__(processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			PRNG, allow_exchange_one_particle_more_than_once)

		self.weiszfeld_parameters = weiszfeld_parameters

		self.i_own_particles_within_PEs = [PRNG.randint(
			n_particles_per_processing_element, size=self.n_particles_exchanged_between_neighbours
		) for _ in range(self._n_PEs)]

	def perform_exchange(self, DPF):

		for PE, this_PE_neighbours_particles, i_this_PE_particles in zip(
				DPF.PEs, self.neighbours_particles, self.i_own_particles_within_PEs):

			# a list with the subset posterior of each neighbour
			subset_posterior_distributions = [
				DPF.PEs[neighbour_particles.i_neighbour].get_samples_at(neighbour_particles.i_particles).T
				for neighbour_particles in this_PE_neighbours_particles]

			# a subset posterior obtained from this PE is also added: it encompasses
			# the particles whose indexes are given in "i_this_PE_particles"
			subset_posterior_distributions.append(PE.get_samples_at(i_this_PE_particles).T)

			joint_particles, joint_weights = mposterior.find_weiszfeld_median(
					subset_posterior_distributions, **self.weiszfeld_parameters)

			# the indexes of the particles to be kept
			i_new_particles = DPF._resampling_algorithm.get_indexes(joint_weights, PE.n_particles)

			PE.samples = joint_particles[:, i_new_particles]
			PE.log_weights = np.full(PE.n_particles, -np.log(PE.n_particles))
			PE.update_aggregated_weight()

	def messages(self):

		# same as for DRNA...
		n_messages = super().messages()

		# ...but there is no need for a PE to send its aggregated weight to each one of its neighbours
		for neighbours_list in self.neighbours_particles:

			n_messages -= len(neighbours_list)

		return n_messages


class MposteriorWithinRadiusExchangeRecipe(MposteriorExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			weiszfeld_parameters, radius, PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		# this needs to be set before super() because the ancestor class "__init__" depends on "get_PEs_contacts" which,
		#  in turn, depends on radius
		self.radius = radius

		super().__init__(
				processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
				weiszfeld_parameters, PRNG, allow_exchange_one_particle_more_than_once)

	def get_PEs_contacts(self):

		return self._PEs_topology.i_neighbours_within_hops(self.radius)


class SameParticlesMposteriorWithinRadiusExchangeRecipe(MposteriorWithinRadiusExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			weiszfeld_parameters, radius, PRNG=np.random.RandomState()):

		# "allow_exchange_one_particle_more_than_once" is set to true since it is irrelevant here, but could cause the
		# parent class to throw an exception if set to False
		super().__init__(
				processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
				weiszfeld_parameters, radius, PRNG, allow_exchange_one_particle_more_than_once=True)

		i_particles_shared_by_each_PE = [
			PRNG.choice(n_particles_per_processing_element, size=self.n_particles_exchanged_between_neighbours, replace=False)
			for _ in range(self._n_PEs)]

		self._neighbours_particles = []

		for neighbours, i_shared_particles in zip(
				self.processing_elements_contacts, i_particles_shared_by_each_PE):

			self._neighbours_particles.append([NeighbourParticlesTuple(n, i_shared_particles) for n in neighbours])

	@property
	def exchange_tuples(self):

		raise Exception('not implemented!!')

	@property
	def neighbours_particles(self):

		return self._neighbours_particles


class LikelihoodConsensusExchangeRecipe(ExchangeRecipe):

	def __init__(self, processing_elements_topology, max_iterations, polynomial_degree):

		super().__init__(processing_elements_topology)

		self._max_iterations = max_iterations
		self.polynomial_degree = polynomial_degree

		# a list of lists in which each element yields the neighbors of a PE
		self._neighborhoods = processing_elements_topology.get_neighbours()

		# Metropolis weights
		# ==========================

		# this will store tuples (<own weight>,<numpy array with weights for each neighbor>)
		self._metropolis_weights = []

		# for the neighbours of every PE
		for neighbors in self._neighborhoods:

			# the number of neighbors of the PE
			n_neighbors = len(neighbors)

			# the weight assigned to each one of its neighbors
			neighbors_weights = np.array(
				[1/(1+max(n_neighbors, len(self._neighborhoods[i_neighbour]))) for i_neighbour in neighbors])

			# the weight assigned to itself is the first element in the tuple
			self._metropolis_weights.append((1-neighbors_weights.sum(), neighbors_weights))

	def perform_exchange(self, DPF):

		# the first iteration of the consensus algorithm
		# ==========================

		# for every PE, along with its neighbours
		for PE, neighbours, weights in zip(DPF.PEs, self._neighborhoods, self._metropolis_weights):

			# a dictionary for storing the "consensed" beta's
			PE.betaConsensus = {}

			# for every combination of exponents, r
			for r in DPF._r_d_tuples:

				PE.betaConsensus[r] = PE.beta[r]*weights[0] + np.array(
					[DPF.PEs[i_neighbour].beta[r] for i_neighbour in neighbours]).dot(weights[1])

		# the remaining iterations of the consensus algorithm
		# ==========================

		# the same operations as above using "betaConsensus" rather than beta
		for _ in range(self._max_iterations-1):

			# for every PE, along with its neighbours
			for PE, neighbours, weights in zip(DPF.PEs, self._neighborhoods, self._metropolis_weights):

				# for every combination of exponents, r
				for r in DPF._r_d_tuples:

					PE.betaConsensus[r] = PE.betaConsensus[r]*weights[0] + np.array(
						[DPF.PEs[i_neighbour].betaConsensus[r] for i_neighbour in neighbours]).dot(weights[1])

		# every average is turned into a sum
		# ==========================

		# for every PE...
		for PE in DPF.PEs:

			# ...and every coefficient computed
			for r in DPF._r_d_tuples:

				PE.betaConsensus[r] *= self._n_PEs

	def messages(self):

		# the length of subset of the state on which the likelihood depends
		M = 2

		# theoretically, this is the number of beta components that should result
		n_consensus_algorithms = scipy.misc.comb(2*self.polynomial_degree + M, 2*self.polynomial_degree, exact=True) - 1

		# overall number of neighbours: #neighbours of the 1st PE + #neighbours of the 2nd PE +...
		n_neighbours = sum([len(neighbours) for neighbours in self._neighborhoods])

		# each PE sends "n_consensus_algorithms" values to each one of its neighbours, once per iteration...
		n_messages = n_neighbours*n_consensus_algorithms*self._max_iterations

		# ...additionally it needs to send each neighbour the number of neighbours it has itself (Metropolis weights)
		n_messages += n_neighbours

		return n_messages


class GaussianExchangeRecipe(ExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_PE, ad_hoc_parameters, room, PRNG):

		super().__init__(processing_elements_topology)

		self._n_particles_per_PE = n_particles_per_PE
		self._ad_hoc_parameters = ad_hoc_parameters

		self._room = room
		self._PRNG = PRNG

		# if no number of iterations is provided through the parameters file...
		if "n_iterations" not in ad_hoc_parameters:

			# ...then it is determined by the (maybe estimated) number of PEs
			# TODO: this assumes the actual number of PEs is known to *every* PE
			self.n_iterations = 5*self._n_PEs**2

		else:

			self.n_iterations = ad_hoc_parameters["n_iterations"]

		self._edge_epsilon = 0.001

		# so that the corresponding method is only called once
		self._PEs_neighbors = processing_elements_topology.get_neighbours()

	def perform_exchange(self, DPF):

		# indexes of the PEs that will wake up during this exchange
		i_waking_PEs = self.randomized_wakeup(self.n_iterations, self._PRNG)

		for i_PE in i_waking_PEs:

			i_selected_neighbor = self._PRNG.choice(self._PEs_neighbors[i_PE])

			# average of the Qs
			DPF.PEs[i_PE]._Q = DPF.PEs[i_selected_neighbor]._Q\
				= (DPF.PEs[i_PE]._Q + DPF.PEs[i_selected_neighbor]._Q)/2

			# average of the nus
			DPF.PEs[i_PE]._nu = DPF.PEs[i_selected_neighbor]._nu\
				= (DPF.PEs[i_PE]._nu + DPF.PEs[i_selected_neighbor]._nu) / 2

	def messages(self):

		# accumulator
		res = 0

		# "nu" is state.n_elements x 1, "Q" is state.n_elements x state.n_elements
		# multiplying by 2 because PE #1 sends data to PE #2 and the other way around
		res += (2 * (state.n_elements + (state.n_elements**2 + state.n_elements)/2)) * self.n_iterations

		return res

	def size_estimation(self, DPF):

		# for the sake of a shorter code...
		alpha = self._ad_hoc_parameters["size_estimate_alpha"]

		# different random walks have different associated data
		PEs_data = [{} for _ in range(self._n_PEs)]

		# id is the number of a PE, which is here used as "token" (identifier); it is a keyword inside the dictionary
		for id, PE in enumerate(DPF.PEs):

			# the probability of a starting value equal to 1
			ro = 1 / PE.estimated_n_PEs

			# draw a sample 0 or 1 according to the previous probability
			starting_value = self._PRNG.choice(2, p=[1 - ro, ro])

			# this is a metric used in the stopping rule
			exponential_moving_average = 1

			# if the starting value is 1
			if starting_value:

				print('PE #{} starting a random walk'.format(id))

				# the current PE is the one starting the random walk
				i_current_PE = id

				# the value associated with the current PE AND this particular random walk identified by "id"
				PEs_data[i_current_PE][id] = 1

				# the stopping rule
				while exponential_moving_average > self._ad_hoc_parameters["size_estimate_threshold"]:

					# the next PE is selected randomly among the neighbors of the current one
					i_next = self._PRNG.choice(self._PEs_neighbors[i_current_PE])

					# this is needed to update exponential_moving_average
					difference = PEs_data[i_current_PE].setdefault(id, 0) - PEs_data[i_next].setdefault(id, 0)

					# the value for the current PE *and* the selected neighbor is updated
					PEs_data[i_current_PE][id] = PEs_data[i_next][id] = (PEs_data[i_current_PE].setdefault(id, 0) + PEs_data[i_next].setdefault(id, 0))/2

					# the metric for the stopping rule is updated
					exponential_moving_average = (1-alpha)*exponential_moving_average + alpha*difference**2

					# the next PE becomes the current
					i_current_PE = i_next

		# for every PE
		for i_PE, PE in enumerate(DPF.PEs):

			# if the corresponding dictionary is not empty...
			if PEs_data[i_PE]:

				# its estimate of the number of PEs is the inverse of the average of all the values (for different
				# "tokens") that it has
				PE.estimated_n_PEs = 1//(sum([x for x in PEs_data[i_PE].values()])/len(PEs_data[i_PE].values()))

				# # TODO: cheating!!
				# PE.estimated_n_PEs = self.n_processing_elements

			# print(PE.estimated_n_PEs)

		# print([PE.estimated_n_PEs for PE in DPF.PEs])
		print('size_estimation: mean is {}'.format(np.array([PE.estimated_n_PEs for PE in DPF.PEs]).mean()))


class SetMembershipConstrainedExchangeRecipe(ExchangeRecipe):

	def __init__(self, processing_elements_topology, ad_hoc_parameters, n_particles_per_PE, PRNG):

		super().__init__(processing_elements_topology)

		self._n_particles_per_PE = n_particles_per_PE
		self._PRNG = PRNG

		self._n_iterations_likelihood_consensus = ad_hoc_parameters["iterations for likelihood consensus"]

		if "iterations for global set determination" in ad_hoc_parameters:

			self._n_iterations_global_set_determination = ad_hoc_parameters["iterations for global set determination"]

		else:

			self._n_iterations_global_set_determination = self._network_diameter

		if "iterations for likelihood max/min consensus" in ad_hoc_parameters:

			self._n_iterations_max_min_likelihood_consensus = ad_hoc_parameters[
				"iterations for likelihood max/min consensus"]
		else:

			self._n_iterations_max_min_likelihood_consensus = self._network_diameter

		self._mu = ad_hoc_parameters["mu for likelihood consensus"]

		# a list of lists in which each element yields the neighbors of a PE
		self._neighborhoods = processing_elements_topology.get_neighbours()

	def messages(self):

		# the number of neighbors each PE has
		n_neighbors = np.array([len(neigh) for neigh in self._neighborhoods])

		# accumulator
		res = 0

		# global set determination
		# REMARK: *2 is due to min/max gossip
		res += (state.n_elements*2*n_neighbors).sum()*self._n_iterations_global_set_determination

		# consensus on likelihood
		res += (self._n_particles_per_PE * n_neighbors).sum() * self._n_iterations_likelihood_consensus

		# max/min consensus
		res += (self._n_particles_per_PE * n_neighbors * 2).sum() * self._n_iterations_max_min_likelihood_consensus

		return res

	def global_set_determination(self, DPF):

		# Max gossip
		for _ in range(self._n_iterations_global_set_determination):

			for PE, neighbours in zip(DPF.PEs, self._neighborhoods):

				PE.bounding_box_min = np.max(
					[PE.bounding_box_min] + [DPF.PEs[i_neighbor].bounding_box_min for i_neighbor in neighbours], axis=0
				)

				PE.bounding_box_max = np.min(
					[PE.bounding_box_max] + [DPF.PEs[i_neighbor].bounding_box_max for i_neighbor in neighbours], axis=0
				)

	def consensus_on_likelihood(self, DPF):

		# every row a PE, every column a particle
		loglikelihoods = np.array([PE.loglikelihoods for PE in DPF.PEs])

		for _ in range(self._n_iterations_likelihood_consensus):

			loglikelihoods_copy = loglikelihoods.copy()

			for i_PE, (PE, neighbours) in enumerate(zip(DPF.PEs, self._neighborhoods)):

				loglikelihoods[i_PE, :] += self._mu*(
					loglikelihoods_copy[neighbours, :] - loglikelihoods[i_PE, :]).sum(axis=0)

		# initialization (every row a PE, every column a particle)
		max_loglikelihoods = loglikelihoods.copy()
		min_loglikelihoods = loglikelihoods.copy()

		for _ in range(self._n_iterations_max_min_likelihood_consensus):

			max_loglikelihoods_copy = max_loglikelihoods.copy()
			min_loglikelihoods_copy = min_loglikelihoods.copy()

			for i_PE, (PE, neighbours) in enumerate(zip(DPF.PEs, self._neighborhoods)):

				max_loglikelihoods[i_PE, :] = np.vstack(
					(max_loglikelihoods_copy[neighbours, :], max_loglikelihoods[i_PE, :])
				).max(axis=0)

				min_loglikelihoods[i_PE, :] = np.vstack(
					(min_loglikelihoods_copy[neighbours, :], min_loglikelihoods[i_PE, :])
				).min(axis=0)

		sum_loglikelihoods_estimate = (max_loglikelihoods +  min_loglikelihoods)/2 * self._n_PEs

		for PE, estimate in zip(DPF.PEs, sum_loglikelihoods_estimate):

			PE.loglikelihoods = estimate


class SelectiveGossipExchangeRecipe(ExchangeRecipe):

	def __init__(self, processing_elements_topology, ad_hoc_parameters, PRNG):

		super().__init__(processing_elements_topology)

		self._PRNG = PRNG

		self._n_iterations_selective_gossip = ad_hoc_parameters["iterations for selective gossip"]
		self._n_components_selective_gossip = ad_hoc_parameters["number of significant components for selective gossip"]

		if "iterations for max gossip" in ad_hoc_parameters:

			self._n_iterations_max_gossip = ad_hoc_parameters["iterations for max gossip"]

		else:

			self._n_iterations_max_gossip = self._network_diameter

		# a list of lists in which each element yields the neighbors of a PE
		self._neighborhoods = processing_elements_topology.get_neighbours()

	def messages(self):

		# accumulator
		res = 0

		# in order to consent on the first-stage weight
		# =================================

		# 2 (indexes and values) for PE #1 -> PE #2, 2 for PE #2 -> PE #1
		# REMARK: this assumes the list os significant matches, otherwise a third transmission from #1 to #2 with the
		# values that are significant for #2 but not for #1
		res += self._n_components_selective_gossip * 4 * self._n_iterations_selective_gossip

		# max-gossip
		# REMARK: this assumes they all reached consensus on the correct indexes (no need to transmit them)
		res += self._n_components_selective_gossip * 2 * self._n_iterations_max_gossip

		# in order to consent on the actual weights
		# =================================

		# same operations
		res *= 2

		return res

	def selective_and_max_gossip(self, DPF):

		# ------------------- selective Gossip ---------------------

		# the gamma's obtained by every PE are collected in an array
		# REMARK: every row is a different PE, every column a value
		gammas = np.array([PE.gamma for PE in DPF.PEs])

		# for every PE, a list with the indexes of the significant values *according to that PE*
		PEs_i_significant = [[]] * self._n_PEs

		# indexes of the nodes to be wakened during this gossip operation
		i_nodes_to_be_wakened = self.randomized_wakeup(self._n_iterations_selective_gossip, self._PRNG)

		for i in i_nodes_to_be_wakened:

			# index of the selected neighbor
			i_neigh = self._PRNG.choice(self._neighborhoods[i])

			# for the sake of convenience
			i_involved_PEs = [i, i_neigh]

			# for the PE and its neighbor, the indexes of the largest "self._n_components_selective_gossip" gamma's
			i_largest = np.argsort(gammas[i_involved_PEs, :], axis=1)[:, -self._n_components_selective_gossip:]

			# the union of the indexes for the significant components of the PE and its neighbor
			i_significant = list(set(i_largest[0, :]) | set(i_largest[1, :]))

			# the list with the significant values for every node is updated
			PEs_i_significant[i] = PEs_i_significant[i_neigh] = i_significant

			# in order to select the components of "gammas" that must be updated
			array_range = np.ix_(i_involved_PEs, i_significant)

			# the significant components are updated in *both* PEs to the mean
			gammas[array_range] = gammas[array_range].mean(axis=0)

		# ---------------------- MAX Gossip ------------------------

		# indexes of the nodes to be wakened during this gossip operation
		i_nodes_to_be_wakened = self.randomized_wakeup(self._n_iterations_max_gossip, self._PRNG)

		for i in i_nodes_to_be_wakened:

			# index of the selected neighbor
			i_neigh = self._PRNG.choice(self._neighborhoods[i])

			# for the sake of convenience
			i_involved_PEs = [i, i_neigh]

			# every PE obtains the maximum for the values *it* considers significant
			# REMARK: this assumes the neighbor sends the information about those values *even if they are not its
			# significant values*
			gammas[i, PEs_i_significant[i]] = gammas[np.ix_(i_involved_PEs, PEs_i_significant[i])].max(axis=0)
			gammas[i_neigh, PEs_i_significant[i_neigh]] = gammas[np.ix_(i_involved_PEs, PEs_i_significant[i_neigh])].max(axis=0)

		# the results of the consensus for every PE are stored within it for later access
		for PE, gamma, indexes in zip(DPF.PEs, gammas, PEs_i_significant):

			PE.gamma_postgossip = gamma[indexes]
			PE.i_significant = indexes


class PerfectConsensusGaussianExchangeRecipe(GaussianExchangeRecipe):

	def messages(self):

		return np.NaN

	def perform_exchange(self, DPF):

		# lists with all the Q's and nu's
		Qs = [PE._Q for PE in DPF.PEs]
		nus = [PE._nu for PE in DPF.PEs]

		# the mean of each one is computed along the appropriate axis
		mean_Q = np.stack(Qs, axis=2).mean(axis=2)
		mean_nu = np.stack(nus, axis=1).mean(axis=1)

		# the computed means are set within every PE
		for PE in DPF.PEs:

			PE._Q = mean_Q
			PE._nu = mean_nu


class PerfectSelectiveGossipExchangeRecipe(SelectiveGossipExchangeRecipe):

	def messages(self):

		return np.NaN

	def selective_and_max_gossip(self, DPF):

		# ------------------- selective Gossip ---------------------

		# the gamma's obtained by every PE are collected in an array
		# REMARK: every row is a different PE, every column a value
		gammas = np.array([PE.gamma for PE in DPF.PEs])

		# the indexes of the "self._n_components_selective_gossip" largest gammas in every PE
		PEs_i_significant = np.argsort(gammas, axis=1)[:, -self._n_components_selective_gossip:]

		# accumulator for the union
		union = set()

		for pe in PEs_i_significant:

			union |= set(pe)

		# the union of all the signficant indexes
		i_significant = list(union)

		# the mean of the corresponding gammas in every PE
		gammas_means = gammas[:, i_significant].mean(axis=0)

		# the result is stored within every PE
		for PE in DPF.PEs:

			PE.gamma_postgossip = gammas_means
			PE.i_significant = i_significant
