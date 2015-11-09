import collections
import abc
import numpy as np
import scipy

import state


class ExchangeRecipe(metaclass=abc.ABCMeta):

	def __init__(self, processing_elements_topology):

		self._PEs_topology = processing_elements_topology

		# for the sake of convenience, we keep the number of PEs...
		self._n_PEs = processing_elements_topology.n_processing_elements

	@abc.abstractmethod
	def perform_exchange(self, DPF):

		pass

	@abc.abstractmethod
	def messages(self):

		return


class DRNAExchangeRecipe(ExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		"""Computes which particles from which PEs are exchanged.
		
		"""

		super().__init__(processing_elements_topology)

		# the "contacts" of each PE are the PEs it is going to exchange/share particles with
		processing_elements_contacts = self.get_PEs_contacts()

		# a named tuple for a more intuitive access to a "exchange tuple"
		ExchangeTuple = collections.namedtuple(
			'ExchangeTuple', ['i_PE', 'i_particle_within_PE', 'i_neighbour', 'i_particle_within_neighbour'])

		# a named tuple for a more intuitive access to a "exchange tuple"
		NeighbourParticlesTuple = collections.namedtuple(
			'NeighbourParticlesTuple', ['i_neighbour', 'i_particles'])

		# indexes of the particles...just for the sake of efficiency (this array will be used many times)
		i_particles = np.arange(n_particles_per_processing_element)

		# the number of particles that are to be exchanged between a couple of neighbours is computed (or set)
		if type(exchanged_particles) is int:

			self.n_particles_exchanged_between_neighbours = exchanged_particles

		elif type(exchanged_particles) is float:

			# it is computed accounting for the maximum number of processing_elements_contacts a given PE can have
			self.n_particles_exchanged_between_neighbours = int(
				(n_particles_per_processing_element * exchanged_particles) // max(
					[len(neighbourhood) for neighbourhood in processing_elements_contacts]))

		else:

			raise Exception('type of exchanged_particles is not valid')

		if self.n_particles_exchanged_between_neighbours is 0:

			raise Exception('no particles are to be shared by a PE with its processing_elements_contacts')

		# an array to keep tabs on pairs of PEs already processed
		already_processed_PEs = np.zeros((self._n_PEs, self._n_PEs), dtype=bool)

		# in order to keep tabs on which particles a given PE has already "promised" to exchange
		particles_not_swapped_yet = np.ones((self._n_PEs, n_particles_per_processing_element), dtype=bool)

		# this will be always true across all the iterations of the loop below
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

		for iPE, i_this_PE_neighbours in enumerate(processing_elements_contacts):

			for iNeighbour in i_this_PE_neighbours:

				if not already_processed_PEs[iPE, iNeighbour]:

					# the particles to be exchanged are chosen randomly (with no replacement) for both, the this PE...
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

					self._neighbours_particles[iPE].append(
						NeighbourParticlesTuple(iNeighbour, i_exchanged_particles_within_PE))

					self._neighbours_particles[iNeighbour].append(
						NeighbourParticlesTuple(iPE, i_exchanged_particles_within_neighbour))

	def get_PEs_contacts(self):

		return self._PEs_topology.get_neighbours()

	@property
	def n_processing_elements(self):

		return self._n_PEs

	def perform_exchange(self, DPF):

		# first, we compile all the particles that are going to be exchanged in an auxiliar variable
		aux = []
		for exchangeTuple in self._exchangeTuples:
			aux.append(
				(DPF._PEs[exchangeTuple.i_PE].get_particle(exchangeTuple.i_particle_within_PE),
				DPF._PEs[exchangeTuple.i_neighbour].get_particle(exchangeTuple.i_particle_within_neighbour)))

		# afterwards, we loop through all the exchange tuples performing the real exchange
		for (exchangeTuple, particles) in zip(self._exchangeTuples, aux):
			DPF._PEs[exchangeTuple.i_PE].set_particle(exchangeTuple.i_particle_within_PE, particles[1])
			DPF._PEs[exchangeTuple.i_neighbour].set_particle(exchangeTuple.i_particle_within_neighbour, particles[0])

	def messages(self):

		# the number of hops between each pair of PEs
		distances = self._PEs_topology.distances_between_processing_elements

		# overall number of messages sent/received in an exchange step
		n_messages = 0

		# for every PE (index) along with its list of neighbours
		for i_processing_element, neighbours_list in enumerate(self._neighbours_particles):

			# each element of the list is a tuple (<index neighbour>,<indexes of the particles exchanged with that neighbour>)
			for i_neighbour, i_particles in neighbours_list:

				# the number of messages required to send the samples
				n_messages += distances[i_processing_element, i_neighbour]*len(i_particles)*state.nElements

			# we also need to send the aggregated weight to each neighbour
			n_messages += len(neighbours_list)

		return n_messages


class MposteriorExchangeRecipe(DRNAExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		super().__init__(processing_elements_topology, n_particles_per_processing_element, exchanged_particles,
			PRNG, allow_exchange_one_particle_more_than_once)

		self.i_particles_within_processing_elements = [np.random.randint(
			n_particles_per_processing_element, size=self.n_particles_exchanged_between_neighbours
		) for _ in range(self._n_PEs)]

	def perform_exchange(self, DPF):

		for PE, this_PE_neighbours_particles, i_this_PE_particles in zip(
				DPF._PEs, self._neighbours_particles, self.i_particles_within_processing_elements):

			# a list with the subset posterior of each neighbour
			subset_posterior_distributions = [
				(DPF._PEs[neighbour_particles.i_neighbour].get_samples_at(neighbour_particles.i_particles).T,
				np.full(self.n_particles_exchanged_between_neighbours, 1.0/self.n_particles_exchanged_between_neighbours))
				for neighbour_particles in this_PE_neighbours_particles]

			# a subset posterior obtained from this PE is also added: it encompasses
			# the particles whose indexes are given in "i_this_PE_particles"
			subset_posterior_distributions.append(
				(PE.get_samples_at(i_this_PE_particles).T,
				 np.full(self.n_particles_exchanged_between_neighbours, 1.0/self.n_particles_exchanged_between_neighbours)))

			# M posterior on the posterior distributions collected above
			joint_particles, joint_weights = DPF.Mposterior(subset_posterior_distributions)

			# the indexes of the particles to be kept
			i_new_particles = DPF._resamplingAlgorithm.getIndexes(joint_weights, PE._nParticles)

			PE.samples = joint_particles[:, i_new_particles]
			PE.log_weights = np.full(PE._nParticles, -np.log(PE._nParticles))
			PE.update_aggregated_weight()

	def messages(self):

		# same as for DRNA...
		n_messages = super().messages()

		# ...but there is no need for a PE to send its aggregated weight to each one of its neighbours
		for neighbours_list in self._neighbours_particles:

			n_messages -= len(neighbours_list)

		return n_messages


class MposteriorWithinRadiusExchangeRecipe(MposteriorExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles, radius,
			PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		# this needs be before super() because the ancestor class is depends "get_PEs_contacts" which,
		#  in turn, depends on radius
		self.radius = radius

		super().__init__(
			processing_elements_topology, n_particles_per_processing_element, exchanged_particles, PRNG,
			allow_exchange_one_particle_more_than_once)

	def get_PEs_contacts(self):

		return self._PEs_topology.i_neighbours_within_hops(self.radius)


class IteratedMposteriorExchangeRecipe(MposteriorExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_processing_element, exchanged_particles, number_iterations,
			PRNG=np.random.RandomState(), allow_exchange_one_particle_more_than_once=False):

		super().__init__(
			processing_elements_topology, n_particles_per_processing_element, exchanged_particles, PRNG,
			allow_exchange_one_particle_more_than_once)

		self._number_iterations = number_iterations

	def perform_exchange(self, DPF):

		for _ in range(self._number_iterations):

			super().perform_exchange(DPF)

	def messages(self):

		return super().messages()*self._number_iterations


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
		for PE, neighbours, weights in zip(DPF._PEs, self._neighborhoods, self._metropolis_weights):

			# a dictionary for storing the "consensed" beta's
			PE.betaConsensus = {}

			# for every combination of exponents, r
			for r in DPF._r_d_tuples:

				PE.betaConsensus[r] = PE.beta[r]*weights[0] + np.array(
					[DPF._PEs[i_neighbour].beta[r] for i_neighbour in neighbours]).dot(weights[1])

		# the remaining iterations of the consensus algorithm
		# ==========================

		# the same operations as above using "betaConsensus" rather than beta
		for _ in range(self._max_iterations-1):

			# for every PE, along with its neighbours
			for PE, neighbours, weights in zip(DPF._PEs, self._neighborhoods, self._metropolis_weights):

				# for every combination of exponents, r
				for r in DPF._r_d_tuples:

					PE.betaConsensus[r] = PE.betaConsensus[r]*weights[0] + np.array(
						[DPF._PEs[i_neighbour].betaConsensus[r] for i_neighbour in neighbours]).dot(weights[1])

		# every average is turned into a sum
		# ==========================

		# for every PE...
		for PE in DPF._PEs:

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
