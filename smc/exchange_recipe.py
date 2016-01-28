import collections
import abc
import numpy as np
import scipy

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

	@abc.abstractmethod
	def perform_exchange(self, DPF):

		pass

	@abc.abstractmethod
	def messages(self):

		return

	@property
	def n_processing_elements(self):

		return self._n_PEs


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

		self.i_own_particles_within_PEs = [np.random.randint(
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


class MeanCovarianceAggregatedWeightExchangeRecipe(ExchangeRecipe):

	def __init__(
			self, processing_elements_topology, resampling_algorithm, n_particles_per_PE, bottom_left_corner,
			top_right_corner, PRNG, every_PE=False):

		super().__init__(processing_elements_topology)

		self.resampling_algorithm = resampling_algorithm
		self._n_particles_per_PE = n_particles_per_PE
		self._bottom_left_corner = bottom_left_corner
		self._top_right_corner = top_right_corner
		self._PRNG = PRNG

		# for the sake of convenience
		n = processing_elements_topology.n_processing_elements

		if every_PE:
			self._PEs_partners = [[j for j in range(n) if j != i] for i in range(n)]
		else:
			self._PEs_partners = self._PEs_topology.get_neighbours()

		# from IPython.core.debugger import Tracer; debug_here = Tracer()
		# debug_here()

	def get_means_covariances_aggregated_weights(self, DPF):

		aggregated_weights = np.empty(self._n_PEs)
		means = np.empty((state.n_elements, self._n_PEs))
		covariances = np.empty((state.n_elements, state.n_elements, self._n_PEs))

		for i_PE, PE in enumerate(DPF.PEs):

			# the mean of the current particles
			means[:, i_PE:i_PE+1] = PE.compute_mean()

			# the covariance thereof
			covariances[:, :, i_PE] = np.cov(PE.samples)

			# the aggregated weight BEFORE normalization
			aggregated_weights[i_PE] = PE.old_aggregated_weight

		return means, covariances, aggregated_weights

	def truncate_samples(self, samples):

		res = samples.copy()

		res[0, res[0, :] < self._bottom_left_corner[0]] = self._bottom_left_corner[0] + 0.1
		res[0, res[0, :] > self._top_right_corner[0]] = self._top_right_corner[0] - 0.1
		res[1, res[1, :] < self._bottom_left_corner[1]] = self._bottom_left_corner[1] + 0.1
		res[1, res[1, :] > self._top_right_corner[1]] = self._top_right_corner[1] - 0.1

		return res

	def perform_exchange(self, DPF):

		# mean, covariance and aggregated weight (before NORMALIZATION) from every PE
		means, covariances, aggregated_weights = self.get_means_covariances_aggregated_weights(DPF)

		for i_PE, (PE, i_neighbours) in enumerate(zip(DPF.PEs, self._PEs_partners)):

			# in addition to its neighbours, the PE itself is also involved in this information exchange
			i_involved_PEs = [i_PE] + i_neighbours

			# the aggregated weights of all the PE participating are normalized...
			normalized_aggregated_weights = aggregated_weights[i_involved_PEs]/aggregated_weights[i_involved_PEs].sum()

			# ...and used to decide how many particles are drawn from each Gaussian
			n_particles_from_each_PE = self._PRNG.multinomial(self._n_particles_per_PE, normalized_aggregated_weights)

			# a list with a numpy array of particles coming from every involved PE
			new_particles = []

			for n, mean, covariance in zip(
					n_particles_from_each_PE, means[:, i_involved_PEs].T, covariances[:, :, i_involved_PEs].T):

				# the number of particles previously assigned to this PE are drawn using the corresponding mean and covariance
				new_particles.append(self._PRNG.multivariate_normal(mean, covariance, n))

			# from IPython.core.debugger import Tracer; debug_here = Tracer()
			# debug_here()

			PE.samples = self.truncate_samples(np.vstack(new_particles).T)
			PE.weights = np.full(self._n_particles_per_PE, 1/self._n_particles_per_PE)
			PE.update_aggregated_weight()

		# from IPython.core.debugger import Tracer; debug_here = Tracer()
		# debug_here()

	def messages(self):

		# the number of hops between each pair of PEs
		distances = self._PEs_topology.distances_between_processing_elements

		n_elements_covariance_matrix = (state.n_elements * (state.n_elements - 1))/2 + state.n_elements

		n_messages = 0

		for i_PE, i_neighbours in enumerate(self._PEs_partners):

			# mean, covariance and aggregated weight must be sent to each neighbour
			n_messages += distances[i_PE, i_neighbours].sum()*(state.n_elements + n_elements_covariance_matrix + 1)

			# from IPython.core.debugger import Tracer; debug_here = Tracer()
			# debug_here()

		return n_messages