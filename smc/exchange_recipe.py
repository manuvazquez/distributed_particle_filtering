import collections
import abc
import copy
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

	@abc.abstractmethod
	def perform_exchange(self, DPF):

		pass

	@abc.abstractmethod
	def messages(self):

		return

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


class GaussianExchangeRecipe(ExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_PE, ad_hoc_parameters, bottom_left_corner,
			top_right_corner, PRNG):

		super().__init__(processing_elements_topology)

		self._n_particles_per_PE = n_particles_per_PE
		self._ad_hoc_parameters = ad_hoc_parameters

		self._bottom_left_corner = bottom_left_corner
		self._top_right_corner = top_right_corner

		self._PRNG = PRNG

		# if no number of iterations is provided through the parameters file...
		if "n_iterations" not in ad_hoc_parameters:

			# ...the initial estimate is assumed to be...
			# TODO: this assumes the actual number of PEs is known to *every* PE
			self.n_iterations = 5*self._n_PEs**2

		else:

			self.n_iterations = ad_hoc_parameters["n_iterations"]

		self._edge_epsilon = 0.001

		# so that the corresponding method is only called once
		self._PEs_neighbors = processing_elements_topology.get_neighbours()

	def truncate_samples(self, samples):

		res = samples.copy()

		res[0, res[0, :] < self._bottom_left_corner[0]] = self._bottom_left_corner[0] + self._edge_epsilon
		res[0, res[0, :] > self._top_right_corner[0]] = self._top_right_corner[0] - self._edge_epsilon
		res[1, res[1, :] < self._bottom_left_corner[1]] = self._bottom_left_corner[1] + self._edge_epsilon
		res[1, res[1, :] > self._top_right_corner[1]] = self._top_right_corner[1] - self._edge_epsilon

		return res

	def perform_exchange(self, DPF):

		# time elapsed between ticks of the PEs' clocks (each row is the tick of the corresponding PE)
		time_elapsed_between_ticks = self._PRNG.exponential(size=(self._n_PEs, self.n_iterations))

		# waking times for every PE (for every PE, as many as the number of iterations so that, in principle, it may
		# always be the selected PE
		ticks_absolute_time = time_elapsed_between_ticks.cumsum(axis=1)

		# these are the indexes of the PEs that will wake up to exchange statistics with a neighbour (notice that a
		# certain PE can show up several times)
		# [0] is because we only care about the index of the waking PE (and not the instant)
		# [:self.n_iterations] is because we only consider the "self.n_iterations" earliest wakings
		i_waking_PEs = np.unravel_index(
			np.argsort(ticks_absolute_time, axis=None), (self._n_PEs, self.n_iterations)
		)[0][:self.n_iterations]

		for i_PE in i_waking_PEs:

			i_selected_neighbor = self._PRNG.choice(self._PEs_neighbors[i_PE])

			# average of the Qs
			DPF.PEs[i_PE]._Q = DPF.PEs[i_selected_neighbor]._Q\
				= (DPF.PEs[i_PE]._Q + DPF.PEs[i_selected_neighbor]._Q)/2

			# average of the nus
			DPF.PEs[i_PE]._nu = DPF.PEs[i_selected_neighbor]._nu\
				= (DPF.PEs[i_PE]._nu + DPF.PEs[i_selected_neighbor]._nu) / 2

		for PE in DPF.PEs:

			# an estimate of (n times) the *global* covariance...
			covariance = np.linalg.inv(PE._Q)

			# ...and another for the *global* mean
			# >= python 3.5 / numpy 1.10
			# mean = covariance @ PE._nu
			mean = np.dot(covariance, PE._nu)

			PE.samples = self.truncate_samples(np.random.multivariate_normal(mean, covariance, size=self._n_particles_per_PE).T)

	def messages(self):

		return np.NaN

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


class GaussianMixturesExchangeRecipe(ExchangeRecipe):

	def __init__(
			self, processing_elements_topology, n_particles_per_PE, ad_hoc_parameters, resampling_algorithm, PRNG):

		super().__init__(processing_elements_topology)

		self._n_particles_per_PE = n_particles_per_PE
		self._resampling_algorithm = resampling_algorithm

		self._PRNG = PRNG

		self._C = ad_hoc_parameters["number_of_components"]
		self._n_particles_for_fusion = ad_hoc_parameters["number_of_particles_for_fusion"]
		self._epsilon = ad_hoc_parameters["epsilon"]
		self._convergence_threshold = ad_hoc_parameters["convergence_threshold"]
		self._convergence_n_maximum_iterations = ad_hoc_parameters["convergence_n_maximum_iterations"]

		self._neighbors = self._PEs_topology.get_neighbours()

	def messages(self):

		return np.NaN

	def perform_exchange(self, DPF):

		gaussian_mixtures = [self.learn(PE.samples.T, PE.weights) for PE in DPF.PEs]
		predictive_gaussian_mixtures = [self.learn(PE.samples.T, PE.previous_weights) for PE in DPF.PEs]

		# import code
		# code.interact(local=dict(globals(), **locals()))

		PE_has_converged = np.full(self._n_PEs, False, dtype=bool)
		n_iterations_convergence = 0

		while n_iterations_convergence < self._convergence_n_maximum_iterations and not np.all(PE_has_converged):

			print('seeking convergence #{}...'.format(n_iterations_convergence))

			for i_PE in range(self._n_PEs):

				neighbors_gms = [gaussian_mixtures[i] for i in self._neighbors[i_PE]]

				gaussian_mixture_before_update = copy.deepcopy(gaussian_mixtures[i_PE])

				gaussian_mixtures[i_PE] = self.fusion(gaussian_mixtures[i_PE], neighbors_gms)

				means_frob = np.linalg.norm(gaussian_mixtures[i_PE].means_ - gaussian_mixture_before_update.means_)

				# print('difference for PE {} = {}'.format(i_PE, means_frob))

				PE_has_converged[i_PE] = means_frob < self._convergence_threshold

			n_iterations_convergence += 1

			# means = [g.means_ for g in gaussian_mixtures]
			# covars = [g.covars_ for g in gaussian_mixtures]
			# weights = [g.weights_ for g in gaussian_mixtures]
			#
			# import code
			# code.interact(local=dict(globals(), **locals()))

		if n_iterations_convergence < self._convergence_n_maximum_iterations:

			print(colorama.Fore.GREEN + 'converged!!' + colorama.Style.RESET_ALL)

		else:

			print(colorama.Fore.RED + 'reached maximum number of iterations!!' + colorama.Style.RESET_ALL)

		for i_PE in range(self._n_PEs):

			recovered_gaussian_mixture = self.recovery(gaussian_mixtures[i_PE], predictive_gaussian_mixtures[i_PE])

			DPF.PEs[i_PE].samples = recovered_gaussian_mixture.sample(self._n_particles_per_PE, self._PRNG).T
			DPF.PEs[i_PE].weights = np.full(self._n_particles_per_PE, 1/self._n_particles_per_PE)

	def learn(self, samples, weights):

		i_resampled = self._resampling_algorithm.get_indexes(weights)

		resulting_gm = sklearn.mixture.GMM(self._C, covariance_type='full')
		resulting_gm.fit(samples[i_resampled, :])

		return resulting_gm

	def fusion(self, gaussian_mixture, neighbors_gaussian_mixtures):

		# number of neighbors
		N_k = len(neighbors_gaussian_mixtures)

		samples = gaussian_mixture.sample(self._n_particles_for_fusion, self._PRNG)

		# every column is a sample, every row a different GM
		prob = np.vstack([np.exp(gm.score(samples)) for gm in [gaussian_mixture] + neighbors_gaussian_mixtures])

		# first GM is associated with the PE performing the fusion
		prob[0, :] **= -self._epsilon*N_k

		# the rest of the PEs are the neighbors
		prob[1:, :] **= self._epsilon

		weights = prob.prod(axis=0) + 1e-200

		weights /= weights.sum()

		# import code
		# code.interact(local=dict(globals(), **locals()))

		return self.learn(samples, weights)

	def recovery(self, gaussian_mixture, previous_gaussian_mixture):

		samples = previous_gaussian_mixture.sample(self._n_particles_for_fusion, self._PRNG)

		prob = np.vstack([np.exp(gm.score(samples)) for gm in [gaussian_mixture, previous_gaussian_mixture]])

		weights = (prob[0, :]/prob[1, :])**self._n_PEs + 1e-200

		weights /= weights.sum()

		return self.learn(samples, weights)


