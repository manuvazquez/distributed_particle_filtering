import collections
import abc
import numpy as np
import scipy

import state


class ExchangeRecipe(metaclass=abc.ABCMeta):

	def __init__(self, PEsTopology):

		self._PEs_topology = PEsTopology

		# for the sake of convenience, we keep the number of PEs...
		self._n_PEs = PEsTopology.getNumberOfPEs()

	@abc.abstractmethod
	def perform_exchange(self):

		pass

	@abc.abstractmethod
	def messages(self):

		return


class DRNAExchangeRecipe(ExchangeRecipe):

	def __init__(self, PEsTopology, n_particles_per_PE, exchanged_particles, PRNG=np.random.RandomState()):

		"""Computes which particles from which PEs are exchanged.
		
		"""

		super().__init__(PEsTopology)

		# ...and the all_PEs_neighbours of each PE are extracted from the topology
		all_PEs_neighbours = PEsTopology.get_neighbours()

		# a named tuple for a more intuitive access to a "exchange tuple"
		ExchangeTuple = collections.namedtuple(
			'ExchangeTuple',['iPE', 'iParticleWithinPE', 'iNeighbour', 'iParticleWithinNeighbour'])

		# a named tuple for a more intuitive access to a "exchange tuple"
		NeighbourParticlesTuple = collections.namedtuple(
			'NeighbourParticlesTuple', ['iNeighbour', 'iParticles'])

		# indexes of the particles...just for the sake of efficiency (this array will be used many times)
		iParticles = np.arange(n_particles_per_PE)

		# the number of particles that are to be exchanged between a couple of neighbours is computed (or set)
		if type(exchanged_particles) is int:

			self.n_particles_exchanged_between_neighbours = exchanged_particles

		elif type(exchanged_particles) is float:

			# it is computed accounting for the maximum number of all_PEs_neighbours a given PE can have
			self.n_particles_exchanged_between_neighbours = int((n_particles_per_PE*exchanged_particles)//max([len(neighbourhood) for neighbourhood in all_PEs_neighbours]))

		else:

			raise Exception('type of exchanged_particles is not valid')

		if self.n_particles_exchanged_between_neighbours is 0:

			raise Exception('no particles are to be shared by a PE with its all_PEs_neighbours')

		# an array to keep tabs on pairs of PEs already processed
		already_processed_PEs = np.zeros((self._n_PEs,self._n_PEs),dtype=bool)

		# in order to keep tabs on which particles a given PE has already "promised" to exchange
		i_particles_not_swapped_yet = np.ones((self._n_PEs,n_particles_per_PE),dtype=bool)

		# named tuples as defined above, each representing an exchange
		self._exchangeTuples = []

		# a list in which the i-th element is also a list containing tuples of the form (<neighbour index>,<numpy array>
		#  with the indices of particles to be exchanged with that neighbour>)
		self._neighbours_particles = [[] for i in range(self._n_PEs)]

		for iPE,i_this_PE_neighbours in enumerate(all_PEs_neighbours):

			for iNeighbour in i_this_PE_neighbours:

				if not already_processed_PEs[iPE,iNeighbour]:

					# the particles to be exchanged are chosen randomly (with no replacement) for both, the considered PE...
					i_exchanged_particles_within_PE = PRNG.choice(
						iParticles[i_particles_not_swapped_yet[iPE,:]],
						size=self.n_particles_exchanged_between_neighbours, replace=False)

					# ...and the corresponding neighbour
					i_exchanged_particles_within_neighbour = PRNG.choice(
						iParticles[i_particles_not_swapped_yet[iNeighbour,:]],
						size=self.n_particles_exchanged_between_neighbours, replace=False)

					# new "exchange tuple"s are generated
					self._exchangeTuples.extend([ExchangeTuple(iPE=iPE,iParticleWithinPE=iParticleWithinPE,iNeighbour=iNeighbour,iParticleWithinNeighbour=iParticleWithinNeighbour)
							for iParticleWithinPE,iParticleWithinNeighbour in zip(i_exchanged_particles_within_PE,i_exchanged_particles_within_neighbour)])

					# these PEs (the one considered in the main loop and the neighbour being processed) should not exchange the selected particles (different in each case) with other PEs
					i_particles_not_swapped_yet[iPE,i_exchanged_particles_within_PE] = False
					i_particles_not_swapped_yet[iNeighbour,i_exchanged_particles_within_neighbour] = False

					# we "mark" this pair of PEs as already processed (only "already_processed_PEs[iNeighbour,iPe]" should be accessed later on, though...)
					already_processed_PEs[iNeighbour,iPE] = already_processed_PEs[iPE,iNeighbour] = True

					self._neighbours_particles[iPE].append(NeighbourParticlesTuple(iNeighbour,i_exchanged_particles_within_PE))
					self._neighbours_particles[iNeighbour].append(NeighbourParticlesTuple(iPE,i_exchanged_particles_within_neighbour))

	def getNumberOfPEs(self):

		return self._n_PEs

	def perform_exchange(self, DPF):

		# first, we compile all the particles that are going to be exchanged in an auxiliar variable
		aux = []
		for exchangeTuple in self._exchangeTuples:
			aux.append(
				(DPF._PEs[exchangeTuple.iPE].getParticle(exchangeTuple.iParticleWithinPE),
				DPF._PEs[exchangeTuple.iNeighbour].getParticle(exchangeTuple.iParticleWithinNeighbour)))

		# afterwards, we loop through all the exchange tuples performing the real exchange
		for (exchangeTuple,particles) in zip(self._exchangeTuples,aux):
			DPF._PEs[exchangeTuple.iPE].setParticle(exchangeTuple.iParticleWithinPE,particles[1])
			DPF._PEs[exchangeTuple.iNeighbour].setParticle(exchangeTuple.iParticleWithinNeighbour,particles[0])

	def messages(self):

		# the number of hops between each pair of PEs
		distances = self._PEs_topology.distances_between_PEs()

		# overall number of messages sent/received in an exchange step
		n_messages = 0

		# for every PE (index) along with its list of neighbours
		for iPE,neighboursList in enumerate(self._neighbours_particles):

			# each element of the list is a tuple (<index neighbour>,<indexes of the particles exchanged with that neighbour>)
			for iNeighbour,iParticles in  neighboursList:

				# the number of messages required to send the samples
				n_messages += distances[iPE,iNeighbour]*len(iParticles)*state.nElements

			# we also need to send the aggregated weight to each neighbour
			n_messages += len(neighboursList)

		# import code
		# code.interact(local=dict(globals(), **locals()))

		return n_messages


class MposteriorExchangeRecipe(DRNAExchangeRecipe):

	def perform_exchange(self, DPF):

		for PE, this_PE_neighbours_particles in zip(DPF._PEs, self._neighbours_particles):

			# a list with the subset posterior of each neighbour
			subset_posterior_distributions = [
				(DPF._PEs[neighbour_particles.iNeighbour].getSamplesAt(neighbour_particles.iParticles).T,
				np.full(self.n_particles_exchanged_between_neighbours, 1.0/self.n_particles_exchanged_between_neighbours))
				for neighbour_particles in this_PE_neighbours_particles]

			# a subset posterior obtained from this PE is also added: it encompasses
			# its FIRST "self.n_particles_exchanged_between_neighbours" particles
			subset_posterior_distributions.append(
				(PE.getSamplesAt(range(self.n_particles_exchanged_between_neighbours)).T,
				np.full(self.n_particles_exchanged_between_neighbours,1.0/self.n_particles_exchanged_between_neighbours)))

			# M posterior on the posterior distributions collected above
			joint_particles,joint_weights = DPF.Mposterior(subset_posterior_distributions)

			# the indexes of the particles to be kept
			i_new_particles = DPF._resamplingAlgorithm.getIndexes(joint_weights, PE._nParticles)

			PE.samples = joint_particles[:,i_new_particles]
			PE.logWeights = np.full(PE._nParticles,-np.log(PE._nParticles))
			PE.updateAggregatedWeight()

	def messages(self):

		# same as for DRNA...
		nMessages = super().messages()

		# ...but there is no need for a PE to send its aggregated weight to each one of its neighbours
		for neighboursList in self._neighbours_particles:

			nMessages -= len(neighboursList)

		return nMessages


class IteratedMposteriorExchangeRecipe(MposteriorExchangeRecipe):

	def __init__(self, PEsTopology, n_particles_per_PE, exchanged_particles, number_iterations, PRNG=np.random.RandomState()):

		super().__init__(PEsTopology, n_particles_per_PE, exchanged_particles, PRNG)

		self._number_iterations = number_iterations

	def perform_exchange(self, DPF):

		for _ in range(self._number_iterations):

			super().perform_exchange(DPF)

	def messages(self):

		return super().messages()*self._number_iterations


class LikelihoodConsensusExchangeRecipe(ExchangeRecipe):

	def __init__(self,PEsTopology,maxNumberOfIterations,polynomialDegree):

		super().__init__(PEsTopology)

		self._maxNumberOfIterations = maxNumberOfIterations
		self.polynomialDegree = polynomialDegree

		# a list of lists in which each element yields the neighbors of a PE
		self._neighborhoods = PEsTopology.get_neighbours()

		# Metropolis weights
		# ==========================

		# this will store tuples (<own weight>,<numpy array with weights for each neighbor>)
		self._metropolisWeights = []

		# for the neighbours of every PE
		for neighbors in self._neighborhoods:

			# the number of neighbors of the PE
			nNeighbors = len(neighbors)

			# the weight assigned to each one of its neighbors
			neighborsWeights = np.array([1/(1+max(nNeighbors,len(self._neighborhoods[iNeighbor]))) for iNeighbor in neighbors])

			# the weight assigned to itself is the first element in the tuple
			self._metropolisWeights.append((1-neighborsWeights.sum(),neighborsWeights))

	def perform_exchange(self,DPF):

		# the first iteration of the consensus algorithm
		# ==========================

		# for every PE, along with its neighbors
		for PE,neighbors,weights in zip(DPF._PEs,self._neighborhoods,self._metropolisWeights):

			# a dictionary for storing the "consensed" beta's
			PE.betaConsensus = {}

			# for every combination of exponents, r
			for r in DPF._r_d_tuples:

				#import code
				#code.interact(local=dict(globals(), **locals()))

				PE.betaConsensus[r] = PE.beta[r]*weights[0] + np.array([DPF._PEs[iNeighbor].beta[r] for iNeighbor in neighbors]).dot(weights[1])

		# the remaining iterations of the consensus algorithm
		# ==========================

		# the same operations as above using "betaConsensus" rather than beta
		for _ in range(self._maxNumberOfIterations-1):

			# for every PE, along with its neighbors
			for PE,neighbors,weights in zip(DPF._PEs,self._neighborhoods,self._metropolisWeights):

				# for every combination of exponents, r
				for r in DPF._r_d_tuples:

					PE.betaConsensus[r] = PE.betaConsensus[r]*weights[0] + np.array([DPF._PEs[iNeighbor].betaConsensus[r] for iNeighbor in neighbors]).dot(weights[1])

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
		n_consensus_algorithms = scipy.misc.comb(2*self.polynomialDegree + M, 2*self.polynomialDegree, exact=True) - 1

		# overall number of neighbours: #neighbours of the 1st PE + #neighbours of the 2nd PE +...
		n_neighbours = sum([len(neighbours) for neighbours in self._neighborhoods])

		# each PE sends "n_consensus_algorithms" values to each one of its neighbours, once per iteration...
		n_messages = n_neighbours*n_consensus_algorithms*self._maxNumberOfIterations

		# ...additionally it needs to send each neighbour the number of neighbours it has itself (Metropolis weights)
		n_messages += n_neighbours

		return n_messages