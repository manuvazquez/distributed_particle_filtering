import collections
import numpy as np

class ExchangeRecipe:

	def __init__(self,PEsTopology,nParticlesPerPE,exchangePercentage,PRNG=np.random.RandomState()):
		
		"""Computes which particles from which PEs are exchanged.
		
		"""
		
		# the number of PEs...
		self._nPEs = PEsTopology.getNumberOfPEs()
		
		# ...and the neighbours of each PE are extracted from the topology
		neighbours = PEsTopology.getNeighbours()
		
		# a named tuple for a more intutive access to a "exchange tuple"
		ExchangeTuple = collections.namedtuple('ExchangeTuple',['iPE','iParticleWithinPE','iNeighbour','iParticleWithinNeighbour'])
		
		# indexes of the particles...just for the sake of efficiency (this array will be used many times)
		iParticles = np.arange(nParticlesPerPE)
		
		# it is computed accounting for the maximum number of neighbours a given PE can have
		self._nParticlesExchangedBetweenTwoNeighbours = int((nParticlesPerPE*exchangePercentage)//max([len(neighbourhood) for neighbourhood in neighbours]))
		
		if self._nParticlesExchangedBetweenTwoNeighbours is 0:
			
			raise Exception('no particles are to be shared by a PE with its neighbours')
		
		# an array to keep tabs on pairs of PEs already processed
		alreadyProcessedPEs = np.zeros((self._nPEs,self._nPEs),dtype=bool)
		
		# in order to keep tabs on which particles a given PE has already "promised" to exchange
		iNotSwappedYetParticles = np.ones((self._nPEs,nParticlesPerPE),dtype=bool)
		
		# named tuples as defined above, each representing a exchange
		self._exchangeTuples = []
		
		# a list in which the i-th element is also a list containing tuples of the form (<neighbour index>,<(numpy) array with the indices of particles to be exchanged with that neighbour>)
		self._neighboursWithParticles = [[] for i in range(self._nPEs)]
		
		for iPE,neighboursPE in enumerate(neighbours):
			
			for iNeighbour in neighboursPE:
				
				if not alreadyProcessedPEs[iPE,iNeighbour]:

					# the particles to be exchanged are chosen randomly (with no replacement) for both, the considered PE...
					iParticlesToExchangeWithinPE = PRNG.choice(iParticles[iNotSwappedYetParticles[iPE,:]],size=self._nParticlesExchangedBetweenTwoNeighbours,replace=False)
					
					# ...and the corresponding neighbour
					iParticlesToExchangeWithinNeighbour = PRNG.choice(iParticles[iNotSwappedYetParticles[iNeighbour,:]],size=self._nParticlesExchangedBetweenTwoNeighbours,replace=False)

					# new "exchange tuple"s are generated
					self._exchangeTuples.extend([ExchangeTuple(iPE=iPE,iParticleWithinPE=iParticleWithinPE,iNeighbour=iNeighbour,iParticleWithinNeighbour=iParticleWithinNeighbour)
							for iParticleWithinPE,iParticleWithinNeighbour in zip(iParticlesToExchangeWithinPE,iParticlesToExchangeWithinNeighbour)])
					
					# these PEs (the one considered in the main loop and the neighbour being processed) should not exchange the selected particles (different in each case) with other PEs
					iNotSwappedYetParticles[iPE,iParticlesToExchangeWithinPE] = False
					iNotSwappedYetParticles[iNeighbour,iParticlesToExchangeWithinNeighbour] = False

					# we "mark" this pair of PEs as already processed (only "alreadyProcessedPEs[iNeighbour,iPe]" should be accessed later on, though...)
					alreadyProcessedPEs[iNeighbour,iPE] = alreadyProcessedPEs[iPE,iNeighbour] = True
					
					self._neighboursWithParticles[iPE].append((iNeighbour,iParticlesToExchangeWithinPE))
					self._neighboursWithParticles[iNeighbour].append((iPE,iParticlesToExchangeWithinNeighbour))
		
	def getExchangeTuples(self):
		
		"""Returns the exchange map computed in the initializer.
		
		Returns
		-------
		exchangeTuples: list
			A list with tuples ("ExchangeTuple") of the form (<PE>,<particle within PE>,<neighbour>,<particle within neighbour>).
		exchangeTuples: list
			A list of lists, one per PE, in which every list contains tuples of the form (<neighbour>,<list of particles exchanged with the neighbour).
		"""
			
		return self._exchangeTuples,self._neighboursWithParticles
	
	def getNumberOfPEs(self):
		
		return self._nPEs
	
	@property
	def nParticlesExchangedBetweenTwoNeighbours(self):
		
		"""The number of particles that are to be exchanged between a couple of neighbours.
		
		Returns
		-------
		self._nParticlesExchangedBetweenTwoNeighbours: int
			number of particles
		"""
		
		return self._nParticlesExchangedBetweenTwoNeighbours