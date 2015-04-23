import numpy as np
import math
import collections

class Topology:
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		self._nPEs = nPEs
		self._nParticlesPerPE = nParticlesPerPE
		self._exchangePercentage = exchangePercentage
		self._PRNG = PRNG
		
		self._topologySpecificParameters = topologySpecificParameters
		
		# indexes of the particles...just for the sake of efficiency (this array will be used many times)
		self._iParticles = np.arange(nParticlesPerPE)
		
		# a named tuple for a more intutive access to a "exchange tuple"
		self._exchangeTuple = collections.namedtuple('ExchangeTuple',['iPE','iParticleWithinPE','iNeighbour','iParticleWithinNeighbour'])

		# these are ultimately what characterizes the topology, and are "None" until "deploy" is called
		self._exchangeTuples,self._neighboursWithParticles = None,None

	def getExchangeTuples(self):
		
		"""Returns the connections between the different PEs at a particle level.
		
		Returns
		-------
		exchangeTuples: list
			A list with tuples ("ExchangeTuple") of the form (<PE>,<particle within PE>,<neighbour>,<particle within neighbour>).
		exchangeTuples: list
			A list of lists, one per PE, in which every list contains tuples of the form (<neighbour>,<list of particles exchanged with the neighbour).
		"""
		
		if self._exchangeTuples==None:
			
			self.deploy()
			
		return self._exchangeTuples,self._neighboursWithParticles

	def deploy(self):
		
		"""Sets up the connections between the different PEs at a particle level.
		
		"""
		
		# an array to keep tabs of pairs of PEs already processed
		alreadyProcessedPEs = np.zeros((self._nPEs,self._nPEs),dtype=bool)
		
		# in order to keep tabs on which particles a given PE has already "promised" to exchange
		iNotSwappedYetParticles = np.ones((self._nPEs,self._nParticlesPerPE),dtype=bool)
		
		# named tuples as defined above, each representing a exchange
		self._exchangeTuples = []
		
		# a list in which the i-th element is also a list containing tuples of the form (<neighbour index>,<(numpy) array with the indices of particles to be exchanged with that neighbour>)
		self._neighboursWithParticles = [[] for i in range(self._nPEs)]
		
		for iPE,neighboursPE in enumerate(self._neighbours):
			
			for iNeighbour in neighboursPE:
				
				if not alreadyProcessedPEs[iPE,iNeighbour]:

					# the particles to be exchanged are chosen randomly (with no replacement) for both, the considered PE...
					iParticlesToExchangeWithinPE = self._PRNG.choice(self._iParticles[iNotSwappedYetParticles[iPE,:]],size=self.nParticlesExchangedBetweenTwoNeighbours,replace=False)
					
					# ...and the corresponding neighbour
					iParticlesToExchangeWithinNeighbour = self._PRNG.choice(self._iParticles[iNotSwappedYetParticles[iNeighbour,:]],size=self.nParticlesExchangedBetweenTwoNeighbours,replace=False)

					# new "exchange tuple"s are generated
					self._exchangeTuples.extend([self._exchangeTuple(iPE=iPE,iParticleWithinPE=iParticleWithinPE,iNeighbour=iNeighbour,iParticleWithinNeighbour=iParticleWithinNeighbour)
							for iParticleWithinPE,iParticleWithinNeighbour in zip(iParticlesToExchangeWithinPE,iParticlesToExchangeWithinNeighbour)])
					
					# these PEs (the one considered in the main loop and the neighbour being processed) should not exchange the selected particles (different in each case) with other PEs
					iNotSwappedYetParticles[iPE,iParticlesToExchangeWithinPE] = False
					iNotSwappedYetParticles[iNeighbour,iParticlesToExchangeWithinNeighbour] = False

					# we "mark" this pair of PEs as already processed (only "alreadyProcessedPEs[iNeighbour,iPe]" should be accessed later on, though...)
					alreadyProcessedPEs[iNeighbour,iPE] = alreadyProcessedPEs[iPE,iNeighbour] = True
					
					self._neighboursWithParticles[iPE].append((iNeighbour,iParticlesToExchangeWithinPE))
					self._neighboursWithParticles[iNeighbour].append((iPE,iParticlesToExchangeWithinNeighbour))
		
		return self._exchangeTuples,self._neighboursWithParticles
	
	def getNumberOfPEs(self):
		
		return self._nPEs
	
	def getNeighbours(self):
		
		"""Get the list of neighbours of every PE.
		
		Returns
		-------
		samples: list of lists
			Each list contains the indexes of the neighbours of the corresponding PE.
		"""
		
		return self._neighbours
	
	@property
	def nParticlesExchangedBetweenTwoNeighbours(self):
		
		"""The number of particles that are to be exchanged between a couple of neighbours.
		
		Returns
		-------
		nParticlesExchangedBetweenTwoNeighbours: int
			number of particles
		"""
		
		# it is computed accounting for the maximum number of neighbours a given PE can have
		res = int((self._nParticlesPerPE*self._exchangePercentage)//max([len(neighbourhood) for neighbourhood in self._neighbours]))
		
		if res is 0:
			
			raise Exception('no particles are to be shared by a PE with its neighbours')
		
		return res

class Customized(Topology):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		# each element in the list is another list specifying the neighbours of the corresponding PE
		self._neighbours = self._topologySpecificParameters["neighbourhoods"]
		
class Ring(Topology):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		self._neighbours = [[(i-1) % nPEs,(i+1) % nPEs] for i in range(nPEs)]

class Mesh(Topology):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		potentialNeighboursRelativePosition = self._topologySpecificParameters["neighbours"]
		nRows,nCols = self._topologySpecificParameters["geometry"]
		
		assert nRows*nCols == nPEs
		
		# for the sake of clarity, and in order to avoid some computations...
		arrayedPEs = np.arange(nPEs).reshape((nRows,nCols),order='F')
		
		self._neighbours = []
		
		for j in range(nCols):
			for i in range(nRows):
				
				# here we store the neighbours of the PE being processed
				currentPEneighbours = []
				
				# for every potential neighbour
				for neighbourRelativePosition in potentialNeighboursRelativePosition:
					
					# we compute its position...
					
					# if the neighbours of a certain PE are allowed to "wrap around" the other side (either horizontally or vertically) of the mesh...
					if topologySpecificParameters['wraparound']:
					
						iNeighbour,jNeighbour = (i+neighbourRelativePosition[0])%nRows,(j+neighbourRelativePosition[1])%nCols
						
					else:
						
						iNeighbour,jNeighbour = i+neighbourRelativePosition[0],j+neighbourRelativePosition[1]
					
						# if the position does not corresponds to that of a PE (i.e., it is NOT within the PEs array)
						if not ( (0 <= iNeighbour < nRows) and (0 <= jNeighbour < nCols) ):
							
							continue
						
					# we add this neighbour to the list
					currentPEneighbours.append(arrayedPEs[iNeighbour,jNeighbour])
				
				# the list of neighbours of this PE is added to the list of lists of neighbours
				self._neighbours.append(currentPEneighbours)

class ConstantDegreeSimpleGraph(Topology):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		nNeighbours = math.ceil(topologySpecificParameters['number of neighbours as a percentage of the number of PEs']*nPEs)
		
		import networkx as nx
		
		# the Havel-Hakimi algorithm is used to obtain a simple graph with the requested degrees
		graph = nx.havel_hakimi_graph([nNeighbours]*nPEs)
		
		self._neighbours = [graph.neighbors(i) for i in range(nPEs)]
		
class FullyConnected(Topology):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		self._neighbours = [[j for j in range(nPEs) if j!=i] for i in range(nPEs)]
			
class FullyConnectedWithRandomLinksRemoved(FullyConnected):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		selectedPEs = PRNG.choice(nPEs,size=self._topologySpecificParameters["number of links to be removed"])
		
		for PE in selectedPEs:
			
			# the link with this neighbour will be deleted
			neighbour = PRNG.choice(self._neighbours[PE])
			
			# this PE is no longer a neighbour of the one selected above...
			self._neighbours[PE].remove(neighbour)
			
			# ...and viceversa
			self._neighbours[neighbour].remove(PE)

class Physical(Topology):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=np.random.RandomState()):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage,topologySpecificParameters,PRNG=PRNG)
		
		import operator
		
		# for the sake of clarity...
		PEsPositions = topologySpecificParameters['PEs positions']
		
		# a list of sets, each one meant to store the neighbours of the corresponding PE
		res = [set() for i in range(PEsPositions.shape[1])]
		
		for iPE in range(PEsPositions.shape[1]):
			
			# the difference vectors between the position of the current PE and the positions of ALL the PEs, which allow to compute...
			diff = PEsPositions - PEsPositions[:,iPE:iPE+1]
			
			# ...the angles
			angles = np.degrees(np.arctan2(diff[1,:],diff[0,:]))
			
			# ...and the distances
			distances = np.linalg.norm(diff,axis=0)
			
			# we need to use the "and" operator for all the comparisons except the one for the left, which requires the "or"
			for (low,up),op in zip(topologySpecificParameters['ranges of vision for right, up, left and down'],
						  [operator.and_,operator.and_,operator.or_,operator.and_]):
			
				# the index of the PEs that are within the range of angles specified
				iPEsWithinRange, = np.where(op(angles >= low,angles<=up) & (distances>0))
				
				# if any PE found within that "range of vision"
				if iPEsWithinRange.size:
				
					# the neighbour is chosen to be that which is closer to the current PE
					iNeighbour = iPEsWithinRange[np.argmin(distances[iPEsWithinRange])]
					
					# it is added to the list of neighbours of the current PE...
					res[iPE].add(iNeighbour)
					
					# ...and the current PE to the list of neighbours of the selected neighbour (if it wasn't yet)
					res[iNeighbour].add(iPE)
					
		print(res)
		self._neighbours = [list(s) for s in res]