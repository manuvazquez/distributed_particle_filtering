import numpy as np
import numpy.random

class PEsNetwork:
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage):
		
		self._nPEs = nPEs
		
		# it is assumed that each PE contains this number of particles...
		self._nParticlesPerPE = nParticlesPerPE
		
		# ...out of which, this will be exchanged
		self._nParticlesToBeExchanged = int(np.ceil(nParticlesPerPE*exchangePercentage))
		
		# no neighbours for the different PEs are specified in this class...this should be done in the children classes
		self._neighbours = []
		
		# indexes of the particles...just for the sake of efficiency (this array will be used many times)
		self._iParticles = np.arange(nParticlesPerPE)
		
	def getNumberOfParticlesToBeExchanged(self):
		
		return self._nParticlesToBeExchanged
	
	def getExchangeTuples(self):
		
		# an array to keep tabs of pairs of PEs already processed
		alreadyProcessedPEs = np.zeros((self._nPEs,self._nPEs),dtype=bool)
		
		# in order to keep tabs on which particles a given PE has already "promised" to exchange
		iNotSwappedYetParticles = np.ones((self._nPEs,self._nParticlesPerPE),dtype=bool)
		
		exchangeTuples = []
		
		for iPE,neighboursPE in enumerate(self._neighbours):
			
			for iNeighbour in neighboursPE:
				
				if not alreadyProcessedPEs[iPE,iNeighbour]:

					# the particles to be exchanged are chosen randomly (with no replacement) for both, the considered PE...
					iParticlesToExchangeWithinPE = numpy.random.choice(self._iParticles[iNotSwappedYetParticles[iPE,:]],size=self._nParticlesToBeExchanged,replace=False)
					
					# ...and the corresponding neighbour
					iParticlesToExchangeWithinNeighbour = numpy.random.choice(self._iParticles[iNotSwappedYetParticles[iNeighbour,:]],size=self._nParticlesToBeExchanged,replace=False)

					# new "exchange tuple"s are generated
					exchangeTuples.extend([[iPE,iParticleWithinPE,iNeighbour,iParticleWithinNeighbour] for iParticleWithinPE,iParticleWithinNeighbour in zip(iParticlesToExchangeWithinPE,iParticlesToExchangeWithinNeighbour)])
					
					# these PEs (the one considered in the main loop and the neighbour being processed) should not exchange the selected particles (different in each case) with other PEs
					iNotSwappedYetParticles[iPE,iParticlesToExchangeWithinPE] = False
					iNotSwappedYetParticles[iNeighbour,iParticlesToExchangeWithinNeighbour] = False

					# we "mark" this pair of PEs as already processed (only "alreadyProcessedPEs[iNeighbour,iPe]" should be accessed later on, though...)
					alreadyProcessedPEs[iNeighbour,iPE] = alreadyProcessedPEs[iPE,iNeighbour] = True
					
		return exchangeTuples

class Customized(PEsNetwork):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,neighbours):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage)
			
		# each element in the list is another list specifying the neighbours of the corresponding PE
		self._neighbours = neighbours
		
class Ring(PEsNetwork):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage)
		
		self._neighbours = [[(i-1) % nPEs,(i+1) % nPEs] for i in range(nPEs)]

class Mesh(PEsNetwork):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,potentialNeighboursRelativePosition,nRows,nCols):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage)
		
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
					
					# we compute its position
					iNeighbour,jNeighbour = i+neighbourRelativePosition[0],j+neighbourRelativePosition[1]
					
					# if the position corresponds to that of a PE (i.e., it is within the PEs array)
					if (0 <= iNeighbour < nRows) and (0 <= jNeighbour < nCols):
						
						# we add this neighbour to the list
						currentPEneighbours.append(arrayedPEs[iNeighbour,jNeighbour])
				
				# the list of neighbours of this PE is added to the list of lists of neighbours
				self._neighbours.append(currentPEneighbours)

class FullyConnected(PEsNetwork):
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage):
		
		super().__init__(nPEs,nParticlesPerPE,exchangePercentage)
		
		self._neighbours = [list(range(nPEs)) for i in range(nPEs)]
		
		for i in range(nPEs):
			
			del self._neighbours[i][i]