import numpy as np
import numpy.random

class PEsNetwork:
	
	def __init__(self,nPEs,nParticlesPerPE,exchangePercentage,neighbours):
		
		self._nPEs = nPEs
		
		# it is assumed that each PE contains this number of particles...
		self._nParticlesPerPE = nParticlesPerPE
		
		# ...out of which, this will be exchanged
		self._nParticlesToBeExchanged = int(np.ceil(nParticlesPerPE*exchangePercentage))
		self._rParticlesToBeExchanged = range(self._nParticlesToBeExchanged)
		
		# each element in the list is another list specifying the neighbours of the corresponding PE
		self._neighbours = neighbours
		
	def getExchangeTuples(self):
		
		# an array to keep tabs of pairs of PEs already processed
		alreadyProcessedPEs = np.zeros((self._nPEs,self._nPEs),dtype=bool)
		
		exchangeTuples = []
		
		for iPE,neighboursPE in enumerate(self._neighbours):
			
			for neighbour in neighboursPE:
				
				if not alreadyProcessedPEs[iPE,neighbour]:

					# the particles to be exchanged are chosen randomly (each column corresponds to an exchange, first row for the 1st PE, second row for the 2nd one)
					iParticlesToExchange = numpy.random.random_integers(0,self._nParticlesPerPE-1,(2,self._nParticlesToBeExchanged))
				
					# new "exchange tuple"s are generated
					exchangeTuples.extend([[iPE,iParticlesToExchange[0,i],neighbour,iParticlesToExchange[1,i]] for i in self._rParticlesToBeExchanged])

					# we "mark" this pair of PEs as already processed (only "alreadyProcessedPEs[neighbour,iPe]" should be accessed later on, though...)
					alreadyProcessedPEs[neighbour,iPE] = alreadyProcessedPEs[iPE,neighbour] = True
					
		return exchangeTuples