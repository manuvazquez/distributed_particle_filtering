import ParticleFilter
import numpy as np

class ParticleFiltersCompoundWithDRNA(ParticleFilter.ParticleFilter):
	
	def __init__(self,nPEs,exchangePeriod,exchangeMap,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors):
		
		super().__init__(nPEs*nParticlesPerPE,resamplingAlgorithm,resamplingCriterion)
		
		self._nParticlesPerPE = nParticlesPerPE

		# the PEs list will be looped through many times...so, for the sake of convenience
		self._rPEs = range(nPEs)

		# the particle filters are built
		self._PEs = [ParticleFilter.TrackingParticleFilter(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,1.0/nPEs) for i in self._rPEs]
		
		# a exchange of particles among PEs will happen every...
		self._exchangePeriod = exchangePeriod
		
		# ...seconds, according to the map
		self._exchangeMap = exchangeMap
		
		# the number of exchanges will be used repeatedly in loops
		self._rExchanges = range(len(self._exchangeMap))
		
		# because of the exchange period, we must keep track of the elapsed (discreet) time instants
		self._n = 0
		
	def initialize(self):
		
		super().initialize()
		
		# all the PFs are initialized
		for i in self._rPEs:
			
			self._PEs[i].initialize()
			
	def step(self,observations):
		
		super().step(observations)
		
		# a step is taken in every PF (ideally, this would occur concurrently)
		for i in self._rPEs:
			
			self._PEs[i].step(observations)
		
		# a new time instant has elapsed
		self._n += 1
		
		if self._n % self._exchangePeriod:
			
			self.exchangeParticles()
			
			# after the exchange, the aggregated weight of every PE must be computed updated
			for i in self._rPEs:
				
				self._PEs[i].updateAggregatedWeight()
		
	def exchangeParticles(self):
		
		# all the exchanges specified in the map are carried out
		for i in self._rExchanges:
			
			# for the sake of convenience when referring to the current exchange tuple
			exchangeTuple = self._exchangeMap[i]
			
			# auxiliar variable storing the first particle
			particle = self._PEs[exchangeTuple[0]].getParticle(exchangeTuple[1])
			
			# the first particle is set to the second
			self._PEs[exchangeTuple[0]].setParticle(
				exchangeTuple[1],
				self._PEs[exchangeTuple[2]].getParticle(exchangeTuple[3])
				)
			
			self._PEs[exchangeTuple[2]].setParticle(exchangeTuple[3],particle)
			
			
	def generateExchangeTuples(self,nParticlesPerNeighbour=3,nSurroundingNeighbours=2):
		
		pass
	
	def getState(self):
		
		# the state from every PE is gathered together
		return np.hstack([self._PEs[i].getState() for i in self._rPEs])