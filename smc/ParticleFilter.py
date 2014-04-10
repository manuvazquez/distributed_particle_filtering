import numpy as np
import State

class ParticleFilter:
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion):
		
		self._nParticles = nParticles
		
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		
	def initialize(self):

		pass
		
	def step(self,observations):
		
		pass
	
	def getState(self):
		
		pass

# =========================================================================================================

class PlainParticleFilter(ParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,aggregatedWeight):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion)
		
		self._weights = np.empty(nParticles)
		
		# at first...the state is empty
		self._state = None

		self._aggregatedWeight = aggregatedWeight

	def initialize(self):
		
		super().initialize()
		
		# the weights are assigned equal probabilities
		self._weights.fill(self._aggregatedWeight/self._nParticles)

	def step(self,observations):
		
		super().step(observations)
		
	def getState(self):
		
		return self._state
	
	def normalizeWeights(self):
		
		self._weights /= self._weights.sum()
		
	def keepParticles(self,indexes):
		
		self._state = self._state[:,indexes]
		self._weights.fill(self._aggregatedWeight/self._nParticles)
		
	def getParticle(self,index):
		
		return (self._state[:,index:index+1],self._weights[index])
	
	def setParticle(self,index,particle):
		
		self._state[:,index:index+1] = particle[0]
		self._weights[index] = particle[1]

# =========================================================================================================

class TrackingParticleFilter(PlainParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,aggregatedWeight=1.0):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion,aggregatedWeight)
		
		# the state equation is encoded in the transition kernel
		self._stateTransitionKernel = stateTransitionKernel
		
		self._prior = prior
		
		self._sensors = sensors
		
		self._rangeSensors = range(len(sensors))
	
	def initialize(self):
		
		# let the parent do its thing...
		super().initialize()
		
		# initial samples...
		self._state = self._prior.sample(self._nParticles)
		
	def step(self,observations):
		
		super().step(observations)
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._stateTransitionKernel.nextState(self._state[:,i:i+1]) for i in range(self._nParticles)])
		
		# for each sensor, we compute the likelihood of EVERY particle (position)
		likelihoods = np.array([self._sensors[i].likelihood(observations[i],State.position(self._state)) for i in self._rangeSensors])
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		likelihoodsProduct = likelihoods.prod(axis=0)
		
		# the weights are updated
		self._weights *= likelihoodsProduct
		
		self.updateAggregatedWeight()
		
		# the normalized weights are computed
		normalizedWeights = self._weights / self._aggregatedWeight

		# if resampling is needed
		if self._resamplingCriterion.isResamplingNeeded(normalizedWeights):
			
			# the resampling algorithm is used to decide which particles to keep
			iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(normalizedWeights)
			
			# actual resampling
			self.keepParticles(iParticlesToBeKept)
			
	def updateAggregatedWeight(self):
		
		# the aggregated weight is simply the sum of the non-normalized weights
		self._aggregatedWeight = self._weights.sum()

# =========================================================================================================

class ParticleFiltersCompoundWithDRNA(ParticleFilter):
	
	def __init__(self,nPEs,exchangePeriod,exchangeMap,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors):
		
		super().__init__(nPEs*nParticlesPerPE,resamplingAlgorithm,resamplingCriterion)
		
		self._nParticlesPerPE = nParticlesPerPE

		# the PEs list will be looped through many times...so, for the sake of convenience
		self._rPEs = range(nPEs)

		# the particle filters are built
		self._PEs = [TrackingParticleFilter(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,1.0/nPEs) for i in self._rPEs]
		
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