import numpy as np
import math

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

class CentralizedTargetTrackingParticleFilter(ParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,aggregatedWeight=1.0):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion)
		
		# a vector with the weights is created...but not initialized (that must be done by the "initialize" method)
		self._weights = np.empty(nParticles)
		
		# the state equation is encoded in the transition kernel
		self._stateTransitionKernel = stateTransitionKernel
		
		# the prior is needed to inialize the state
		self._prior = prior
		
		# the sensors are kept
		self._sensors = sensors
		
		# we will frequently loop through all the sensors, so...for the sake of convenience
		self._rSensors = range(len(sensors))
		
		# this variable just keeps tabs on the sum of all the weights
		self._aggregatedWeight = aggregatedWeight
	
	def initialize(self):
		
		# let the parent do its thing...
		super().initialize()
		
		# initial samples...
		self._state = self._prior.sample(self._nParticles)
		
		# the weights are assigned equal probabilities
		self._weights.fill(1.0/self._nParticles)
		
	def step(self,observations):
		
		super().step(observations)
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._stateTransitionKernel.nextState(self._state[:,i:i+1]) for i in range(self._nParticles)])
		
		# for each sensor, we compute the likelihood of EVERY particle (position)
		likelihoods = np.array([self._sensors[i].likelihood(observations[i],State.position(self._state)) for i in self._rSensors])
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		likelihoodsProduct = likelihoods.prod(axis=0)
		
		# the weights are updated
		self._weights *= likelihoodsProduct
		
		# the aggregated weight is kept up to date at any time
		self.updateAggregatedWeight()
		
		# the normalized weights are computed
		normalizedWeights = self._weights / self._aggregatedWeight

		# if resampling is needed
		if self._resamplingCriterion.isResamplingNeeded(normalizedWeights):
			
			# the resampling algorithm is used to decide which particles to keep
			iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(normalizedWeights)
			
			# actual resampling
			self.keepParticles(iParticlesToBeKept)

	def getState(self):
		
		return self._state

	def keepParticles(self,indexes):
		
		self._state = self._state[:,indexes]
		self._weights.fill(1.0/self._nParticles)
		
		# we forced this above
		self._aggregatedWeight = 1.0
		
	def getParticle(self,index):
		
		return (self._state[:,index:index+1],self._weights[index])
	
	def setParticle(self,index,particle):
		
		self._state[:,index:index+1] = particle[0]
		self._weights[index] = particle[1]

	def updateAggregatedWeight(self):
		
		# the aggregated weight is simply the sum of the non-normalized weights
		self._aggregatedWeight = self._weights.sum()

	def computeMean(self):
		
		normalizedWeights = self._weights / self._aggregatedWeight
		
		#import code
		#code.interact(local=dict(globals(), **locals()))
		
		np.tile(normalizedWeights,(self._state.shape[0],1))


# =========================================================================================================

class EmbeddedTargetTrackingParticleFilter(CentralizedTargetTrackingParticleFilter):
	
	def initialize(self):
		
		# the grandfather's method...
		ParticleFilter.initialize(self)
		
		# state initialization...just like in the centralized version
		self._state = self._prior.sample(self._nParticles)
		
		# NOT exactly the same in the centralized version
		self._weights.fill(self._aggregatedWeight/self._nParticles)

	def keepParticles(self,indexes):
		
		# just like in the centralized version
		self._state = self._state[:,indexes]
		
		# NOT exactly the same in the centralized version
		self._weights.fill(self._aggregatedWeight/self._nParticles)

	def getAggregatedWeight(self):
		
		return self._aggregatedWeight

# =========================================================================================================

class TargetTrackingParticleFilterWithDRNA(ParticleFilter):
	
	def __init__(self,nPEs,exchangePeriod,exchangeMap,c,epsilon,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors):
		
		super().__init__(nPEs*nParticlesPerPE,resamplingAlgorithm,resamplingCriterion)
		
		self._nPEs = nPEs

		# the PEs list will be looped through many times...so, for the sake of convenience
		self._rPEs = range(nPEs)
		
		self._nParticlesPerPE = nParticlesPerPE

		# the particle filters are built
		self._PEs = [EmbeddedTargetTrackingParticleFilter(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,1.0/nPEs) for i in self._rPEs]
		
		# a exchange of particles among PEs will happen every...
		self._exchangePeriod = exchangePeriod
		
		# ...time instants, according to the map
		self._exchangeMap = exchangeMap
		
		# parameters for checking the aggregated weights degeneration
		self._c = c
		self._epsilon = epsilon
		
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
		
		if self.degeneratedAggregatedWeights():
			
			print('aggregated weights degenerated...')
		
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
			
			
	#def generateExchangeTuples(self,nParticlesPerNeighbour=3,nSurroundingNeighbours=2):
		
		#pass
	
	def getState(self):
		
		# the state from every PE is gathered together
		return np.hstack([self._PEs[i].getState() for i in self._rPEs])
	
	def getAggregatedWeights(self):
		
		return np.array([self._PEs[i].getAggregatedWeight() for i in self._rPEs])
	
	def degeneratedAggregatedWeights(self):

		if self.getAggregatedWeights().max() > self._c/math.pow(self._nPEs,1.0-self._epsilon):
			
			return True