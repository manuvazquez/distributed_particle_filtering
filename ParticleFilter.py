import numpy as np
import State

class ParticleFilter:
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,aggregatedWeight):
		
		self._nParticles = nParticles
		self._weights = np.empty(nParticles)
		
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		
		# at first...the state is empty
		self._state = None

		self._aggregatedWeight = aggregatedWeight

	def initialize(self):
		
		# the weights are assigned equal probabilities
		self._weights.fill(self._aggregatedWeight/self._nParticles)
		
	def getState(self):
		
		return self._state

	def step(self,observations):
		
		pass
	
	def normalizeWeights(self):
		
		self._weights /= self._weights.sum()
		
	def keepParticles(self,indexes):
		
		self._state = self._state[:,indexes]
		self._weights.fill(self._aggregatedWeight/self._nParticles)

class TrackingParticleFilter(ParticleFilter):
	
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
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._stateTransitionKernel.nextState(self._state[:,i:i+1]) for i in range(self._nParticles)])
		
		# for each sensor, we compute the likelihood of EVERY particle (position)
		likelihoods = np.array([self._sensors[i].likelihood(observations[i],State.position(self._state)) for i in self._rangeSensors])
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		likelihoodsProduct = likelihoods.prod(axis=0)
		
		# the weights are updated
		self._weights *= likelihoodsProduct
		
		# the aggregated weight is simply the sum of the non-normalized weights
		self._aggregatedWeight = self._weights.sum()
		
		# the normalized weights are computed
		normalizedWeights = self._weights / self._aggregatedWeight

		# if resampling is needed
		if self._resamplingCriterion.isResamplingNeeded(normalizedWeights):
			
			# the resampling algorithm is used to decide which particles to keep
			iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(normalizedWeights)
			
			# actual resampling
			self.keepParticles(iParticlesToBeKept)