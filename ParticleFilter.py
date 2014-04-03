import numpy as np
import State

class ParticleFilter:
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion):
		
		self._nParticles = nParticles
		self._weights = np.empty(nParticles)
		
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		
		# at first...the state is empty
		self._state = None
		
	def initialize(self):
		
		# the weights are assigned equal probabilities
		self._weights.fill(1/self._nParticles)
		
	def getState(self):
		
		return self._state

	def step(self,observations):
		
		pass
	
	def normalizeWeights(self):
		
		self._weights /= self._weights.sum()
		
	def keepParticles(self,indexes):
		
		self._state = self._state[:,indexes]
		self._weights.fill(1/self._nParticles)

class TrackingParticleFilter(ParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion)
		
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
		
		# weights are normalized
		self.normalizeWeights()

		# if resampling is needed
		if self._resamplingCriterion.isResamplingNeeded(self._weights):
			
			# the resampling algorithm is used to decide which particles to keep
			iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(self._weights)
			
			# actual resampling
			self.keepParticles(iParticlesToBeKept)