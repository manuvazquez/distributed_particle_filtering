import numpy as np
#import State

class ParticleFilter:
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion):
		
		self._nParticles = nParticles
		self._weights = np.empty([1,nParticles])
		
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		
		# at first...the state is empty
		self._state = None
		
	def initialize(self):
		
		# the weights are assigned equal probabilities
		self._weights.fill(1/nParticles)
		
	def getState(self):
		
		return self._state

	def step(self,observations):
		
		pass

class TrackingParticleFilter(ParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion)
		
		self._stateTransitionKernel = stateTransitionKernel
		self._prior = prior
	
	def initialize(self):
		
		# initial samples...
		self._state = self._prior.sample(self._nParticles)
		
	def step(self,observations):
		
		for i in range(self._nParticles):
			
			self._state[:,i:i+1] = self._stateTransitionKernel.nextState(self._state[:,i:i+1])