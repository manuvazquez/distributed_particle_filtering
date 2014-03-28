import numpy as np

class ParticleFilter:
	
	def __init__(self,nParticles):
		
		self._nParticles = nParticles
		self._weights = np.empty([1,nParticles])
		
	def initialize(self):
		
		# the weights are assigned equal probabilities
		self._weights.fill(1/nParticles)
		
class ResamplingAlgorithm:
	
	def __init__(self):
		
		pass
	
	def getIndexes(self,weights):
		
		pass
	
class MultinomialResamplingAlgorithm(ResamplingAlgorithm):
	
	def __init__(self):
		
		super().__init__()
		
	def getIndexes(self,weights):
		
		return np.random.choice(range(weights.size), weights.size, p=weights)

class ResampleCriterion:
	
	def __init__(resamplingRatio):
		
		self._resamplingRatio = resamplingRatio
		
	def isResamplingNeeded(self,weights):
		
		# a division by zero may occur...
		try:
			nEffectiveParticles = 1/np.dot(weights,weights)
		except ZeroDivisionError:
			print('ResampleCriterion::isResamplingNeeded: all the weights are zero!!...quitting')
			raise SystemExit(0)
			
		return nEffectiveParticles<(self._resamplingRatio*weights.size)