import numpy as np

class ResamplingAlgorithm:
	
	def __init__(self):
		
		pass
	
	def getIndexes(self,weights):
		"""It returns the indexes of the particles that should be kept after resampling.
		
		Notice that it doesn't perform any "real" resampling"...the real work must be performed somewhere else.
		"""
		pass
	
class MultinomialResamplingAlgorithm(ResamplingAlgorithm):
	
	def __init__(self,PRNG=np.random.RandomState()):
		
		super().__init__()
		
		self._PRNG = PRNG
		
	def getIndexes(self,weights):
		
		return self._PRNG.choice(range(weights.size), weights.size, p=weights)

class ResamplingCriterion:
	
	def isResamplingNeeded(self,weights):
		
		pass

class EffectiveSampleSizeBasedResamplingCriterion(ResamplingCriterion):
	
	def __init__(self,resamplingRatio):
		
		self._resamplingRatio = resamplingRatio
		
	def isResamplingNeeded(self,weights):
		
		super().isResamplingNeeded(weights)
		
		# a division by zero may occur...
		try:
			nEffectiveParticles = 1/np.dot(weights,weights)
		except ZeroDivisionError:
			print('ResampleCriterion::isResamplingNeeded: all the weights are zero!!...quitting')
			raise SystemExit(0)
		
		return nEffectiveParticles<(self._resamplingRatio*weights.size)
	
class AlwaysResamplingCriterion(ResamplingCriterion):
	
	def isResamplingNeeded(self,weights):
		
		super().isResamplingNeeded(weights)
		
		return True