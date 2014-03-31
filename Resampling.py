class ResamplingAlgorithm:
	
	def __init__(self):
		
		pass
	
	def getIndexes(self,weights):
		"""It returns the indexes of the particles that should be kept after resampling.
		
		Notice that it doesn't perform any "real" resampling"...the real work must be performed somewhere else.
		"""
		pass
	
class MultinomialResamplingAlgorithm(ResamplingAlgorithm):
	
	def __init__(self):
		
		super().__init__()
		
	def getIndexes(self,weights):
		
		return np.random.choice(range(weights.size), weights.size, p=weights)

class ResampleCriterion:
	
	def __init__(self,resamplingRatio):
		
		self._resamplingRatio = resamplingRatio
		
	def isResamplingNeeded(self,weights):
		
		# a division by zero may occur...
		try:
			nEffectiveParticles = 1/np.dot(weights,weights)
		except ZeroDivisionError:
			print('ResampleCriterion::isResamplingNeeded: all the weights are zero!!...quitting')
			raise SystemExit(0)
			
		return nEffectiveParticles<(self._resamplingRatio*weights.size)