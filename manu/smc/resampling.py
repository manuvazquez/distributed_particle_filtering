import numpy as np


class ResamplingAlgorithm:
	
	def __init__(self):
		
		pass
	
	def get_indexes(self, weights, n):
		"""It returns the indexes of the particles that should be kept after resampling.
		
		Notice that it doesn't perform any "real" resampling"...the real work must be performed somewhere else.

		Parameters
		----------
		weights : array_like
			The weights of the particles.
		n : int, optional
			The number of indexes requested

		Returns
		-------
		indices : array_like
			The indices of the selected particles

		"""
		pass


class MultinomialResamplingAlgorithm(ResamplingAlgorithm):
	
	def __init__(self, PRNG=np.random.RandomState()):
		
		super().__init__()
		
		self._PRNG = PRNG
		
	def get_indexes(self, weights, n=None):
		
		if not n:

			n = weights.size
		
		return self._PRNG.choice(range(weights.size), n, p=weights)


class ResamplingCriterion:
	
	def is_resampling_needed(self, weights):
		
		pass


class EffectiveSampleSizeBasedResamplingCriterion(ResamplingCriterion):
	
	def __init__(self, resampling_ratio):
		
		self._resampling_ratio = resampling_ratio
		
	def is_resampling_needed(self, weights):
		
		super().is_resampling_needed(weights)
		
		# a division by zero may occur...
		try:
			
			n_effective_particles = 1/np.dot(weights, weights)
			
		except ZeroDivisionError:
			
			raise Exception('all the weights are zero!!')
			
		return n_effective_particles < (self._resampling_ratio * weights.size)


class AlwaysResamplingCriterion(ResamplingCriterion):
	
	def is_resampling_needed(self, weights):
		
		super().is_resampling_needed(weights)
		
		return True
