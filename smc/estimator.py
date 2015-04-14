import abc

import numpy as np
import numpy.linalg

import state

def geometric_median(points,max_iterations=100,tolerance=0.001):

	# initial estimate
	estimate = np.median(points,axis=1)
	
	for i in range(max_iterations):
		
		# the norms for the vectors joining the previous estimate with every point
		norms = numpy.linalg.norm(np.subtract(points,estimate[:,np.newaxis]),axis=0)
		
		# is any of the norms is zero?
		is_zero = np.isclose(norms,0.0)
		
		# if one of the norm is zero (there should be one at most)
		if np.any(is_zero):
			
			# we find out its position...
			iZero, = np.where(is_zero)
			
			# ...and the estimate of the median is the corresponding point
			estimate = points[:,iZero[0]]
			
			return estimate
		
		# this is used a couple of times below
		invnorms = 1.0 / norms
		
		# a new estimate according to the Weiszfeld algorithm
		new_estimate = np.multiply(points,invnorms[np.newaxis,:]).sum(axis=1)/invnorms.sum()
	
		# if the new estimate is close enough to the old one...
		if numpy.linalg.norm(new_estimate-estimate)<tolerance:
			
			# ...it gets a pass
			return new_estimate
		
		# ...otherwise, the new estimate becomes will be used in the next iteration
		estimate = new_estimate
		
		#print('iteration {}: {}'.format(i,estimate))
		
	#print('total distance = {}'.format(numpy.linalg.norm(np.subtract(points,estimate[:,np.newaxis]))))
		
	return estimate

class Estimator(metaclass=abc.ABCMeta):
	
	@abc.abstractmethod
	def estimate(self,DPF):
		
		return

class Mean(Estimator):
	
	def estimate(self,DPF):
	
		# the means from all the PEs are stacked (horizontally) in a single array
		jointMeans = np.hstack([PE.computeMean() for PE in DPF._PEs])
		
		return jointMeans.mean(axis=1)[:,np.newaxis]


class WeightedMean(Estimator):
	
	def estimate(self,DPF):
		
		aggregatedWeights = DPF.getAggregatedWeights()

		# the aggregated weights are not necessarily normalized
		normalizedAggregatedWeights = aggregatedWeights/aggregatedWeights.sum()
		
		# notice that "computeMean" will return a numpy array the size of the state (rather than a scalar)
		return np.multiply(np.hstack([PE.computeMean() for PE in DPF._PEs]),normalizedAggregatedWeights).sum(axis=1)[:,np.newaxis]

class Mposterior(Estimator):
	
	def combinePosteriorDistributions(self,DPF,posteriors):
		
		# the Mposterior algorithm is used to obtain a a new distribution
		jointParticles,jointWeights = DPF.Mposterior(posteriors)
		
		return np.multiply(jointParticles,jointWeights).sum(axis=1)[np.newaxis].T
	
	def estimate(self,DPF):
		
		# the (FULL) distributions computed by every PE are gathered in a list of tuples (samples and weights)
		posteriors = [(PE.getState().T,np.exp(PE.logWeights)) for PE in DPF._PEs]
		
		return self.combinePosteriorDistributions(DPF,posteriors)

class PartialMposterior(Mposterior):
	
	def __init__(self,nParticles):
		
		self._nParticles = nParticles
	
	def estimate(self,DPF):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		posteriors = [(PE.getSamplesAt(DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),self._nParticles)).T,
				 np.full(self._nParticles,1.0/self._nParticles)) for PE in DPF._PEs]
		
		return self.combinePosteriorDistributions(DPF,posteriors)

class GeometricMedian(Estimator):
	
	def __init__(self,maxIterations=100,tolerance=0.001):
		
		self._maxIterations = maxIterations
		self._tolerance = tolerance
	
	def estimate(self,DPF):
		
		# a 2D array is initialized to store the samples from the different PEs
		samples = np.empty((state.nElements,len(DPF._PEs)))
		
		## for every PE...
		#for iPE,PE in enumerate(DPF._PEs):
			
			## ...the index of the sample with the largest weight is obtained...
			#iMax = PE.logWeights.argmax()
			
			## ...and its corresponding sample extracted and stored in the array initialized above
			#samples[:,iPE:iPE+1] = PE.getSamplesAt([iMax])
		
		# the first (0) sample of each PE is collected
		samples = np.hstack([PE.getSamplesAt([0]) for PE in DPF._PEs])

		return geometric_median(samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]

class StochasticGeometricMedian(Mposterior):
	
	def __init__(self,nParticles,maxIterations=100,tolerance=0.001):
		
		self._nParticles = nParticles
		self._maxIterations = maxIterations
		self._tolerance = tolerance
	
	def estimate(self,DPF):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		samples = np.hstack([PE.getSamplesAt(DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),self._nParticles)) for PE in DPF._PEs])
		
		return geometric_median(samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]

class SinglePE(Estimator):
	
	def __init__(self,iPE):
		
		self._iPE = iPE
		
	def estimate(self,DPF):
		
		return DPF._PEs[self._iPE].computeMean()