import abc

import numpy as np
import numpy.linalg


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

	def __init__(self,DPF):

		self._DPF = DPF

	@abc.abstractmethod
	def estimate(self):

		return

class Delegating(Estimator):

	def estimate(self):

		return self._DPF.computeMean()

class Mean(Estimator):

	def estimate(self):

		# the means from all the PEs are stacked (horizontally) in a single array
		jointMeans = np.hstack([PE.computeMean() for PE in self._DPF._PEs])

		return jointMeans.mean(axis=1)[:,np.newaxis]


class WeightedMean(Estimator):

	def estimate(self):

		aggregatedWeights = self._DPF.getAggregatedWeights()

		# the aggregated weights are not necessarily normalized
		normalizedAggregatedWeights = aggregatedWeights/aggregatedWeights.sum()

		# notice that "computeMean" will return a numpy array the size of the state (rather than a scalar)
		return np.multiply(np.hstack([PE.computeMean() for PE in self._DPF._PEs]),normalizedAggregatedWeights).sum(axis=1)[:,np.newaxis]

class Mposterior(Estimator):

	def combinePosteriorDistributions(self,posteriors):

		# the Mposterior algorithm is used to obtain a a new distribution
		jointParticles,jointWeights = self._DPF.Mposterior(posteriors)

		return np.multiply(jointParticles,jointWeights).sum(axis=1)[np.newaxis].T

	def estimate(self):

		# the (FULL) distributions computed by every PE are gathered in a list of tuples (samples and weights)
		posteriors = [(PE.getState().T,np.exp(PE.logWeights)) for PE in self._DPF._PEs]

		return self.combinePosteriorDistributions(posteriors)


class PartialMposterior(Mposterior):

	def __init__(self,DPF,nParticles):

		super().__init__(DPF)

		self._nParticles = nParticles

	def estimate(self):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		posteriors = [(PE.getSamplesAt(self._DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),self._nParticles)).T,
				 np.full(self._nParticles,1.0/self._nParticles)) for PE in self._DPF._PEs]

		return self.combinePosteriorDistributions(posteriors)


class GeometricMedian(Estimator):

	def __init__(self,DPF,maxIterations=100,tolerance=0.001):

		super().__init__(DPF)

		self._maxIterations = maxIterations
		self._tolerance = tolerance

	def estimate(self):

		# the first (0) sample of each PE is collected
		samples = np.hstack([PE.getSamplesAt([0]) for PE in self._DPF._PEs])

		return geometric_median(samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]


class StochasticGeometricMedian(Mposterior):

	def __init__(self,DPF,nParticles,maxIterations=100,tolerance=0.001):

		super().__init__(DPF)

		self._nParticles = nParticles
		self._maxIterations = maxIterations
		self._tolerance = tolerance

	def estimate(self):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		samples = np.hstack([PE.getSamplesAt(self._DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),self._nParticles)) for PE in self._DPF._PEs])

		return geometric_median(samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]


class SinglePEmean(Estimator):

	def __init__(self,DPF,iPE):

		super().__init__(DPF)

		self._iPE = iPE

	def estimate(self):

		return self._DPF._PEs[self._iPE].computeMean()


class SinglePEgeometricMedian(SinglePEmean):

	def __init__(self, DPF, iPE, maxIterations=100, tolerance=0.001):

		super().__init__(DPF,iPE)

		self._maxIterations = maxIterations
		self._tolerance = tolerance

	def estimate(self):

		return geometric_median(self._DPF._PEs[self._iPE].samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]


class SinglePEgeometricMedianWithinRadius(SinglePEgeometricMedian):

	def __init__(self, DPF, iPE, PEs_topology, radius, maxIterations=100, tolerance=0.001):

		super().__init__(DPF, iPE, maxIterations, tolerance)

		self._distances = PEs_topology.distances_between_PEs()
		self._radius = radius

	def estimate(self):

		# the indexes of the PEs that are at most "radius" hops from the selected PE
		i_relevant_PEs, = np.where((self._distances[self._iPE] > 0) & (self._distances[self._iPE]<=self._radius))

		# the selected PE is also included
		i_relevant_PEs = np.append(i_relevant_PEs,self._iPE)

		# one sample from each of the above PEs
		samples = np.vstack([self._DPF._PEs[iPE].getSamplesAt(0) for iPE in i_relevant_PEs]).T

		return geometric_median(samples)[:,np.newaxis]