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


class Estimator:

	def __init__(self, DPF, i_PE=0):

		self._DPF = DPF
		self.i_PE = i_PE

	# by default, it is assumed no communication is required
	def estimate(self):

		return 0


class Delegating(Estimator):

	def estimate(self):

		return self._DPF.computeMean()

class Mean(Estimator):

	def estimate(self):

		# the means from all the PEs are stacked (horizontally) in a single array
		jointMeans = np.hstack([PE.computeMean() for PE in self._DPF._PEs])

		return jointMeans.mean(axis=1)[:,np.newaxis]

	def messages(self, PEs_topology):

		# the distances (in hops) between each pair of PEs
		distances = PEs_topology.distances_between_PEs()

		return distances[self.i_PE,:].sum()*state.n_elements_position


class WeightedMean(Mean):

	def estimate(self):

		aggregatedWeights = self._DPF.getAggregatedWeights()

		# the aggregated weights are not necessarily normalized
		normalizedAggregatedWeights = aggregatedWeights/aggregatedWeights.sum()

		# notice that "computeMean" will return a numpy array the size of the state (rather than a scalar)
		return np.multiply(np.hstack([PE.computeMean() for PE in self._DPF._PEs]),normalizedAggregatedWeights).sum(axis=1)[:,np.newaxis]


class Mposterior(Estimator):

	def combine_posterior_distributions(self, posteriors):

		# the Mposterior algorithm is used to obtain a a new distribution
		joint_particles, joint_weights = self._DPF.Mposterior(posteriors)

		return np.multiply(joint_particles, joint_weights).sum(axis=1)[np.newaxis].T

	def estimate(self):

		# the (FULL) distributions computed by all the PEs are gathered in a list of tuples (samples and weights)
		posteriors = [(PE.getState().T,np.exp(PE.logWeights)) for PE in self._DPF._PEs]

		return self.combine_posterior_distributions(posteriors)

	def messages(self, PEs_topology):

		# the distances (in hops) between each pair of PEs
		distances = PEs_topology.distances_between_PEs()

		# TODO: this assumes all PEs have the same number of particles: that of the self.i_PE-th one
		return distances[self.i_PE,:].sum()*self._DPF._PEs[self.i_PE].n_particles*state.n_elements_position


class PartialMposterior(Mposterior):

	def __init__(self, DPF, nParticles, i_PE=0):

		super().__init__(DPF, i_PE)

		self.n_particles = nParticles

	def estimate(self):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		posteriors = [(PE.getSamplesAt(self._DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),self.n_particles)).T,
				 np.full(self.n_particles,1.0/self.n_particles)) for PE in self._DPF._PEs]

		return self.combine_posterior_distributions(posteriors)

	def messages(self, PEs_topology):

		# the distances (in hops) between each pair of PEs
		distances = PEs_topology.distances_between_PEs()

		return distances[self.i_PE,:].sum()*self.n_particles*state.n_elements_position


class GeometricMedian(Estimator):

	def __init__(self, DPF, i_PE=0, maxIterations=100, tolerance=0.001):

		super().__init__(DPF, i_PE)

		self._maxIterations = maxIterations
		self._tolerance = tolerance

	def estimate(self):

		# the first (0) sample of each PE is collected
		samples = np.hstack([PE.getSamplesAt([0]) for PE in self._DPF._PEs])

		return geometric_median(samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]

	def messages(self, PEs_topology):

		# the distances (in hops) between each pair of PEs
		distances = PEs_topology.distances_between_PEs()

		return distances[self.i_PE,:].sum()*state.n_elements_position


class StochasticGeometricMedian(GeometricMedian):

	def __init__(self, DPF, nParticles, i_PE=0, maxIterations=100, tolerance=0.001):

		super().__init__(DPF, i_PE, maxIterations, tolerance)

		self.n_particles = nParticles

	def estimate(self):

		# a number of samples is drawn from the distribution of each PE (all equally weighted)
		# to build a list of tuples (samples and weights)
		samples = np.hstack(
			[PE.getSamplesAt(self._DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),
			self.n_particles)) for PE in self._DPF._PEs])

		return geometric_median(samples,max_iterations=self._maxIterations,tolerance=self._tolerance)[:,np.newaxis]

	def messages(self, PEs_topology):

		return super().messages(PEs_topology)*self.n_particles


class SinglePEMean(Estimator):

	def estimate(self):

		return self._DPF._PEs[self.i_PE].computeMean()


class SinglePEGeometricMedian(Estimator):

	def __init__(self, DPF, iPE, maxIterations=100, tolerance=0.001):

		super().__init__(DPF, iPE)

		self._maxIterations = maxIterations
		self._tolerance = tolerance

	def estimate(self):

		return geometric_median(self._DPF._PEs[self.i_PE].samples, max_iterations=self._maxIterations, tolerance=self._tolerance)[:,np.newaxis]


class SinglePEGeometricMedianWithinRadius(SinglePEGeometricMedian):

	def __init__(self, DPF, iPE, PEs_topology, radius, maxIterations=100, tolerance=0.001):

		super().__init__(DPF, iPE, maxIterations, tolerance)

		self._distances = PEs_topology.distances_between_PEs()

		# the indexes of the PEs that are at most "radius" hops from the selected PE
		self._i_relevant_PEs, = np.where((self._distances[self.i_PE] > 0) & (self._distances[self.i_PE]<=radius))

		# the selected PE is also included
		self._i_relevant_PEs = np.append(self._i_relevant_PEs,self.i_PE)

	def estimate(self):

		# one sample from each of the above PEs
		samples = np.vstack([self._DPF._PEs[iPE].getSamplesAt(0) for iPE in self._i_relevant_PEs]).T

		return geometric_median(samples, max_iterations=self._maxIterations, tolerance=self._tolerance)[:,np.newaxis]
	
	def messages(self, PEs_topology):
		"""
		:return: the number of messages exchanged between PEs due to a call to "estimate"
		"""

		# the number of hops for each neighbour times the number of floats sent per message
		return self._distances[self.i_PE,self._i_relevant_PEs].sum()*state.n_elements_position