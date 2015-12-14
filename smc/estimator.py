import abc

import numpy as np
import numpy.linalg

import state
import mposterior


def geometric_median(points, max_iterations=100, tolerance=0.001):

	# initial estimate
	estimate = np.median(points, axis=1)

	for i in range(max_iterations):

		# the norms for the vectors joining the previous estimate with every point
		norms = numpy.linalg.norm(np.subtract(points, estimate[:, np.newaxis]), axis=0)

		# is any of the norms is zero?
		is_zero = np.isclose(norms, 0.0)

		# if one of the norms is zero (there should be one at most)
		if np.any(is_zero):

			# we find out its position...
			i_zero_norm, = np.where(is_zero)

			# ...and the estimate of the median is the corresponding point
			estimate = points[:, i_zero_norm[0]]

			return estimate

		# this is used a couple of times below
		invnorms = 1.0 / norms

		# a new estimate according to the Weiszfeld algorithm
		new_estimate = np.multiply(points, invnorms[np.newaxis, :]).sum(axis=1)/invnorms.sum()

		# if the new estimate is close enough to the old one...
		if numpy.linalg.norm(new_estimate-estimate) < tolerance:

			# ...it gets a pass
			return new_estimate

		# ...otherwise, the new estimate becomes will be used in the next iteration
		estimate = new_estimate

	return estimate


class Estimator:

	def __init__(self, distributed_particle_filter, i_processing_element=0):

		self.DPF = distributed_particle_filter
		self.i_processing_element = i_processing_element

	# by default, it is assumed no communication is required
	def messages(self, processing_elements_topology):

		return 0

	def estimate(self):

		return


class Delegating(Estimator):

	def estimate(self):

		return self.DPF.compute_mean()


class Mean(Estimator):

	def estimate(self):

		# the means from all the PEs are stacked (horizontally) in a single array
		joint_means = np.hstack([PE.compute_mean() for PE in self.DPF._PEs])

		return joint_means.mean(axis=1)[:, np.newaxis]

	def messages(self, processing_elements_topology):

		# the distances (in hops) between each pair of PEs
		distances = processing_elements_topology.distances_between_processing_elements

		return distances[self.i_processing_element, :].sum()*state.n_elements_position


class WeightedMean(Mean):

	def estimate(self):

		aggregated_weights = self.DPF.aggregated_weights

		# the aggregated weights are not necessarily normalized
		normalized_aggregated_weights = aggregated_weights/aggregated_weights.sum()

		# notice that "compute_mean" will return a numpy array the size of the state (rather than a scalar)
		return np.multiply(
			np.hstack([PE.compute_mean() for PE in self.DPF._PEs]),
			normalized_aggregated_weights).sum(axis=1)[:, np.newaxis]

	def messages(self, processing_elements_topology):

		distances = processing_elements_topology.distances_between_processing_elements

		# the same as in "Mean" but we also have to transmit the aggregated weight
		return super().messages(processing_elements_topology) + distances[self.i_processing_element, :].sum()


class Mposterior(Estimator):

	def __init__(self, distributed_particle_filter, weiszfeld_parameters, i_processing_element=0):

		super().__init__(distributed_particle_filter, i_processing_element)

		self.weiszfeld_parameters = weiszfeld_parameters

	def combine_posterior_distributions(self, posteriors):

		joint_particles, joint_weights = mposterior.find_weiszfeld_median(posteriors, **self.weiszfeld_parameters)

		return np.multiply(joint_particles, joint_weights).sum(axis=1)[np.newaxis].T

	def estimate(self):

		# the (FULL) distributions computed by all the PEs are gathered in a list of tuples (samples and weights)
		posteriors = [(PE.get_state().T, np.exp(PE.log_weights)) for PE in self.DPF._PEs]

		return self.combine_posterior_distributions(posteriors)

	def messages(self, processing_elements_topology):

		# the distances (in hops) between each pair of PEs
		distances = processing_elements_topology.distances_between_processing_elements

		# TODO: this assumes all PEs have the same number of particles: that of the self.i_PE-th one
		return distances[self.i_processing_element, :].sum()*self.DPF._PEs[self.i_processing_element].n_particles*state.n_elements_position


class GeometricMedian(Estimator):

	def __init__(self, distributed_particle_filter, i_processing_element=0, max_iterations=100, tolerance=0.001):

		super().__init__(distributed_particle_filter, i_processing_element)

		self._maxIterations = max_iterations
		self._tolerance = tolerance

	def estimate(self):

		# the first (0) sample of each PE is collected
		samples = np.hstack([PE.get_samples_at([0]) for PE in self.DPF._PEs])

		return geometric_median(samples, max_iterations=self._maxIterations, tolerance=self._tolerance)[:, np.newaxis]

	def messages(self, processing_elements_topology):

		# the distances (in hops) between each pair of PEs
		distances = processing_elements_topology.distances_between_processing_elements

		return distances[self.i_processing_element, :].sum()*state.n_elements_position


class StochasticGeometricMedian(GeometricMedian):

	def __init__(
			self, distributed_particle_filter, n_particles, i_processing_element=0, max_iterations=100, tolerance=0.001):

		super().__init__(distributed_particle_filter, i_processing_element, max_iterations, tolerance)

		self.n_particles = n_particles

	def estimate(self):

		# a number of samples is drawn from the distribution of each PE (all equally weighted)
		# to build a list of tuples (samples and weights)
		samples = np.hstack(
			[PE.get_samples_at(self.DPF._resamplingAlgorithm.getIndexes(np.exp(PE.log_weights),
			self.n_particles)) for PE in self.DPF._PEs])

		return geometric_median(samples, max_iterations=self._maxIterations, tolerance=self._tolerance)[:, np.newaxis]

	def messages(self, processing_elements_topology):

		return super().messages(processing_elements_topology)*self.n_particles


class SinglePEMean(Estimator):

	def estimate(self):

		return self.DPF._PEs[self.i_processing_element].compute_mean()


class SinglePEGeometricMedian(Estimator):

	def __init__(self, distributed_particle_filter, iPE, max_iterations=100, tolerance=0.001):

		super().__init__(distributed_particle_filter, iPE)

		self._maxIterations = max_iterations
		self._tolerance = tolerance

	def estimate(self):

		return geometric_median(
			self.DPF._PEs[self.i_processing_element].samples, max_iterations=self._maxIterations, tolerance=self._tolerance
		)[:, np.newaxis]


class SinglePEGeometricMedianWithinRadius(SinglePEGeometricMedian):

	def __init__(
			self, distributed_particle_filter, iPE, PEs_topology, radius, n_particles=1,
			radius_lower_bound=0, max_iterations=100, tolerance=0.001):

		super().__init__(distributed_particle_filter, iPE, max_iterations, tolerance)

		self.n_particles = n_particles

		self._distances = PEs_topology.distances_between_processing_elements

		# the indexes of the PEs that are at most "radius" hops from the selected PE
		self._i_relevant_PEs = PEs_topology.i_neighbours_within_hops(radius, radius_lower_bound)[self.i_processing_element]

		# the selected PE is also included
		self._i_relevant_PEs.append(self.i_processing_element)

	def estimate(self):

		# the first "self.n_particles" samples from each of the above PEs
		samples = np.vstack([self.DPF._PEs[iPE].get_samples_at(range(self.n_particles)).T for iPE in self._i_relevant_PEs]).T

		return geometric_median(samples, max_iterations=self._maxIterations, tolerance=self._tolerance)[:, np.newaxis]
	
	def messages(self, processing_elements_topology):

		# the number of hops for each neighbour times the number of floats sent per message
		return (self._distances[self.i_processing_element, self._i_relevant_PEs].sum()*state.n_elements_position)*self.n_particles
