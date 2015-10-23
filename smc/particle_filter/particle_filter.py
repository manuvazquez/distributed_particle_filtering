import abc


class ParticleFilter(metaclass=abc.ABCMeta):

	def __init__(self, n_particles, resampling_algorithm, resampling_criterion):

		self._nParticles = n_particles

		self._resamplingAlgorithm = resampling_algorithm
		self._resamplingCriterion = resampling_criterion

	@abc.abstractmethod
	def initialize(self):

		pass

	@abc.abstractmethod
	def step(self, observations):

		pass

	@abc.abstractmethod
	def get_state(self):

		pass

	def messages(self, processing_elements_topology, each_processing_element_connected_sensors):

		# to indicate it has not been computed
		return -1