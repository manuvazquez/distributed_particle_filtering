import abc


class ParticleFilter(metaclass=abc.ABCMeta):

	def __init__(self, n_particles, resampling_algorithm, resampling_criterion):

		self._n_particles = n_particles

		self._resampling_algorithm = resampling_algorithm
		self._resampling_criterion = resampling_criterion

	@abc.abstractmethod
	def initialize(self):

		pass

	@abc.abstractmethod
	def step(self, observations):

		pass

	@abc.abstractmethod
	def get_state(self):

		pass

	@property
	def n_particles(self):

		return self._n_particles