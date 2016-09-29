import abc


class ParticleFilter(metaclass=abc.ABCMeta):

	def __init__(self, n_particles, resampling_algorithm, resampling_criterion, name=None):

		self._n_particles = n_particles

		self._resampling_algorithm = resampling_algorithm
		self._resampling_criterion = resampling_criterion

		self._name = name

	@abc.abstractmethod
	def initialize(self):

		pass

	@abc.abstractmethod
	def step(self, observations):

		pass

	@property
	def n_particles(self):

		return self._n_particles

	@property
	def name(self):

		return self._name