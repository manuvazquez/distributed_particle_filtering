import abc
import scipy


class Base(metaclass=abc.ABCMeta):

	def __init__(self, prng, parameters):

		self._prng = prng
		self._parameters = parameters

	@abc.abstractmethod
	def draw_sample(self, n=1, parameters={}):

		pass

	@abc.abstractmethod
	def pdf(self, samples, parameters={}):

		pass

	@property
	@abc.abstractmethod
	def sample_len(self):

		pass


class Gaussian(Base):

	# def __init__(self, prng, parameters):
	#
	# 	self._mean = parameters['mean']
	# 	self._cov = parameters['cov']
	#
	# 	super().__init__(prng)

	def draw_sample(self, n=1, parameters={}):

		# if parameters:
		#
		# 	parameters['size'] = n
		#
		# 	if 'cov' not in parameters:
		#
		# 		parameters['cov'] = self._cov
		#
		# else:
		#
		# 	parameters = {'mean': self._mean, 'cov': self._cov, 'size': n}
		#
		# self._prng.multivariate_normal(**parameters)

		return self._prng.multivariate_normal(**{**self._parameters, **parameters, **{'size': n}})

	def pdf(self, samples, parameters={}):

		# if not parameters:
		#
		# 	parameters = {'mean': self._mean, 'cov': self._cov}
		#
		# return scipy.stats.multivariate_normal.pdf(x=samples, **parameters)

		return scipy.stats.multivariate_normal.pdf(samples, **{**self._parameters, **parameters})

	@property
	def sample_len(self):

		return self._parameters['cov'].shape[0]
