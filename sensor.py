import abc

import numpy as np
import scipy.stats


class Sensor(metaclass=abc.ABCMeta):

	def __init__(self, position, pseudo_random_numbers_generator):

		# position is saved for later use
		self.position = position

		# pseudo random numbers generator
		self._pseudo_random_numbers_generator = pseudo_random_numbers_generator

	@abc.abstractmethod
	def detect(self, target_pos):

		pass

	@abc.abstractmethod
	def likelihood(self, observation, positions):

		""" It computes the likelihoods of several positions.
		
		Parameters
		----------
		observation: float
			the observation whose probability is computed
		positions: numpy array
			positions of several particles
		"""
		pass


class BinarySensor(Sensor):

	def __init__(
			self, position, radius, probability_of_detection_within_the_radius=0.9, probability_of_false_alarm=0.01,
			pseudo_random_numbers_generator=np.random.RandomState()):

		super().__init__(position, pseudo_random_numbers_generator)

		# the distance within reach of the sensor
		self._threshold = radius

		# the probability of (correct) detection
		self._prob_detection = probability_of_detection_within_the_radius

		# the probability of false alarm
		self._prob_false_alarm = probability_of_false_alarm

		# for the sake of convenience when computing the likelihood: we keep an array with the probability mass function
		# of the observations conditional on the target being close enough (it depends on the threshold)...
		# self._pmf_observations_when_close[x] = p(observation=x | |<target position> - <sensor position>| < threshold)
		self._pmf_observations_when_close = np.array(
			[1-probability_of_detection_within_the_radius, probability_of_detection_within_the_radius])

		# ...and that of the observations conditional on the target being far
		self._pmf_observations_when_far = np.array([1 - probability_of_false_alarm, probability_of_false_alarm])

	def detect(self, target_pos):

		distance = np.linalg.norm((self.position - target_pos))

		if distance < self._threshold:
			return self._pseudo_random_numbers_generator.rand() < self._prob_detection
		else:
			return self._pseudo_random_numbers_generator.rand() < self._prob_false_alarm

	def likelihood(self, observation, positions):

		# the distances to ALL the positions are computed
		distances = np.linalg.norm(np.subtract(positions, self.position), axis=0)

		# an empty array with the same dimensions as distances is created
		likelihoods = np.empty_like(distances)

		# the likelihood for a given observation is computed using probability mass function if the target
		# is within the reach of the sensor...
		likelihoods[distances < self._threshold] = self._pmf_observations_when_close[observation]

		# ...and a different one if it's outside it
		likelihoods[distances >= self._threshold] = self._pmf_observations_when_far[observation]

		return likelihoods


class RSSsensor(Sensor):

	def __init__(
			self, position, transmitter_power=1, path_loss_exponent=2, noise_variance=1, minimum_amount_of_power=1e-5,
			pseudo_random_numbers_generator=np.random.RandomState()):

		super().__init__(position, pseudo_random_numbers_generator)

		# the power of the transmitter
		self._tx_power = transmitter_power

		# the path loss exponent (depending on the medium)
		self._path_loss_exponent = path_loss_exponent

		# the variance of the additive noise in the model (it is meant to be accessed from outside)
		self.noise_var = noise_variance

		# ...and, for the sake of efficiency, the standard deviation
		self._noise_std = np.sqrt(noise_variance)

		# minimum amount of power the sensor is able to measure
		self._minimum_power = minimum_amount_of_power

	def likelihood_mean(self, distances):

		return 10*np.log10(self._tx_power / distances ** self._path_loss_exponent + self._minimum_power)

	def detect(self, target_pos):

		distance = np.linalg.norm((self.position - target_pos))

		return self.likelihood_mean(distance) + self.measurement_noise()

	def measurement_noise(self):

		return self._pseudo_random_numbers_generator.randn()*self._noise_std

	def likelihood(self, observation, positions):

		# the distances to ALL the positions are computed
		distances = np.linalg.norm(np.subtract(positions, self.position), axis=0)

		return scipy.stats.norm.pdf(observation, self.likelihood_mean(distances), self._noise_std)

	def set_parameters(self,  tx_power, minimum_amount_of_power, path_loss_exponent):

		self._tx_power = tx_power
		self._minimum_power = minimum_amount_of_power
		self._path_loss_exponent = path_loss_exponent


class RSSsensorsArray:

	def __init__(self, sensors):

		self._tx_power = np.array([s._tx_power for s in sensors])
		self._path_loss_exponent = np.array([s._path_loss_exponent for s in sensors])
		self._noise_std = np.array([s._noise_std for s in sensors])
		self._minimum_power = np.array([s._minimum_power for s in sensors])
		self._positions = np.hstack([s.position for s in sensors])

	def likelihood(self, observations, positions):

		# each row a sensor, every column a different particle (position received)
		distances = np.linalg.norm(positions[:, np.newaxis, :] - self._positions[:, :, np.newaxis], axis=0)

		# each row a sensor, every column a different particle (position received)
		likelihood_mean = 10*np.log10(
			self._tx_power[:, np.newaxis] / distances ** self._path_loss_exponent[:, np.newaxis] +
			self._minimum_power[:, np.newaxis]
		)

		return scipy.stats.norm.pdf(observations[:, np.newaxis], likelihood_mean, self._noise_std[:, np.newaxis])


class BinarySensorsArray:

	def __init__(self, sensors):

		self._sensors_range = np.arange(len(sensors))

		self._positions = np.hstack([s.position for s in sensors])
		self._thresholds = np.array([s._threshold for s in sensors])

		self._pmf_obs_when_close = np.vstack([s._pmf_observations_when_close for s in sensors])
		self._pmf_observations_when_far = np.vstack([s._pmf_observations_when_far for s in sensors])

		self._full = np.stack((self._pmf_obs_when_close, self._pmf_observations_when_far))

	def likelihood(self, observations, positions):

		# each row a sensor, every column a different particle (position received)
		distances = np.linalg.norm(positions[:, np.newaxis, :] - self._positions[:, :, np.newaxis], axis=0)

		return self._full[
			(distances >= np.broadcast_to(self._thresholds[:, np.newaxis], distances.shape)).astype(int),
			np.broadcast_to(self._sensors_range[:, np.newaxis], distances.shape),
			np.broadcast_to(observations[:, np.newaxis], distances.shape).astype(int)
		]