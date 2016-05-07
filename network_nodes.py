import math

import numpy as np
import scipy.cluster.vq


class Network:
	
	def __init__(self, bottom_left_corner, top_right_corner, n_PEs, n_sensors):
		
		self._bottom_left_corner = bottom_left_corner
		self._top_right_corner = top_right_corner
		self._n_PEs = n_PEs
		self._n_sensors = n_sensors

		self._PEs_positions = None
		self._sensors_positions = None

	def order_positions(self, positions):
		
		# the initial position is the upper-left corner
		previous_pos = np.array([self._bottom_left_corner[0], self._top_right_corner[1]])
		
		# we need to modify the "positions" array during the algorithm
		positions_copy = positions.copy()
		
		# a numpy array to store the result (just like the sensors positions, every column contains the two coordinates
		# for a position)
		ordered = np.empty_like(positions.T)
		
		for i in range(len(positions)):
			
			# the distance from the previuos position to ALL the PEs
			distances = np.linalg.norm(previous_pos - positions_copy, axis=1)
			
			# the index of the minimum distance...
			i_min = np.argmin(distances)
			
			# ...is used to pick the next position in the "ordered" array
			ordered[:, i] = positions[i_min, :]
			
			# we make sure this PE is not going to be selected again
			positions_copy[i_min, 0] = np.Inf
			
			# the previous position is the selected new one
			previous_pos = ordered[:, i]
			
		return ordered
	
	def random_positions(self, n, n_samples):
		
		# seed for the default numpy random generator (used by scipy)
		np.random.seed(123)
		
		# ...is generated from a uniform distribution within the limits of the room
		points = np.vstack((
			np.random.uniform(self._bottom_left_corner[0], self._top_right_corner[0], (1, n_samples)),
			np.random.uniform(self._bottom_left_corner[1], self._top_right_corner[1], (1, n_samples))))
		
		# "nPEs" centroids for the above coordinates are computed using K-Means; initial random centroids are passed to
		# the function so it does not generate them with its own random generator
		positions, _ = scipy.cluster.vq.kmeans(points.T, n)

		return positions

	def equispaced_positions(self, n):
		
		# a vector representing the diagonal of the rectangle...
		diagonal = self._top_right_corner - self._bottom_left_corner
		
		# ...from which we compute the area
		area = diagonal.prod()
		
		# if the positions are equispaced, each one should "cover" an area equal to
		area_per_sensor = area/n
		
		# if the area "covered" by each sensor is a square, then its side is
		square_side = math.sqrt(area_per_sensor)
		
		# number of "full" squares that fit in each dimension
		n_squares_x_dim, n_squares_y_dim = np.floor(diagonal[0]/square_side), np.floor(diagonal[1]/square_side)
		
		# if by adding one position in each dimension...
		n_overfitting_sensors = (n_squares_x_dim+1)*(n_squares_y_dim+1)
		
		# ...we get closer to the number of requested sensors...
		if (n-(n_squares_x_dim*n_squares_y_dim)) > (n_overfitting_sensors-n):
			
			# ...we repeat the computations with the "overfitting" number of sensors
			area_per_sensor = area/n_overfitting_sensors
			
			square_side = math.sqrt(area_per_sensor)
			n_squares_x_dim, n_squares_y_dim = np.floor(diagonal[0]/square_side), np.floor(diagonal[1]/square_side)
		
		# in each dimension there is a certain length that is not covered (using % "weird" things happen sometimes...)
		remaining_x_dim = diagonal[0] - n_squares_x_dim*square_side
		remaining_y_dim = diagonal[1] - n_squares_y_dim*square_side
		
		res = np.transpose(
				np.array(
					[
						[
							self._bottom_left_corner[0] + (remaining_x_dim + square_side) / 2 + i * square_side,
							self._bottom_left_corner[1] + (remaining_y_dim + square_side) / 2 + j * square_side
						] for i in range(int(n_squares_x_dim)) for j in range(int(n_squares_y_dim))
					]))

		return res

	@property
	def PEs_positions(self):
		
		return self._PEs_positions

	@property
	def sensors_positions(self):
		
		return self._sensors_positions


class FixedNumberOfSensorsPerPE(Network):
	
	def __init__(self, bottom_left_corner, top_right_corner, n_PEs, n_sensors, radius=2, phase=0, n_samples=10000):
		
		super().__init__(bottom_left_corner, top_right_corner, n_PEs, n_sensors)
		
		# there should be an integer number of sensors per PE...
		assert(n_sensors % n_PEs == 0)

		# ...which is
		n_sensors_per_PE = n_sensors // n_PEs
		
		self._PEs_positions = self.order_positions(self.random_positions(n_PEs, n_samples))
		
		# the sensors will be positioned at these angles around each PE
		angles = phase + np.arange(0, 2*np.pi, 2*np.pi/n_sensors_per_PE)
		
		self._sensors_positions = np.empty((2, n_sensors))
		i_sensor = 0
		
		for PE_position in self._PEs_positions.T:
			
			for angle in angles:

				self._sensors_positions[:, i_sensor] = PE_position + np.array([radius * np.cos(angle), radius * np.sin(angle)])
				
				i_sensor += 1
		
		assert(np.all(
				(self._bottom_left_corner[0] < self._sensors_positions[0, :]) &
				(self._sensors_positions[0, :] < self._top_right_corner[0])))
		assert(np.all(
				(self._bottom_left_corner[1] < self._sensors_positions[1, :]) &
				(self._sensors_positions[1, :] < self._top_right_corner[1])))


class PositionlessPEsEquispacedSensors(Network):
	
	def __init__(self, bottom_left_corner, top_right_corner, n_PEs, n_sensors):
		
		super().__init__(bottom_left_corner, top_right_corner, n_PEs, n_sensors)
		
		self._sensors_positions = self.equispaced_positions(self._n_sensors)


class RandomlyStrewnSensorsAndPEs(Network):
	
	def __init__(self, bottom_left_corner, top_right_corner, n_PEs, n_sensors, nSamples):
		
		super().__init__(bottom_left_corner, top_right_corner, n_PEs, n_sensors)
		
		self._sensors_positions = self.random_positions(n_sensors, nSamples).T
		self._PEs_positions = self.order_positions(self.random_positions(n_PEs, nSamples))


class IntegratedPEsAndSensors(Network):

	def __init__(self, bottom_left_corner, top_right_corner, n_PEs, n_sensors, nSamples):

		assert n_PEs == n_sensors

		super().__init__(bottom_left_corner, top_right_corner, n_PEs, n_PEs)

		self._PEs_positions = self._sensors_positions = self.random_positions(n_sensors, nSamples).T