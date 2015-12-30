import math
import numpy as np

# a state vector has the structure:
# 	[ pos_x ]
# 	| pos_y ]
# 	| vel_x ]
# 	[ vel_y ]

# so that only this module needs to know about implementation details...
n_elements = 4
n_elements_position = 2


def to_position(state):
	"""It extracts the position elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the
	state vector.

	Parameters
	----------
	state : array_like
		The source array.

	Returns
	-------
	position : array_like
		The position embedded in the state.

	"""
	
	return state[0:2, :]


def to_velocity(state):
	"""It extracts the velocity elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the
	state vector.

	Parameters
	----------
	state : array_like
		The source array.

	Returns
	-------
	velocity : array_like
		The velocity embedded in the state.

	"""
	
	return state[2:4, :]


def build_state(position, velocity):
	"""It builds a state vector given a position and a velocity vectors.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the
	state vector.

	Parameters
	----------
	position : array_like
		source for the position.
	velocity : array_like
		source for the velocity.

	Returns
	-------
	state : array_like
		The state determined by the given position and velocity.

	"""
	
	return np.vstack((position, velocity))


class Prior:
	
	def sample(self):
		
		pass


class UniformBoundedPositionGaussianVelocityPrior(Prior):
	
	def __init__(
			self, bottom_left_corner, top_right_corner, velocity_mean=0, velocity_variance=0.25,
			PRNG=np.random.RandomState()):
		
		self._bottom_left_corner = bottom_left_corner
		self._top_right_corner = top_right_corner
		
		self._velocity_mean = velocity_mean
		self._velocity_variance = velocity_variance
		
		self._PRNG = PRNG
		
	def sample(self, n_samples=1, PRNG=None):
		
		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:

			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		# position
		position = np.empty((2, n_samples))
		position[0, :] = PRNG.uniform(self._bottom_left_corner[0], self._top_right_corner[0], n_samples)
		position[1, :] = PRNG.uniform(self._bottom_left_corner[1], self._top_right_corner[1], n_samples)
		
		# velocity
		velocity = PRNG.normal(self._velocity_mean, math.sqrt(self._velocity_variance / 2), (2, n_samples))
		
		return np.vstack((position, velocity))


class TransitionKernel:
	
	def __init__(self, step_duration):
		
		# a step lasts this time units (distance travelled each step equals velocity times this value)
		self._step_duration = step_duration
	
	def next_state(self, state):
		
		pass


class UnboundedTransitionKernel(TransitionKernel):
	
	def __init__(
			self, bottom_left_corner, top_right_corner, velocity_variance=0.5, noise_variance=0.1, step_duration=1,
			PRNG=np.random.RandomState()):
		
		# the parent's constructor is called
		super().__init__(step_duration)
		
		self._bottom_left_corner = bottom_left_corner
		self._top_right_corner = top_right_corner
		
		# variance of the noise affecting the velocity
		self._velocity_variance = velocity_variance
		
		# variance of the noise affecting the position
		self._noise_variance = noise_variance
		
		# the pseudo random numbers generator
		self._PRNG = PRNG
		
		# top right, top left, bottom left, and bottom right corners stored as column vectors
		self._corners = [
			top_right_corner[np.newaxis].T,
			np.array([[bottom_left_corner[0]], [top_right_corner[1]]]),
			bottom_left_corner[np.newaxis].T,
			np.array([[top_right_corner[0]], [bottom_left_corner[1]]])
		]
		
	def next_state(self, state, PRNG=None):
		
		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		# not actually needed...but for the sake of clarity...
		velocity = state[2:4]

		# the velocity changes BEFORE moving...
		velocity += PRNG.normal(0, math.sqrt(self._velocity_variance / 2), (2, 1))
		
		# step to be taken is obtained from the velocity and a noise component
		step = velocity*self._step_duration + PRNG.normal(0, math.sqrt(self._noise_variance / 2), (2, 1))
		
		return np.vstack((state[0:2] + step, velocity))


class BouncingWithinRectangleTransitionKernel(UnboundedTransitionKernel):
	
	def __init__(
			self, bottom_left_corner, top_right_corner, velocity_variance=0.5, noise_variance=0.1, step_duration=1,
			PRNG=np.random.RandomState()):
		
		# the parent's constructor is called
		super().__init__(bottom_left_corner, top_right_corner, velocity_variance, noise_variance, step_duration, PRNG)
		
		# canonical vectors used in computations
		self._iVector = np.array([[1.0], [0.0]])
		self._jVector = np.array([[0.0], [1.0]])
		
		# top right, top left, bottom left, and bottom right corners stored as column vectors
		self._corners = [
			top_right_corner[np.newaxis].T, np.array([[bottom_left_corner[0]], [top_right_corner[1]]]),
			bottom_left_corner[np.newaxis].T, np.array([[top_right_corner[0]], [bottom_left_corner[1]]])
		]
		
	def next_state(self, state, PRNG=None):
		
		# this may be updated in the while loop when bouncing off several walls
		previous_pos = state[0:2].copy()
		
		# the new (unbounded) state as computed by the parent class
		unbounded_state = super().next_state(state, PRNG)
		
		# the first two elements of the above state is the new (here tentative) position...
		tentative_new_pos = unbounded_state[:2]
		
		# ...whereas the last ones are the new velocity
		velocity = unbounded_state[2:]
		
		# the step "suggested" by the parent class is then given by the new tentative position and the starting 
		step = tentative_new_pos - previous_pos
		
		# to be used in the loop...
		angles_with_corners = np.empty(4)
		
		while True:
			
			if (tentative_new_pos > self._corners[2]).all() and (tentative_new_pos < self._corners[0]).all():
			
				# the new position is OK
				break
			
			else:
				
				for i, corner in enumerate(self._corners):
					
					# a vector joining the previous position and the corresponding corner
					position_to_corner = corner - previous_pos
					
					# the angle between the above vector and a horizontal line
					angles_with_corners[i] = math.acos(position_to_corner[0]/np.linalg.norm(position_to_corner))
				
				# we account for the fact that the angle between two vectors computed by means of the dot product is
				# always between 0 and pi (the shortest)
				angles_with_corners[2:] = 2*math.pi - angles_with_corners[2:]
				
				# angle between the step to be taken and the horizontal line
				angle = math.acos(step[0]/np.linalg.norm(step))
				
				if step[1] <= 0:
					# we account for the fact that the angle between two vectors computed by means of the dot product
					# (the shortest) is always between 0 and pi
					angle = 2*math.pi - angle
				
				# up
				if angles_with_corners[0] <= angle < angles_with_corners[1]:
					normal = -self._jVector
					scale_factor = (self._top_right_corner[1] - previous_pos[1]) / step[1]
				# left
				elif angles_with_corners[1] <= angle < angles_with_corners[2]:
					normal = self._iVector
					scale_factor = (self._bottom_left_corner[0] - previous_pos[0]) / step[0]
				# down
				elif angles_with_corners[2] <= angle < angles_with_corners[3]:
					normal = self._jVector
					scale_factor = (self._bottom_left_corner[1] - previous_pos[1]) / step[1]
				# right
				else:
					normal = -self._iVector
					scale_factor = (self._top_right_corner[0] - previous_pos[0]) / step[0]

				# the components of the "step" vector before...
				step_before_bounce = scale_factor*step
				
				# ...and after reaching the wall
				step_after_bounce = (1.0-scale_factor)*step
				
				# in case we need to bounce again, we update the previous position...
				previous_pos = previous_pos + step_before_bounce

				# ...and the step, this one by computing the reflected ray using a formula involving the normal of the
				# reflecting surface...
				step = step_after_bounce - 2*normal*np.dot(normal.T, step_after_bounce)

				tentative_new_pos = previous_pos + step
				
				# the direction of the velocity is updated according to the reflection that occurred in the trajectory
				# note that this only needs to be done in the last iteration of the while loop, but since the velocity
				# is not used within the "else" part, it's not a problem
				velocity = step/np.linalg.norm(step)*np.linalg.norm(velocity)

		return np.vstack((tentative_new_pos, velocity))


class OnEdgeResetTransitionKernel(UnboundedTransitionKernel):
	
	def __init__(
			self, bottom_left_corner, top_right_corner, velocity_variance=0.5, noise_variance=0.1, step_duration=1,
			PRNG=np.random.RandomState(), reset_velocity_variance=0.01):
		
		# the parent's constructor is called
		super().__init__(bottom_left_corner, top_right_corner, velocity_variance, noise_variance, step_duration, PRNG)
		
		self._reset_velocity_variance = reset_velocity_variance
		
	def next_state(self, state, PRNG=None):
		
		# the new (unbounded) state as computed by the parent class
		unbounded_state = super().next_state(state, PRNG)
		
		# the first two elements of the above state is the new (here tentative) position...
		tentative_new_pos = unbounded_state[:2]
		
		# if the tentative position is within the bounds of the rectangle...
		if (tentative_new_pos > self._corners[2]).all() and (tentative_new_pos < self._corners[0]).all():
			
			# it's already ok
			return unbounded_state
		
		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		velocity = PRNG.normal(0, math.sqrt(self._reset_velocity_variance / 2), (2, 1))
		
		return np.vstack((state[0:2], velocity))
