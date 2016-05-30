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

i_position = range(2)
i_velocity = range(2, 4)


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

	def __init__(self, room, velocity_mean=0, velocity_variance=0.25, PRNG=np.random.RandomState()):

		self._room = room
		self._PRNG = PRNG

		self._velocity_mean = velocity_mean
		self._velocity_variance = velocity_variance

	def sample(self, n_samples=1, PRNG=None):

		return None


class UniformBoundedPositionGaussianVelocityPrior(Prior):
		
	def sample(self, n_samples=1, PRNG=None):

		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:

			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		# position
		position = np.empty((2, n_samples))
		position[0, :] = PRNG.uniform(*self._room.x_range, n_samples)
		position[1, :] = PRNG.uniform(*self._room.y_range, n_samples)
		
		# velocity
		velocity = PRNG.normal(self._velocity_mean, math.sqrt(self._velocity_variance / 2), (2, n_samples))
		
		return np.vstack((position, velocity))


class GaussianPositionAndVelocity(Prior):

	def __init__(
			self, room, position_mean, position_variance, velocity_mean=0, velocity_variance=0.25,
			PRNG=np.random.RandomState()):

		super().__init__(room, velocity_mean, velocity_variance, PRNG)

		self._position_mean = position_mean
		self._position_variance = position_variance

	def sample(self, n_samples=1, PRNG=None):

		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:

			# ...the corresponding class attribute is used
			PRNG = self._PRNG

		position = PRNG.multivariate_normal(self._position_mean, self._position_variance*np.identity(2), n_samples).T
		velocity = PRNG.normal(self._velocity_mean, np.sqrt(self._velocity_variance / 2), (2, n_samples))

		return np.vstack((position, velocity))


class TransitionKernel:
	
	def __init__(self, step_duration):
		
		# a step lasts this time units (distance travelled each step equals velocity times this value)
		self._step_duration = step_duration
	
	def next_state(self, state):
		
		pass


class UnboundedTransitionKernel(TransitionKernel):
	
	def __init__(self, room, velocity_variance=0.5, noise_variance=0.1, step_duration=1, PRNG=np.random.RandomState()):
		
		# the parent's constructor is called
		super().__init__(step_duration)

		self._room = room

		# variance of the noise affecting the velocity
		self._velocity_variance = velocity_variance
		
		# variance of the noise affecting the position
		self._noise_variance = noise_variance
		
		# the pseudo random numbers generator
		self._PRNG = PRNG

	def next_state(self, state, PRNG=None):

		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG

		n = state.shape[1]

		velocity = to_velocity(state) + PRNG.normal(0, math.sqrt(self._velocity_variance / 2), (2, n))

		# step to be taken is obtained from the velocity and a noise component
		step = velocity * self._step_duration + PRNG.normal(0, math.sqrt(self._noise_variance / 2), (2, n))

		return np.vstack((to_position(state) + step, velocity))


class BouncingWithinRectangleTransitionKernel(UnboundedTransitionKernel):
	
	def __init__(
			self, room, velocity_variance=0.5, noise_variance=0.1, step_duration=1, PRNG=np.random.RandomState(),
			max_bounces=5):
		
		# the parent's constructor is called
		super().__init__(room, velocity_variance, noise_variance, step_duration, PRNG)
		
		# canonical vectors used in computations
		self._iVector = np.array([[1.0], [0.0]])
		self._jVector = np.array([[0.0], [1.0]])

		# maximum number of bounces before giving up on unbounded positions
		self._max_bounces = max_bounces

	def next_state(self, state, PRNG=None):

		# this may be updated in the while loop when bouncing off several walls
		previous_pos = to_position(state).copy()

		# the new (unbounded) state as computed by the parent class
		unbounded_state = super().next_state(state, PRNG)

		# the first two elements of the above state is the new (here tentative) position...
		tentative_new_pos = to_position(unbounded_state)

		# ...whereas the last ones are the new velocity
		velocity = to_velocity(unbounded_state)

		# those positions that are NOT bounded before update are ignored below
		i_bounded_before_update = self._room.belong(previous_pos)

		n_bounces = 0

		while True:

			# the positions that need update are those that were bounded BEFORE the update and are not AFTERWARDS
			i_invalid = np.where(np.logical_and(
				np.logical_not(self._room.belong(tentative_new_pos)), i_bounded_before_update)
			)[0]

			if (i_invalid.size == 0) or (n_bounces==self._max_bounces):

				return np.vstack((tentative_new_pos, velocity))

			# a vector joining the previous position and the corresponding corner (dimension 2 is associated with a corner)
			position_to_corner = self._room.tr_tl_bl_br_corners[:, np.newaxis, :] - previous_pos[:, i_invalid, np.newaxis]

			# the angle between the above vector and a horizontal line
			angles_with_corners = np.arccos(position_to_corner[0, ...] / np.linalg.norm(position_to_corner, axis=0))

			# we account for the fact that the angle between two vectors computed by means of the dot product is
			# always between 0 and pi (the shortest)
			angles_with_corners[:, 2:] = 2 * np.pi - angles_with_corners[:, 2:]

			step = tentative_new_pos[:, i_invalid] - previous_pos[:, i_invalid]

			# angle between the step to be taken and the horizontal line
			angle = np.arccos(step[0, :] / np.linalg.norm(step, axis=0))

			# in order to "fix" negative angles
			angle[step[1, :] <= 0] = 2 * np.pi - angle[step[1, :] <= 0]

			normal = np.empty((2, angle.shape[-1]))
			scale_factor = np.empty(angle.shape[-1])

			i_up = np.logical_and(angle > angles_with_corners[:, 0], angle < angles_with_corners[:, 1])
			normal[:, i_up] = -self._jVector
			scale_factor[i_up] = (self._room.tr_corner[1] - previous_pos[1, i_invalid][i_up])/step[1, i_up]
			
			i_left = np.logical_and(angle > angles_with_corners[:, 1], angle < angles_with_corners[:, 2])
			normal[:, i_left] = self._iVector
			scale_factor[i_left] = (self._room.bl_corner[0] - previous_pos[0, i_invalid][i_left])/step[0, i_left]
			
			i_down = np.logical_and(angle > angles_with_corners[:, 2], angle < angles_with_corners[:, 3])
			normal[:, i_down] = self._jVector
			scale_factor[i_down] = (self._room.bl_corner[1] - previous_pos[1, i_invalid][i_down])/step[1, i_down]
			
			i_right = np.logical_not(np.logical_or(np.logical_or(i_up, i_left), i_down))
			normal[:, i_right] = -self._iVector
			scale_factor[i_right] = (self._room.tr_corner[0] - previous_pos[0, i_invalid][i_right])/step[0, i_right]

			# the components of the "step" vector before...
			step_before_bounce = scale_factor * step

			# ...and after reaching the wall
			step_after_bounce = (1.0 - scale_factor) * step

			# in case we need to bounce again, we update the previous position...
			previous_pos[:, i_invalid] = previous_pos[:, i_invalid] + step_before_bounce

			# ...and the step, this one by computing the reflected ray using a formula involving the normal of the
			# reflecting surface...
			step = step_after_bounce - 2 * normal * (normal * step_after_bounce).sum(axis=0)

			tentative_new_pos[:, i_invalid] = previous_pos[:, i_invalid] + step

			# the direction of the velocity is updated according to the reflection that occurred in the trajectory
			# note that this only needs to be done in the last iteration of the while loop, but since the velocity
			# is not used within the "else" part, it's not a problem
			velocity[:, i_invalid] = step / np.linalg.norm(step, axis=0) * np.linalg.norm(velocity[:, i_invalid], axis=0)

			n_bounces += 1


class OnEdgeResetTransitionKernel(UnboundedTransitionKernel):
	
	def __init__(
			self, room, velocity_variance=0.5, noise_variance=0.1, step_duration=1, PRNG=np.random.RandomState(),
			reset_velocity_variance=0.01):
		
		# the parent's constructor is called
		super().__init__(room, velocity_variance, noise_variance, step_duration, PRNG)
		
		self._reset_velocity_variance = reset_velocity_variance
		
	def next_state(self, state, PRNG=None):
		
		# the new (unbounded) state as computed by the parent class
		unbounded_state = super().next_state(state, PRNG)
		
		# indexes of the samples whose position is already bounded
		invalid = ~self._room.belong(to_position(unbounded_state))

		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG

		# new positions that "fall" outside the bounded region are replaced with old positions...
		unbounded_state[np.ix_(i_position, invalid)] = state[np.ix_(i_position, invalid)]

		# ...and their corresponding velocities are reset
		unbounded_state[np.ix_(i_velocity, invalid)] = PRNG.normal(0, np.sqrt(self._reset_velocity_variance / 2), (2, invalid.sum()))
		
		return unbounded_state
