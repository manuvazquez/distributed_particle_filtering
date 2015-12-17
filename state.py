import math
import numpy as np

# a state vector has the structure:
# 	[ pos_x ]
# 	| pos_y ]
# 	| vel_x ]
# 	[ vel_y ]

# so that only this module needs to know about implementation details...
nElements = 4
n_elements_position = 2


def position(state):
	"""It extracts the position elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the
	state vector.
	"""
	
	return state[0:2, :]


def velocity(state):
	"""It extracts the velocity elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the
	state vector.
	"""
	
	return state[2:4, :]


def build_state(position, velocity):
	"""It builds a state vector given a position and a velocity vectors.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the
	state vector.
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
		previousPos = state[0:2].copy()
		
		# the new (unbounded) state as computed by the parent class
		unboundedState = super().next_state(state, PRNG)
		
		# the first two elements of the above state is the new (here tentative) position...
		tentativeNewPos = unboundedState[:2]
		
		# ...whereas the last ones are the new velocity
		velocity = unboundedState[2:]
		
		# the step "suggested" by the parent class is then given by the new tentative position and the starting 
		step = tentativeNewPos - previousPos
		
		# to be used in the loop...
		anglesWithCorners = np.empty(4)
		
		while(True):
			
			if (tentativeNewPos > self._corners[2]).all() and (tentativeNewPos < self._corners[0]).all():
			
				# the new position is OK
				break
			
			else:
				
				for i, corner in enumerate(self._corners):
					
					# a vector joining the previous position and the corresponding corner
					positionToCorner = corner - previousPos
					
					# the angle between the above vector and a horizontal line
					anglesWithCorners[i] = math.acos(positionToCorner[0]/np.linalg.norm(positionToCorner))
				
				# we account for the fact that the angle between two vectors computed by means of the dot product is
				# always between 0 and pi (the shortest)
				anglesWithCorners[2:] = 2*math.pi - anglesWithCorners[2:]
				
				# angle between the step to be taken and the horizontal line
				angle = math.acos(step[0]/np.linalg.norm(step))
				
				if step[1] <= 0:
					# we account for the fact that the angle between two vectors computed by means of the dot product
					# (the shortest) is always between 0 and pi
					angle = 2*math.pi - angle
				
				# up
				if anglesWithCorners[0] <= angle < anglesWithCorners[1]:
					normal = -self._jVector
					scaleFactor = (self._top_right_corner[1] - previousPos[1]) / step[1]
				# left
				elif anglesWithCorners[1] <= angle < anglesWithCorners[2]:
					normal = self._iVector
					scaleFactor = (self._bottom_left_corner[0] - previousPos[0]) / step[0]
				# down
				elif anglesWithCorners[2] <= angle < anglesWithCorners[3]:
					normal = self._jVector
					scaleFactor = (self._bottom_left_corner[1] - previousPos[1]) / step[1]
				# right
				else:
					normal = -self._iVector
					scaleFactor = (self._top_right_corner[0] - previousPos[0]) / step[0]

				# the components of the "step" vector before...
				stepBeforeBounce = scaleFactor*step
				
				# ...and after reaching the wall
				stepAfterBounce = (1.0-scaleFactor)*step
				
				# in case we need to bounce again, we update the previous position...
				previousPos = previousPos + stepBeforeBounce

				# ...and the step, this one by computing the reflected ray using a formula involving the normal of the
				# reflecting surface...
				step = stepAfterBounce - 2*normal*np.dot(normal.T, stepAfterBounce)

				tentativeNewPos = previousPos + step
				
				# the direction of the velocity is updated according to the reflection that occured in the trajectory
				# note that this only needs to be done in the last iteration of the while loop, but since the velocity
				# is not used within the "else" part, it's not a problem
				velocity = step/np.linalg.norm(step)*np.linalg.norm(velocity)

		return np.vstack((tentativeNewPos, velocity))


class OnEdgeResetTransitionKernel(UnboundedTransitionKernel):
	
	def __init__(self, bottom_left_corner, top_right_corner, velocity_variance=0.5, noise_variance=0.1, step_duration=1, PRNG=np.random.RandomState(), reset_velocity_variance=0.01):
		
		# the parent's constructor is called
		super().__init__(bottom_left_corner, top_right_corner, velocity_variance, noise_variance, step_duration, PRNG)
		
		self._resetVelocityVariance = reset_velocity_variance
		
	def next_state(self, state, PRNG=None):
		
		# the new (unbounded) state as computed by the parent class
		unboundedState = super().next_state(state, PRNG)
		
		# the first two elements of the above state is the new (here tentative) position...
		tentativeNewPos = unboundedState[:2]
		
		# if the tentative position is within the bounds of the rectangle...
		if (tentativeNewPos > self._corners[2]).all() and (tentativeNewPos < self._corners[0]).all():
			
			# it's already ok
			return unboundedState
		
		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		velocity = PRNG.normal(0, math.sqrt(self._resetVelocityVariance/2), (2,1))
		
		return np.vstack((state[0:2], velocity))
