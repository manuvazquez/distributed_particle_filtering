import math
import numpy as np

# a state vector has the structure:
#	[ pos_x ]
#	| pos_y ]
#	| vel_x ]
#	[ vel_y ]

def position(state):
	"""It extracts the position elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the state vector.
	"""
	
	return state[0:2,:]

def velocity(state):
	"""It extracts the velocity elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the state vector.
	"""
	
	return state[2:4,:]

def buildState(position,velocity):
	"""It builds a state vector given a position and a velocity vectors.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the state vector.
	"""
	
	return np.vstack((position,velocity))

class Prior:
	
	def sample(self):
		
		pass

class UniformBoundedPositionGaussianVelocityPrior(Prior):
	
	def __init__(self,bottomLeftCorner,topRightCorner,velocityMean=0,velocityVariance=0.25,PRNG=np.random.RandomState()):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
		self._velocityMean = velocityMean
		self._velocityVariance = velocityVariance
		
		self._PRNG = PRNG
		
	def sample(self,nSamples=1,PRNG=None):
		
		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		# position
		position = np.empty((2,nSamples))
		position[0,:],position[1,:] = PRNG.uniform(self._bottomLeftCorner[0],self._topRightCorner[0],nSamples),PRNG.uniform(self._bottomLeftCorner[1],self._topRightCorner[1],nSamples)
		
		# velocity
		velocity = PRNG.normal(self._velocityMean,math.sqrt(self._velocityVariance/2),(2,nSamples))
		
		return np.vstack((position,velocity))

class TransitionKernel:
	
	def __init__(self,stepDuration):
		
		# a step lasts this time units (distance travelled each step equals velocity times this value)
		self._stepDuration = stepDuration
	
	def nextState(self,state):
		
		pass
	
class StraightTransitionKernel(TransitionKernel):
	
	def __init__(self,angle,stepDuration=1):
		
		# the parent's constructor is called
		super().__init__(stepDuration)

		self._angle = math.radians(angle)

	def nextState(self,state):
		
		# the position is increased linearly sloping at the given angle
		position = state[0:2] + np.array([[math.cos(self._angle)],[math.sin(self._angle)]])
		
		# the velocity stays the same...so
		velocity = state[2:4]
	
		return np.vstack((position,velocity))
	

class BouncingWithinRectangleTransitionKernel(TransitionKernel):
	
	def __init__(self,bottomLeftCorner,topRightCorner,velocityVariance=0.5,noiseVariance=0.1,stepDuration=1,PRNG=np.random.RandomState()):
		
		# the parent's constructor is called
		super().__init__(stepDuration)
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
		# variance of the noise affecting the velocity
		self._velocityVariance = velocityVariance
		
		# variance of the noise affecting the position
		self._noiseVariance = noiseVariance
		
		# the pseudo random numbers generator
		self._PRNG = PRNG
		
		# canonical vectors used in computations
		self._iVector = np.array([[1.0],[0.0]])
		self._jVector = np.array([[0.0],[1.0]])
		
		# top right, top left, bottom left, and bottom right corners stored as column vectors
		self._corners = [topRightCorner[np.newaxis].T,np.array([[bottomLeftCorner[0]],[topRightCorner[1]]]),bottomLeftCorner[np.newaxis].T,np.array([[topRightCorner[0]],[bottomLeftCorner[1]]])]
		
	def nextState(self,state,PRNG=None):
		
		# if for this particular call, no pseudo random numbers generator is received...
		if PRNG is None:
			# ...the corresponding class attribute is used
			PRNG = self._PRNG
		
		# not actually needed...but for the sake of clarity...
		velocity = state[2:4]

		# the velocity changes BEFORE moving...
		velocity += PRNG.normal(0,math.sqrt(self._velocityVariance/2),(2,1))
		
		# step to be taken is obtained from the velocity and a noise component
		step = velocity*self._stepDuration + PRNG.normal(0,math.sqrt(self._noiseVariance/2),(2,1))
		
		# this may be updated in the while loop when bouncing off several walls
		previousPos = state[0:2].copy()
		
		# a new "tentative" position is obtained from the previous one and the step
		tentativeNewPos = previousPos + step
		
		# to be used in the loop...
		anglesWithCorners = np.empty(4)
		
		while(True):
			
			if (tentativeNewPos > self._corners[2]).all() and (tentativeNewPos < self._corners[0]).all():
			
				# the new position is OK
				break
			
			else:
				
				for i,corner in enumerate(self._corners):
					
					# a vector joining the previous position and the corresponding corner
					positionToCorner = corner - previousPos
					
					# the angle between the above vector and a horizontal line
					anglesWithCorners[i] = math.acos(positionToCorner[0]/np.linalg.norm(positionToCorner))
				
				# we account for the fact that the angle between two vectors computed by means of the dot product is always between 0 and pi (the shortest)
				anglesWithCorners[2:] = 2*math.pi - anglesWithCorners[2:]
				
				# angle between the step to be taken and the horizontal line
				angle = math.acos(step[0]/np.linalg.norm(step))
				
				if step[1]<=0:
					# we account for the fact that the angle between two vectors computed by means of the dot product is always between 0 and pi (the shortest)
					angle = 2*math.pi - angle
				
				# up
				if anglesWithCorners[0] <= angle < anglesWithCorners[1]:
					normal = -self._jVector
					scaleFactor = (self._topRightCorner[1] - previousPos[1])/step[1]
				# left
				elif anglesWithCorners[1] <= angle < anglesWithCorners[2]:
					normal = self._iVector
					scaleFactor = (self._bottomLeftCorner[0] - previousPos[0])/step[0]
				# down
				elif anglesWithCorners[2] <= angle < anglesWithCorners[3]:
					normal = self._jVector
					scaleFactor = (self._bottomLeftCorner[1] - previousPos[1])/step[1]
				# right
				else:
					normal = -self._iVector
					scaleFactor = (self._topRightCorner[0] - previousPos[0])/step[0]

				# the components of the "step" vector before...
				stepBeforeBounce = scaleFactor*step
				
				# ...and after reaching the wall
				stepAfterBounce = (1.0-scaleFactor)*step
				
				# in case we need to bounce again, we update the previous position...
				previousPos = previousPos + stepBeforeBounce

				# ...and the step, this one by computing the reflected ray using a formula involving the normal of the reflecting surface...
				step = stepAfterBounce - 2*normal*np.dot(normal.T,stepAfterBounce)

				tentativeNewPos = previousPos + step
				
				# the direction of the velocity is updated according to the reflection that occured in the trajectory
				# note that this only needs to be done in the last iteration of the while loop, but since the velocity is not used within the "else" part, it's not a problem
				velocity = step/np.linalg.norm(step)*np.linalg.norm(velocity)

		return np.vstack((tentativeNewPos,velocity))