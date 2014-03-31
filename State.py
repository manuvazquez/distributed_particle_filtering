import math
import numpy as np
import numpy.random

# a state vector has the structure:
#	[ pos_x ]
#	| pos_y ]
#	| vel_x ]
#	[ vel_y ]

def position(state):
	"""It extracts the position elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the state vector.
	"""
	
	return state[0:2]

def velocity(state):
	"""It extracts the velocity elements out of the state vector.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the state vector.
	"""
	
	return state[2:4]

def buildState(position,velocity):
	"""It builds a state vector given a position and a velocity vectors.
	
	The purpose is to encapsulate the state so that other modules/classes don't need to know about the structure of the state vector.
	"""
	
	return np.vstack((position,velocity))

class Prior:
	
	def sample(self):
		
		pass

class UniformBoundedPositionGaussianVelocityPrior(Prior):
	
	def __init__(self,bottomLeftCorner,topRightCorner,velocityMean=0,velocityVariance=0.25):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
		self._velocityMean = velocityMean
		self._velocityVariance = velocityVariance
		
	def sample(self,nSamples=1):
		
		# position
		position = np.empty((2,nSamples))
		position[0,:],position[1,:] = numpy.random.uniform(self._bottomLeftCorner[0],self._topRightCorner[0],nSamples),numpy.random.uniform(self._bottomLeftCorner[1],self._topRightCorner[1],nSamples)
		
		# velocity
		velocity = numpy.random.normal(self._velocityMean,math.sqrt(self._velocityVariance/2),(2,nSamples))
		
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
		
		# the position is modified linearly sloping at the given angle, and the speed stays the same
		return (currentPos + np.array([math.cos(self._angle),math.sin(self._angle)]),speed)
	
		# the position is increased linearly sloping at the given angle
		position = state[0:2] + np.array([[math.cos(self._angle)],[math.sin(self._angle)]])
		
		# the velocity stays the same...so
		velocity = state[2:4]
	
		return np.vstack((position,velocity))
	

class UniformBoundedPositionGaussianVelocityTransitionKernel(TransitionKernel):
	
	def __init__(self,bottomLeftCorner,topRightCorner,velocityVariance=0.5,noiseVariance=0.5,stepDuration=1):
		
		# the parent's constructor is called
		super().__init__(stepDuration)
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
		# variance of the noise affecting the velocity
		self._velocityVariance = velocityVariance
		
		# variance of the noise affecting the position
		self._noiseVariance = noiseVariance
		
	def nextState(self,state):
		
		# not actually needed...but for the sake of clarity...
		velocity = state[2:4]
		
		# the velocity changes BEFORE moving...
		velocity += numpy.random.normal(0,math.sqrt(self._velocityVariance/2),(2,1))
		
		# ...and a new "tentative" position is obtained from the previous one, the velocity and a noise component
		tentativeNewPos = state[0:2] + velocity*self._stepDuration + numpy.random.normal(0,math.sqrt(self._noiseVariance/2),(2,1))
		
		while(True):
		
			# if the new position is to the left of the left bound...
			if (tentativeNewPos[0] < self._bottomLeftCorner[0]) or (tentativeNewPos[0] > self._topRightCorner[0]):
				
				# we compute the angle between the velocity vector and a unit vertical vector (with coordinates [0,1]) using the dot product
				angle = math.acos(velocity[1]/np.linalg.norm(velocity))
				
				# the rotation must be clockwise (*-1) if we reached the left bound and counter-clockwise if going through the right one (*1)
				rotationAngle = (-1)**(tentativeNewPos[0] < self._bottomLeftCorner[0]) * 2 * angle
				
				# we compute the corresponding rotation matrix
				rotationMatrix = np.array([[math.cos(rotationAngle),math.sin(rotationAngle)],[-math.sin(rotationAngle),math.cos(rotationAngle)]])
				
				velocity = np.dot(rotationMatrix,velocity)
				
				tentativeNewPos = state[0:2] + velocity*self._stepDuration + numpy.random.normal(0,math.sqrt(self._noiseVariance/2),(2,1))
				
				#import code
				#code.interact(local=dict(globals(), **locals()))

			elif (tentativeNewPos[1] < self._bottomLeftCorner[1]) or (tentativeNewPos[1] > self._topRightCorner[1]):
				
				# we compute the angle between the velocity vector and a unit horizontal vector (with coordinates [1,0]) using the dot product
				angle = math.acos(velocity[0]/np.linalg.norm(velocity))
				
				# we need to distinguish two cases:
				if angle < (math.pi/2):
					rotationAngle = (-1)**(tentativeNewPos[1] >= self._bottomLeftCorner[1]) * 2 * angle
				else:
					rotationAngle = (-1)**(tentativeNewPos[1] >= self._bottomLeftCorner[1]) * 2 * (math.pi - angle)

				# we compute the corresponding rotation matrix
				rotationMatrix = np.array([[math.cos(rotationAngle),math.sin(rotationAngle)],[-math.sin(rotationAngle),math.cos(rotationAngle)]])
				
				velocity = np.dot(rotationMatrix,velocity)
				
				tentativeNewPos = state[0:2] + velocity*self._stepDuration + numpy.random.normal(0,math.sqrt(self._noiseVariance/2),(2,1))
				
			else:
				# the new position is OK
				break
		
		#return (tentativeNewPos,velocity)
		return np.vstack((tentativeNewPos,velocity))