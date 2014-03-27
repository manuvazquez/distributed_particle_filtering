import math
import numpy as np
import numpy.random

class Prior:
	
	def sample(self):
		pass

class BoundedUniformPrior(Prior):
	
	def __init__(self,bottomLeftCorner,topRightCorner):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
	def sample(self):
		
		res = np.empty(2)
		res[0],res[1] = numpy.random.uniform(self._bottomLeftCorner[0],self._topRightCorner[0]),numpy.random.uniform(self._bottomLeftCorner[1],self._topRightCorner[1])
		
		return res

class TransitionKernel:
	def nextState(self,currentPos,speed):
		pass
	
class StraightTransitionKernel(TransitionKernel):
	
	def __init__(self,angle):
		
		# the parent's constructor is called
		super().__init__()

		self._angle = math.radians(angle)

	def nextState(self,currentPos,speed):
		return (currentPos + np.array([math.cos(self._angle),math.sin(self._angle)]),speed)

class BoundedRandomSpeedTransitionKernel(TransitionKernel):
	
	def __init__(self,bottomLeftCorner,topRightCorner,speedVariance=0.5,noiseVariance=0.5,stepDuration=1):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
		self._speedVariance = speedVariance
		self._noiseVariance = noiseVariance
		self._stepDuration = stepDuration
		
	def nextState(self,currentPos,speed):
		
		# the speed changes BEFORE moving...
		speed = speed + numpy.random.randn(2)*math.sqrt(self._speedVariance/2)
		
		# ...and a new "tentative" position is obtained from the previous one, the speed and a noise component
		tentativeNewPos = currentPos + speed*self._stepDuration + numpy.random.randn(2)*math.sqrt(self._noiseVariance)
		
		while(True):
		
			# if the new position is to the left of the left bound...
			if (tentativeNewPos[0] < self._bottomLeftCorner[0]) or (tentativeNewPos[0] > self._topRightCorner[0]):
				
				# we compute the angle between the velocity vector and a unit vertical vector (with coordinates [0,1]) using the dot product
				angle = math.acos(speed[1]/np.linalg.norm(speed))
				
				# the rotation must be clockwise (*-1) if we reached the left bound and counter-clockwise if going through the right one (*1)
				rotationAngle = (-1)**(tentativeNewPos[0] < self._bottomLeftCorner[0]) * 2 * angle
				
				# we compute the corresponding rotation matrix
				rotationMatrix = np.array([[math.cos(rotationAngle),-math.sin(rotationAngle)],[math.sin(rotationAngle),math.cos(rotationAngle)]])
				
				speed = np.dot(rotationMatrix,speed)
				
				tentativeNewPos = currentPos + speed*self._stepDuration + numpy.random.randn(2)*math.sqrt(self._noiseVariance)
				
				#print('going out to the left')
				#print('speed:',speed)
				#print('angle:',math.degrees(angle))
				#print('rotating ',math.degrees(rotationAngle))
				
				#import code
				#code.interact(local=dict(globals(), **locals()))

			elif (tentativeNewPos[1] < self._bottomLeftCorner[1]) or (tentativeNewPos[1] > self._topRightCorner[1]):
				
				# we compute the angle between the velocity vector and a unit horizontal vector (with coordinates [1,0]) using the dot product
				angle = math.acos(speed[0]/np.linalg.norm(speed))
				
				# we need to distinguish two cases:
				if angle < (math.pi/2):
					rotationAngle = (-1)**(tentativeNewPos[1] >= self._bottomLeftCorner[1]) * 2 * angle
				else:
					rotationAngle = (-1)**(tentativeNewPos[1] >= self._bottomLeftCorner[1]) * 2 * (math.pi - angle)

				# we compute the corresponding rotation matrix
				rotationMatrix = np.array([[math.cos(rotationAngle),-math.sin(rotationAngle)],[math.sin(rotationAngle),math.cos(rotationAngle)]])
				
				speed = np.dot(rotationMatrix,speed)
				
				tentativeNewPos = currentPos + speed*self._stepDuration + numpy.random.randn(2)*math.sqrt(self._noiseVariance)
				
				#print('going out below')
				#print('angle:',math.degrees(angle))
				#print('rotating ',math.degrees(rotationAngle))

			else:
				# the new position is OK
				break
		
		return (tentativeNewPos,speed)