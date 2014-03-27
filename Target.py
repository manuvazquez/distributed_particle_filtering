import math
#import random
import numpy as np
import numpy.random

class Motor:
	def move(self,currentPos,speed):
		pass
	
class RadialMotor(Motor):
	
	def __init__(self,angle):
		
		# the parent's constructor is called
		super().__init__()

		self._angle = math.radians(angle)

	def move(self,currentPos,speed):
		return (currentPos + np.array([math.cos(self._angle),math.sin(self._angle)]),speed)

class BoundedRandomSpeedMotor(Motor):
	
	def __init__(self,bottomLeftCorner,topRightCorner,speedVariance=0.5,noiseVariance=0.5,stepDuration=1):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		
		self._speedVariance = speedVariance
		self._noiseVariance = noiseVariance
		self._stepDuration = stepDuration
		
	def move(self,currentPos,speed):
		
		print('in BoundedRandomSpeedMotor.move')
		
		# the speed changes BEFORE moving...
		speed = speed + numpy.random.randn(2)*math.sqrt(self._speedVariance/2)
		
		tentativeNewPos = currentPos + speed*self._stepDuration + numpy.random.randn(2)*math.sqrt(self._noiseVariance)
		
		# if the new position is to the left of the left bound...
		if (tentativeNewPos[0] < self._bottomLeftCorner[0]) or (tentativeNewPos[0] > self._topRightCorner[0]):
			
			print('going out to the left')
			
			angle = math.acos(speed[1]/np.linalg.norm(speed))
			
			print('angle:',math.degrees(angle))
			
			# the rotation must be clockwise (*-1) if we reached the left bound and counter-clockwise if going through the right one (*1)
			rotationAngle = (-1)**(tentativeNewPos[0] < self._bottomLeftCorner[0]) * 2 * angle
			
			print('rotating ',math.degrees(rotationAngle))
			
			# we compute the corresponding rotation matrix
			rotationMatrix = np.array([[math.cos(rotationAngle),-math.sin(rotationAngle)],[math.sin(rotationAngle),math.cos(rotationAngle)]])
			
			speed = np.dot(rotationMatrix,speed)
			
			print('speed:',speed)
			print('---')
			
			tentativeNewPos = currentPos + speed*self._stepDuration + numpy.random.randn(2)*math.sqrt(self._noiseVariance)
			
			#import code
			#code.interact(local=dict(globals(), **locals()))

		#elif tentativeNewPos[1] < self._bottomLeftCorner[1]:
		elif (tentativeNewPos[1] < self._bottomLeftCorner[1]) or (tentativeNewPos[1] > self._topRightCorner[1]):
			
			print('going out below')
			
			angle = math.acos(speed[0]/np.linalg.norm(speed))
			
			if angle < (math.pi/2):
				rotationAngle = (-1)**(tentativeNewPos[1] >= self._bottomLeftCorner[1]) * 2 * angle
			else:
				print('larger than 90..')
				rotationAngle = (-1)**(tentativeNewPos[1] >= self._bottomLeftCorner[1]) * 2 * (math.pi - angle)
				
			print('angle:',math.degrees(angle))
			print('rotating ',math.degrees(rotationAngle))

			# we compute the corresponding rotation matrix
			rotationMatrix = np.array([[math.cos(rotationAngle),-math.sin(rotationAngle)],[math.sin(rotationAngle),math.cos(rotationAngle)]])
			
			speed = np.dot(rotationMatrix,speed)
			
			tentativeNewPos = currentPos + speed*self._stepDuration + numpy.random.randn(2)*math.sqrt(self._noiseVariance)

		else:
			print("that's fine")
		
		# the new position is obtained from the previous one, the speed and a noise component
		return (tentativeNewPos,speed)		

class Target:
	
	def __init__(self, motor, initialPosition=np.array([0,0]), initialSpeed=numpy.random.randn(2), txPower=15):

		# initialization of the position if received as argument...
		self._pos = initialPosition
		
		# ...the same for the transmission power...
		self._txPower = txPower
		
		# ...and the speed
		self._speed = initialSpeed

		self._motor = motor

	def pos(self):
		
		return self._pos
	
	def speed(self):
		
		return self._speed
		
	def step(self):
		
		print('taking a step...')
		
		self._pos,self._speed = self._motor.move(self._pos,self._speed)