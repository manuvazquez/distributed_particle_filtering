import numpy as np
import random
import math

class Sensor:
	def __init__(self,xPos,yPos,threshold,probDetection=0.9,probFalseAlarm=0.1):
		
		# position is saved in a numpy array for later use
		self._pos = np.array([xPos,yPos])
		
		# the distance within reach of the sensor
		self._threshold = threshold
		
		# the probability of (correct) detection
		self._probDetection = probDetection
		
		# the probability of false alarm
		self._probFalseAlarm = probFalseAlarm
		
	def detect(self,targetPos):
		
		print('sensor position:',self._pos)
		print('target position:',targetPos)
		
		distance = np.linalg.norm((self._pos - targetPos))
		
		print('distance:',distance)
		
		if distance<self._threshold:
			return random.random()<self._probDetection
		else:
			return random.random()<self._probFalseAlarm

class SensorLayer:
	
	def __init__(self):
		pass

	def getPositions(self,nSensors):
		pass

class EquispacedOnRectangleSensorLayer(SensorLayer):
	
	def __init__(self,bottomLeftCorner,topRightCorner):

		super().__init__()
	
		# a vector representing the diagonal of the rectangle, from which we compute...
		self._diagonal = topRightCorner - bottomLeftCorner
		
		# ...the area
		self._area = self._diagonal.prod()
		
		self._bottomLeftCorner = bottomLeftCorner
		
	def getPositions(self,nSensors):
		
		# if the sensors are equispaced, each sensor should cover an area equal to
		areaPerSensor = self._area/nSensors
		
		# if the area covered by each sensor is a square, then its side is
		squareSide = math.sqrt(areaPerSensor)
		
		# number of "full" squares that fit in each dimension
		nSquaresInXdimension = np.floor(self._diagonal[0]/squareSide)
		nSquaresInYdimension = np.floor(self._diagonal[1]/squareSide)
		
		nOverfittingSensors = (nSquaresInXdimension+1)*(nSquaresInYdimension+1)
		
		# if adding a sensor in each dimension we get closer to the number of requested sensors...
		if (nSensors-(nSquaresInXdimension*nSquaresInYdimension)) > (nOverfittingSensors-nSensors):
			# ...we repeat the computations with the "overfitting" number of sensors
			areaPerSensor = self._area/nOverfittingSensors
			squareSide = math.sqrt(areaPerSensor)
			nSquaresInXdimension = np.floor(self._diagonal[0]/squareSide)
			nSquaresInYdimension = np.floor(self._diagonal[1]/squareSide)
		
		print('nSquares:',nSquaresInXdimension,nSquaresInYdimension)
		
		# in each dimension there is a certain length that is not covered (using % "weird" things happen sometimes...)
		remainingInXdimension = self._diagonal[0] - nSquaresInXdimension*squareSide
		remainingInYdimension = self._diagonal[1] - nSquaresInYdimension*squareSide
		
		#import code
		#code.interact(local=dict(globals(), **locals()))
		
		#res = [np.array([(remainingInXdimension + squareSide)/2 + i*squareSide,(remainingInYdimension + squareSide)/2 + j*squareSide]) for i in range(int(nSquaresInXdimension)) for j in range(int(nSquaresInYdimension))]
		
		# avoided list comprehension in order to get a numpy array as result
		res = np.empty([nSquaresInXdimension*nSquaresInYdimension,2])
		iSensor = 0
		for i in range(int(nSquaresInXdimension)):
			for j in range(int(nSquaresInYdimension)):
				res[iSensor,0] = self._bottomLeftCorner[0] + (remainingInXdimension + squareSide)/2 + i*squareSide
				res[iSensor,1] = self._bottomLeftCorner[1] + (remainingInYdimension + squareSide)/2 + j*squareSide
				iSensor = iSensor + 1

		print(res)
		
		return res