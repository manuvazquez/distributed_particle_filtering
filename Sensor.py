import numpy as np
import numpy.random
import math

class Sensor:
	def __init__(self,position,threshold,probDetection=0.9,probFalseAlarm=0.01,PRNG=numpy.random.RandomState()):
		
		# position is saved for later use
		self._position = position
		
		# the distance within reach of the sensor
		self._threshold = threshold
		
		# the probability of (correct) detection
		self._probDetection = probDetection
		
		# the probability of false alarm
		self._probFalseAlarm = probFalseAlarm
		
		# pseudo random numbers generator
		self._PRNG = PRNG
		
		# for the sake of convenience when computing the likelihood: we keep an array with the probability mass funciton of the observations conditional on the target being close enough (it depends on the threshold)...
		# self._pmfObservationsWhenClose[x] = p(observation=x | |<target position> - <sensor position>| < threshold)
		self._pmfObservationsWhenClose = np.array([1-probDetection,probDetection])
		
		# ...and that of the observations conditional on the target being far
		self._pmfObservationsWhenFar = np.array([1-probFalseAlarm,probFalseAlarm])
		
	def detect(self,targetPos):
		
		distance = np.linalg.norm((self._position - targetPos))
		
		if distance<self._threshold:
			return self._PRNG.rand()<self._probDetection
		else:
			return self._PRNG.rand()<self._probFalseAlarm

	def likelihood(self,observation,positions):
		
		# the distances to ALL the positions are computed
		distances = np.linalg.norm(np.subtract(positions,self._position),axis=0)

		# an empty array with the same dimensions as distances is created
		likelihoods = np.empty_like(distances)

		# the likelihood for a given observation is computed using probability mass funciton if the target is within the reach of the sensor...
		likelihoods[distances<self._threshold] = self._pmfObservationsWhenClose[observation]
		
		#...and a different one if it's outside it
		likelihoods[distances>=self._threshold] = self._pmfObservationsWhenFar[observation]

		return likelihoods

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
		
		# in each dimension there is a certain length that is not covered (using % "weird" things happen sometimes...)
		remainingInXdimension = self._diagonal[0] - nSquaresInXdimension*squareSide
		remainingInYdimension = self._diagonal[1] - nSquaresInYdimension*squareSide
		
		res = np.transpose(
				np.array(
					[
						[self._bottomLeftCorner[0] + (remainingInXdimension + squareSide)/2 + i*squareSide,self._bottomLeftCorner[1] + (remainingInYdimension + squareSide)/2 + j*squareSide]
						for i in range(int(nSquaresInXdimension)) for j in range(int(nSquaresInYdimension))]))


		return res