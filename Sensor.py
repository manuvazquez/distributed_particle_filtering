import numpy
import random

class Sensor:
	def __init__(self,xPos,yPos,threshold,probDetection=0.9,probFalseAlarm=0.1):
		
		# position is saved in a NUMPY array for later use
		self._pos =numpy.array([xPos,yPos])
		
		# the distance within reach of the sensor
		self._threshold = threshold
		
		# the probability of (correct) detection
		self._probDetection = probDetection
		
		# the probability of false alarm
		self._probFalseAlarm = probFalseAlarm
		
	def detect(self,targetPos):
		
		print('sensor position:',self._pos)
		print('target position:',targetPos)
		
		distance = numpy.linalg.norm((self._pos - targetPos))
		
		print('distance:',distance)
		
		if distance<self._threshold:
			return random.random()<self._probDetection
		else:
			return random.random()<self._probFalseAlarm