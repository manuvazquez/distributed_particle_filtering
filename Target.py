import math
import numpy

class Motor:
	def move(x,y):
		pass
	
class RadialMotor(Motor):
	
	def __init__(self,angle,radius):
		self._angle = math.radians(angle)
		self.radius = radius

	def move(self,pos):
		return pos + numpy.array([math.cos(self._angle),math.sin(self._angle)])

class Target:
	
	def __init__(self, motor, xStart=0 ,yStart=0, txPower=15):

		# initialization of the position if received as argument...
		self._pos = numpy.array([xStart,yStart])
		
		# ...the same for the transmission power
		self._txPower = txPower

		self._motor = motor

	def pos(self):
		
		return self._pos
		
	def step(self):
		
		print('should be taking a step')
		
		self._pos = self._motor.move(self._pos)