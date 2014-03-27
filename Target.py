import numpy as np
import numpy.random

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
		
		self._pos,self._speed = self._motor.nextState(self._pos,self._speed)