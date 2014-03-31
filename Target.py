import State

class Target:
	
	def __init__(self, transitionKernel, initialPosition, initialVelocity, txPower=15):

		# initialization of the position if received as argument...
		self._pos = initialPosition
		
		# ...the same for the transmission power...
		self._txPower = txPower
		
		# ...and the speed
		self._velocity = initialVelocity

		# the transition kernel determines the movement of the target
		self._transitionKernel = transitionKernel

	def pos(self):
		
		return self._pos
	
	def speed(self):
		
		return self._velocity
		
	def step(self):
		
		print('taking a step...')

		# the new state is first computed...
		newState = self._transitionKernel.nextState(State.buildState(self._pos,self._velocity))
		
		# ...and the position and velocity are obtained thereof
		self._pos,self._velocity = State.position(newState),State.velocity(newState)