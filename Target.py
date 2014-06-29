import State

class Target:
	
	def __init__(self, prior, transitionKernel, txPower=15, PRNG=None):
		
		self._PRNG = PRNG
		
		# initial state is obtained by means of the prior...
		initialState = prior.sample(PRNG=self._PRNG)

		# ...and used to initialize the position...
		self._pos = State.position(initialState)
		
		# ...and the speed
		self._velocity = State.velocity(initialState)
		
		# the power with which the target is transmitting
		self._txPower = txPower

		# the transition kernel determines the movement of the target
		self._transitionKernel = transitionKernel

	def pos(self):
		
		return self._pos
	
	def velocity(self):
		
		return self._velocity
		
	def step(self):
		
		# the new state is first computed...
		newState = self._transitionKernel.nextState(State.buildState(self._pos,self._velocity),PRNG=self._PRNG)
		
		# ...and the position and velocity are obtained thereof
		self._pos,self._velocity = State.position(newState),State.velocity(newState)