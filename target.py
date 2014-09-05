import state
import numpy as np

class Target:
	
	def __init__(self, prior, transitionKernel, txPower=15, PRNG=None):
		
		self._PRNG = PRNG
		
		# initial state is obtained by means of the prior...
		self._state = prior.sample(PRNG=self._PRNG)

		# the power with which the target is transmitting
		self._txPower = txPower

		# the transition kernel determines the movement of the target
		self._transitionKernel = transitionKernel

	def pos(self):
		
		return state.position(self._state)
	
	def velocity(self):
		
		return state.velocity(self._state)
		
	def simulateTrajectory(self,sensors,nTimeInstants):
		
		# a trajectory with the requested number of time instants...plus the initial one
		computedTrajectory = np.empty((state.nElements(),nTimeInstants+1))
		
		# initial state is set
		computedTrajectory[:,0:1] = self._state

		print('initial position:\n',state.position(computedTrajectory[:,0:1]))
		print('initial velocity:\n',state.velocity(computedTrajectory[:,0:1]))

		# the trajectory is simulated, and the corresponding observations are obtained (notice that there is no observation for initial position)
		for iTime in range(nTimeInstants):
			
			# a new state is obtained as the target moves...
			self._state = self._transitionKernel.nextState(self._state,PRNG=self._PRNG)
			
			# ..and it is stored in the corresponding position (which is iTime+1)
			computedTrajectory[:,iTime+1:iTime+2] = self._state
		
		return (state.position(computedTrajectory[:,0:1]),state.velocity(computedTrajectory[:,0:1]),state.position(computedTrajectory[:,1:]),state.velocity(computedTrajectory[:,1:]))