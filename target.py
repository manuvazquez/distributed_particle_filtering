import state
import numpy as np

class Target:
	
	def __init__(self, prior, transition_kernel, tx_power=15, pseudo_random_numbers_generator=None):
		
		self._pseudo_random_numbers_generator = pseudo_random_numbers_generator
		self._prior = prior
		
		# the power with which the target is transmitting
		self._tx_power = tx_power

		# the transition kernel determines the movement of the target
		self._transition_kernel = transition_kernel
	
	def reset(self):
		
		self._state = self._prior.sample(PRNG=self._pseudo_random_numbers_generator)

	def pos(self):
		
		return state.position(self._state)
	
	def velocity(self):
		
		return state.velocity(self._state)
		
	def simulate_trajectory(self, nTimeInstants):
		
		# initial state is obtained by means of the prior...
		self._state = self._prior.sample(PRNG=self._pseudo_random_numbers_generator)
		
		# a trajectory with the requested number of time instants...plus the initial one
		computedTrajectory = np.empty((state.nElements,nTimeInstants))
		
		# initial state is set
		computedTrajectory[:,0:1] = self._state

		print('initial position:\n',state.position(computedTrajectory[:, 0:1]))
		print('initial velocity:\n',state.velocity(computedTrajectory[:, 0:1]))

		# the trajectory is simulated, and the corresponding observations are obtained (notice that there is no observation for initial position)
		for iTime in range(1, nTimeInstants):
			
			# a new state is obtained as the target moves...
			self._state = self._transition_kernel.nextState(self._state, PRNG=self._pseudo_random_numbers_generator)
			
			# ..and it is stored in the corresponding position (which is iTime+1)
			computedTrajectory[:,iTime:iTime+1] = self._state
		
		return (state.position(computedTrajectory),state.velocity(computedTrajectory))