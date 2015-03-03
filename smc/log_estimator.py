import numpy as np

import state
import smc.estimator

class Mposterior(smc.estimator.Mposterior):
	
	def estimate(self,DPF):
		
		# the distributions computed by every PE are gathered in a list of tuples (samples and weights after exponentiation)
		posteriors = [(PE.getState().T,np.exp(PE.weights)) for PE in DPF._PEs]
		
		return self.combinePosteriorDistributions(DPF,posteriors)

class MposteriorSubset(smc.estimator.MposteriorSubset):
	
	def estimate(self,DPF):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		posteriors = [(PE.getSamplesAt(DPF._resamplingAlgorithm.getIndexes(np.exp(PE.weights),self._nParticles)).T,
				 np.full(self._nParticles,1.0/self._nParticles)) for PE in DPF._PEs]
		
		return self.combinePosteriorDistributions(DPF,posteriors)

# so that this module shares the same API as the "estimator" module
class GeometricMedian(smc.estimator.GeometricMedian):
	
	pass