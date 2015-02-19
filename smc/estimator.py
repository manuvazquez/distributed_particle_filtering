import abc
import numpy as np

class Estimator(metaclass=abc.ABCMeta):
	
	@abc.abstractmethod
	def estimate(self,DPF):
		
		return

class Mposterior(Estimator):
	
	def combinePosteriorDistributions(self,DPF,posteriors):
		
		# the Mposterior algorithm is used to obtain a a new distribution
		jointParticles,jointWeights = DPF.Mposterior(posteriors)
		
		return np.multiply(jointParticles,jointWeights).sum(axis=1)[np.newaxis].T
	
	def estimate(self,DPF):
		
		# the distributions computed by every PE are gathered in a list of tuples (samples and weights)
		posteriors = [(PE.getState().T,PE.weights) for PE in DPF._PEs]
		
		return self.combinePosteriorDistributions(DPF,posteriors)

class MposteriorSubset(Mposterior):
	
	def __init__(self,nParticles):
		
		self._nParticles = nParticles
	
	def estimate(self,DPF):

		# a number of samples is drawn from the distribution of each PE (all equally weighted) to build a list of tuples (samples and weights)
		posteriors = [(PE.getSamplesAt(DPF._resamplingAlgorithm.getIndexes(PE.weights,self._nParticles)).T,
				 np.full(self._nParticles,1.0/self._nParticles)) for PE in DPF._PEs]
		
		return self.combinePosteriorDistributions(DPF,posteriors)