import numpy as np

import state
from smc import particle_filter

class CentralizedTargetTrackingParticleFilter(particle_filter.CentralizedTargetTrackingParticleFilter):
	
	def initialize(self):
		
		# let the parent do its thing...
		super().initialize()
		
		self._weights = np.log(self._weights)
		
	def step(self,observations):
		
		assert len(observations) == len(self._sensors)
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._stateTransitionKernel.nextState(self._state[:,i:i+1]) for i in range(self._nParticles)])
		
		# for each sensor, we compute the likelihood of EVERY particle (position)
		loglikelihoods = np.log(np.array([sensor.likelihood(obs,state.position(self._state)) for sensor,obs in zip(self._sensors,observations)]))
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		loglikelihoodsProduct = loglikelihoods.sum(axis=0)
		
		# the weights are updated
		self._weights += loglikelihoodsProduct
		
		# the aggregated weight is kept up to date at all times
		self.updateAggregatedWeight()
		
		# whatever is required (it depends on the algorithm) to avoid weights degeneracy...
		self.avoidWeightDegeneracy()

	def resample(self,normalizedLogWeights):
		
		normalizedWeights = np.exp(normalizedLogWeights)
		
		# we check whether a resampling step is actually needed or not
		if self._resamplingCriterion.isResamplingNeeded(normalizedWeights):
			
			try:
				# the resampling algorithm is used to decide which particles to keep
				iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(normalizedWeights)
				
			except ValueError:
				
				print("(log) CentralizedTargetTrackingParticleFilter:resample: this shouldn't have happened...")
				
				import code
				code.interact(local=dict(globals(), **locals()))
			
			# the above indexes are used to update the state
			self._state = self._state[:,iParticlesToBeKept]
			
			# note that if the weights have been normalized ("standard" centralized particle filter), then "self._aggregatedWeight" is equal to 1
			self._weights.fill(np.log(self._aggregatedWeight)-np.log(self._nParticles))
		
	def updateAggregatedWeight(self):
		
		# the aggregated weight is simply the sum of the non-normalized weights...after exponentiation
		self._aggregatedWeight = np.exp(self._weights).sum()

	def computeMean(self):
		
		# if all the weights in this PF/PE are zero...
		if self._aggregatedWeight==0:
			
			# ...then an all-zeros estimate is returned...though any should do since this estimate must contribute zero
			return np.zeros((state.nElements,1))
		
		normalizedLogWeights = self._weights - np.log(self._aggregatedWeight)
		
		# element-wise multiplication of the state vectors and their correspondent weights...followed by addition => weighted mean
		return np.multiply(self._state,np.exp(normalizedLogWeights)).sum(axis=1)[np.newaxis].T

	# this methods encapsulates the parts within the code of "step" which are different in this class and its children
	def avoidWeightDegeneracy(self):
		
		# if all the weights are zero...
		if self._aggregatedWeight==0:
			
			# ...then normalization makes no sense and we just initialize the weights again
			self._weights.fill(-np.log(self._nParticles))

		else:
		
			self._weights -= np.log(self._aggregatedWeight)
			
		# we forced this above
		self._aggregatedWeight = 1.0
		
		# the normalized weights are used to resample
		self.resample(self._weights)
	
# =========================================================================================================

class EmbeddedTargetTrackingParticleFilter(CentralizedTargetTrackingParticleFilter):
	
	def getAggregatedWeight(self):
		
		return self._aggregatedWeight
	
	def scaleWeights(self,factor):
		
		self._weights += np.log(factor)
		self._aggregatedWeight *= factor

	def avoidWeightDegeneracy(self):
		
		# if all the weights are zero...
		if self._aggregatedWeight==0:
			
			# ...there is nothing we can do
			return
		
		else:
			# the normalized weights are used to resample
			self.resample(self._weights - np.log(self._aggregatedWeight))

# =========================================================================================================

class TargetTrackingParticleFilterWithDRNA(particle_filter.TargetTrackingParticleFilterWithDRNA):

	def resetWeights(self):
		
		"""It sets every weight of every PE to the same value.
		"""
		
		# every PE will be assigned the same aggregated weight,
		aggregatedWeight = 1.0/self._nPEs
		
		# for every PE in this DPF...
		for PE in self._PEs:
			
			# the aggregated weight is set...
			PE._aggregatedWeight = aggregatedWeight
			
			# ...along with the individual weights within the PE
			PE.weights = np.full(PE._nParticles,-np.log(self._nPEs)-np.log(PE._nParticles))

class DistributedTargetTrackingParticleFilterWithParticleExchangingMposterior(particle_filter.DistributedTargetTrackingParticleFilterWithParticleExchangingMposterior):
	
	def share(self):
		
		# each PE draws a set of samples from its probability measure...to be shared with its neighbours
		samplesToBeShared = [PE.getSamplesAt(self._resamplingAlgorithm.getIndexes(np.exp(PE.weights),self._nSharedParticles)) for PE in self._PEs]
		
		# the list of neighbours of each PE
		PEsNeighbours = self._topology.getNeighbours()
		
		# for every PE...
		for iPE,(PE,neighbours) in enumerate(zip(self._PEs,PEsNeighbours)):
			
			# ...the particles shared by its neighbours (assumed to be uniformly distributed) are gathered...
			subsetPosteriorDistributions = [(samplesToBeShared[i].T,np.full(self._nSharedParticles,1.0/self._nSharedParticles)) for i in neighbours]
			
			# ...along with its own (shared, already sampled) particles
			subsetPosteriorDistributions.append((samplesToBeShared[iPE].T,np.full(self._nSharedParticles,1.0/self._nSharedParticles)))
			
			# M posterior on the posterior distributions collected above
			jointParticles,jointWeights = self.Mposterior(subsetPosteriorDistributions)
			
			# the indexes of the particles to be kept
			iNewParticles = self._resamplingAlgorithm.getIndexes(jointWeights,PE._nParticles)
			
			PE.samples = jointParticles[:,iNewParticles]
			PE.weights = np.full(PE._nParticles,-np.log(PE._nParticles))
			PE.updateAggregatedWeight()