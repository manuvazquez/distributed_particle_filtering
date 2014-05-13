import numpy as np
import math

import State

class ParticleFilter:
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion):
		
		self._nParticles = nParticles
		
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		
	def initialize(self):

		pass
		
	def step(self,observations):
		
		pass
	
	def getState(self):
		
		pass

# =========================================================================================================

class CentralizedTargetTrackingParticleFilter(ParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,aggregatedWeight=1.0):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion)
		
		# a vector with the weights is created...but not initialized (that must be done by the "initialize" method)
		self._weights = np.empty(nParticles)
		
		# the state equation is encoded in the transition kernel
		self._stateTransitionKernel = stateTransitionKernel
		
		# the prior is needed to inialize the state
		self._prior = prior
		
		# the sensors are kept
		self._sensors = sensors
		
		# this variable just keeps tabs on the sum of all the weights
		self._aggregatedWeight = aggregatedWeight
	
	def initialize(self):
		
		# let the parent do its thing...
		super().initialize()
		
		# initial samples...
		self._state = self._prior.sample(self._nParticles)
		
		# the weights are assigned equal probabilities
		self._weights.fill(self._aggregatedWeight/self._nParticles)
		
	def step(self,observations):
		
		super().step(observations)
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._stateTransitionKernel.nextState(self._state[:,i:i+1]) for i in range(self._nParticles)])
		
		# for each sensor, we compute the likelihood of EVERY particle (position)
		likelihoods = np.array(
			[sensor.likelihood(observations[i],State.position(self._state)) for i,sensor in enumerate(self._sensors)]
			)
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		likelihoodsProduct = likelihoods.prod(axis=0)
		
		# the weights are updated
		self._weights *= likelihoodsProduct
		
		# the aggregated weight is kept up to date at any time
		self.updateAggregatedWeight()
		
		# if required (depending on the algorithm), the weights are normalized...
		self.normalizeWeightsIfRequired()
		
		# ...though the normalized weights are computed anyway if needed/possible (if all the weights are zero normalization makes no sense)
		if self._aggregatedWeight!=0:
			normalizedWeights = self._weights / self._aggregatedWeight
		
		# ...in order to perform resampling if needed
		if self._resamplingCriterion.isResamplingNeeded(normalizedWeights):
			
			try:
				# the resampling algorithm is used to decide which particles to keep
				iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(normalizedWeights)
				
			except ValueError:
				
				print("CentralizedTargetTrackingParticleFilter:step: this shouldn't have happened...")
				
				import code
				code.interact(local=dict(globals(), **locals()))
			
			# actual resampling
			self.resample(iParticlesToBeKept)

	def getState(self):
		
		return self._state

	def resample(self,indexes):
		
		self._state = self._state[:,indexes]
		
		# note that if the weights have been normalized, then "self._aggregatedWeight" is already equal to 1
		self._weights.fill(self._aggregatedWeight/self._nParticles)
		
	def getParticle(self,index):
		
		return (self._state[:,index:index+1].copy(),self._weights[index])
	
	def setParticle(self,index,particle):
		
		self._state[:,index:index+1] = particle[0]
		self._weights[index] = particle[1]
		
		# the sum of the weights might have changed...
		self.updateAggregatedWeight()

	def updateAggregatedWeight(self):
		
		# the aggregated weight is simply the sum of the non-normalized weights
		self._aggregatedWeight = self._weights.sum()

	def computeMean(self):
		
		normalizedWeights = self._weights / self._aggregatedWeight

		# element-wise multiplication of the state vectors and their correspondent weights...followed by addition => weighted mean
		return np.multiply(self._state,normalizedWeights).sum(axis=1)[np.newaxis].T
		
	def normalizeWeightsIfRequired(self):
		
		self._weights /= self._aggregatedWeight
		
		# we forced this above
		self._aggregatedWeight = 1.0

# =========================================================================================================

class EmbeddedTargetTrackingParticleFilter(CentralizedTargetTrackingParticleFilter):
	
	def normalizeWeightsIfRequired(self):
		
		pass

	def getAggregatedWeight(self):
		
		return self._aggregatedWeight
	
	def scaleWeights(self,factor):
		
		self._weights *= factor
		self._aggregatedWeight *= factor

# =========================================================================================================

class TargetTrackingParticleFilterWithDRNA(ParticleFilter):
	
	def __init__(self,nPEs,exchangePeriod,exchangeMap,c,epsilon,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors):
		
		super().__init__(nPEs*nParticlesPerPE,resamplingAlgorithm,resamplingCriterion)
		
		self._nPEs = nPEs

		self._nParticlesPerPE = nParticlesPerPE

		# the particle filters are built
		self._PEs = [EmbeddedTargetTrackingParticleFilter(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,aggregatedWeight=1.0/nPEs) for i in range(nPEs)]
		
		# a exchange of particles among PEs will happen every...
		self._exchangePeriod = exchangePeriod
		
		# ...time instants, according to the map
		self._exchangeMap = exchangeMap
		
		# period for the normalization of the aggregated weights
		self._normalizationPeriod = normalizationPeriod
		
		# parameter for checking the aggregated weights degeneration
		self._aggregatedWeightsUpperBound = c/math.pow(nPEs,1.0-epsilon)
		
		# because of the exchange and normalization periods, we must keep track of the elapsed (discreet) time instants
		self._n = 0
		
	def getAggregatedWeightsUpperBound(self):
		
		return self._aggregatedWeightsUpperBound
	
	def initialize(self):
		
		super().initialize()
		
		# all the PFs are initialized
		for PE in self._PEs:
			
			PE.initialize()
			
	def step(self,observations):
		
		super().step(observations)
		
		# a step is taken in every PF (ideally, this would occur concurrently)
		for PE in self._PEs:
			
			PE.step(observations)
		
		# a new time instant has elapsed
		self._n += 1
		
		# if it is exchanging particles time...
		if self._n % self._exchangePeriod == 0:
			
			self.exchangeParticles()
			
			# after the exchange, the aggregated weight of every PE must be updated
			for PE in self._PEs:
				
				PE.updateAggregatedWeight()
		
		# in order to peform some checks...
		aggregatedWeightsSum = self.getAggregatedWeights().sum()
		
		# the aggregated weights must be normalized every now and then to avoid computer precision issues
		if self._n % self._normalizationPeriod == 0:
			
			# ...to scale all the weights within ALL the PEs
			for PE in self._PEs:
				
				PE.scaleWeights(1.0/aggregatedWeightsSum)
			
		# if the weights degenerate so that they don't satisfy the corresponding assumption...
		if self.degeneratedAggregatedWeights():
			
			print('aggregated weights degenerated...')
			print(self.getAggregatedWeights() / self.getAggregatedWeights().sum())
			
			# a few particles are exchanged
			self.exchangeParticles()
			
			print('still degenerated: ',self.degeneratedAggregatedWeights())
			print(self.getAggregatedWeights() / self.getAggregatedWeights().sum())
		
	def exchangeParticles(self):
		
		print('exchangeParticles: sum before = ',self.getAggregatedWeights().sum())

		# first, we compile all the particles that are going to be exchanged in an auxiliar variable
		aux = []
		for exchangeTuple in self._exchangeMap:
			aux.append([self._PEs[exchangeTuple[0]].getParticle(exchangeTuple[1]),self._PEs[exchangeTuple[2]].getParticle(exchangeTuple[3])])

		# afterwards, we loop through all the exchange tuples performing the real exchange
		for (exchangeTuple,particles) in zip(self._exchangeMap,aux):
			self._PEs[exchangeTuple[0]].setParticle(exchangeTuple[1],particles[1])
			self._PEs[exchangeTuple[2]].setParticle(exchangeTuple[3],particles[0])
		
		print('exchangeParticles: sum after = ',self.getAggregatedWeights().sum())
		
	def getState(self):
		
		# the state from every PE is gathered together
		return np.hstack([PE.getState() for PE in self._PEs])
	
	def getAggregatedWeights(self):
		
		return np.array([PE.getAggregatedWeight() for PE in self._PEs])
	
	def degeneratedAggregatedWeights(self):

		if self.getAggregatedWeights().sum()==0:
			
			print('aggregated weights add up to 0!!')
			
			import code
			code.interact(local=dict(globals(), **locals()))

		normalizedWeights = self.getAggregatedWeights() / self.getAggregatedWeights().sum()

		if normalizedWeights.max() > self._aggregatedWeightsUpperBound:
			
			return True
		
	def computeMean(self):
		
		# the aggregated weights are not necessarily normalized
		normalizedAggregatedWeights = self.getAggregatedWeights()/self.getAggregatedWeights().sum()
		
		#import code
		#code.interact(local=dict(globals(), **locals()))
		
		return np.multiply(np.hstack([PE.computeMean() for PE in self._PEs]),normalizedAggregatedWeights).sum(axis=1)[np.newaxis].T