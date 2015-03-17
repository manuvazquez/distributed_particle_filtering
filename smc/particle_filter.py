import numpy as np
import math
import abc

import state
import smc.estimator
import smc.exchange

# this is required (due to a bug?) for import rpy2
import readline

import rpy2.robjects as robjects

# in order to load an R package
from rpy2.robjects.packages import importr

# for automatic conversion from numpy arrays to R data types
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

class ParticleFilter(metaclass=abc.ABCMeta):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion):
		
		self._nParticles = nParticles
		
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
	
	@abc.abstractmethod
	def initialize(self):

		pass
	
	@abc.abstractmethod
	def step(self,observations):
		
		pass
	
	@abc.abstractmethod
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
		
		# initial samples...
		self._state = self._prior.sample(self._nParticles)
		
		# the weights are assigned equal probabilities
		self._weights.fill(self._aggregatedWeight/self._nParticles)
		
	def step(self,observations):
		
		assert len(observations) == len(self._sensors)
		
		# every particle is updated (previous state is not stored...)
		self._state = np.hstack(
			[self._stateTransitionKernel.nextState(self._state[:,i:i+1]) for i in range(self._nParticles)])
		
		# for each sensor, we compute the likelihood of EVERY particle (position)
		likelihoods = np.array([sensor.likelihood(obs,state.position(self._state)) for sensor,obs in zip(self._sensors,observations)])
		
		# for each particle, we compute the product of the likelihoods for all the sensors
		likelihoodsProduct = likelihoods.prod(axis=0)
		
		# the weights are updated
		self._weights *= likelihoodsProduct
		
		# the aggregated weight is kept up to date at all times
		self.updateAggregatedWeight()
		
		# whatever is required (it depends on the algorithm) to avoid weights degeneracy...
		self.avoidWeightDegeneracy()

	def getState(self):
		
		return self._state

	def resample(self,normalizedWeights):
		
		# we check whether a resampling step is actually needed or not
		if self._resamplingCriterion.isResamplingNeeded(normalizedWeights):
			
			try:
				# the resampling algorithm is used to decide which particles to keep
				iParticlesToBeKept = self._resamplingAlgorithm.getIndexes(normalizedWeights)
				
			except ValueError:
				
				print("CentralizedTargetTrackingParticleFilter:resample: this shouldn't have happened...")
				
				import code
				code.interact(local=dict(globals(), **locals()))
			
			# the above indexes are used to update the state
			self._state = self._state[:,iParticlesToBeKept]
			
			# note that if the weights have been normalized ("standard" centralized particle filter), then "self._aggregatedWeight" is equal to 1
			self._weights.fill(self._aggregatedWeight/self._nParticles)
		
	def getParticle(self,index):
		
		return (self._state[:,index:index+1].copy(),self._weights[index])
	
	def getSamplesAt(self,indexes):
		
		"""Obtain (just) the samples at certain given indexes.
		
		This yields a "view" of the data, rather than a copy.
		
		Parameters
		----------
		indexes: 1-D ndarray
			The indexes of the requested particles
			
		Returns
		-------
		samples: 2-D ndarray
			The selected samples
		"""
		
		return self._state[:,indexes]
	
	@property
	def samples(self):
		
		return self._state
	
	@samples.setter
	def samples(self,value):
		
		if value.shape==self._state.shape:
			
			self._state = value
			
		else:
			
			raise Exception('the number and/or dimensions of the samples are not equal to the current ones')
	
	def setParticle(self,index,particle):
		
		self._state[:,index:index+1] = particle[0]
		self._weights[index] = particle[1]
		
		# the sum of the weights might have changed...
		self.updateAggregatedWeight()

	def updateAggregatedWeight(self):
		
		# the aggregated weight is simply the sum of the non-normalized weights
		self._aggregatedWeight = self._weights.sum()

	def computeMean(self):
		
		# if all the weights in this PF/PE are zero...
		if self._aggregatedWeight==0:
			
			# ...then an all-zeros estimate is returned...though any should do since this estimate must contribute zero
			return np.zeros((state.nElements,1))
		
		normalizedWeights = self._weights / self._aggregatedWeight

		# element-wise multiplication of the state vectors and their correspondent weights...followed by addition => weighted mean
		return np.multiply(self._state,normalizedWeights).sum(axis=1)[np.newaxis].T

	# this methods encapsulates the parts within the code of "step" which are different in this class and its children
	def avoidWeightDegeneracy(self):
		
		# if all the weights are zero...
		if np.isclose(self._aggregatedWeight,0):
			
			# ...then normalization makes no sense and we just initialize the weights again
			self._weights.fill(1.0/self._nParticles)

		else:
		
			self._weights /= self._aggregatedWeight
			
		# we forced this above
		self._aggregatedWeight = 1.0
		
		# the normalized weights are used to resample
		self.resample(self._weights)
	
	@property
	def weights(self):
		
		return self._weights
	
	@weights.setter
	def weights(self,value):
		
		if self._weights.shape==value.shape:
			
			self._weights=value
			
		else:
			
			raise Exception('the number of weights does not match the number of particles')
			
# =========================================================================================================

class EmbeddedTargetTrackingParticleFilter(CentralizedTargetTrackingParticleFilter):
	
	def getAggregatedWeight(self):
		
		return self._aggregatedWeight
	
	def scaleWeights(self,factor):
		
		self._weights *= factor
		self._aggregatedWeight *= factor

	def avoidWeightDegeneracy(self):
		
		# if all the weights are zero...
		if np.isclose(self._aggregatedWeight,0):
			
			# ...there is nothing we can do
			return
		
		else:
			# the normalized weights are used to resample
			self.resample(self._weights/self._aggregatedWeight)

# =========================================================================================================

class DistributedTargetTrackingParticleFilter(ParticleFilter):
	
	def __init__(self,nPEs,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
			  PFsClass=CentralizedTargetTrackingParticleFilter,PFsInitialAggregatedWeight=1.0):
		
		super().__init__(nPEs*nParticlesPerPE,resamplingAlgorithm,resamplingCriterion)
		
		# it is handy to keep the number of PEs in a variable
		self._nPEs = nPEs
		
		# a list of lists, the first one containing the indices of the sensors "seen" by the first PE...and so on
		self._PEsSensorsConnections = PEsSensorsConnections
		
		# number of particles per Pe
		self._nParticlesPerPE = nParticlesPerPE
		
		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [PFsClass(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,
													[s for iSensor,s in enumerate(sensors) if iSensor in connections],
													aggregatedWeight=PFsInitialAggregatedWeight) for connections in  PEsSensorsConnections]

	def initialize(self):
		
		# all the PFs are initialized
		for PE in self._PEs:
			
			PE.initialize()

		# we keep track of the elapsed (discreet) time instants
		self._n = 0

	def step(self,observations):
		
		# a step is taken in every PF (ideally, this would occur concurrently)
		for iPe,PE in enumerate(self._PEs):
			
			# only the appropriate observations are passed to this PE
			# NOTE: it is assumed that the order in which the observations are passed is the same as that of the sensors when building the PF
			PE.step(observations[self._PEsSensorsConnections[iPe]])
			
		# a new time instant has elapsed
		self._n += 1
	
	def getState(self):
		
		# the state from every PE is gathered together
		return np.hstack([PE.getState() for PE in self._PEs])

	def computeMean(self):
	
		# the means from all the PEs are stacked (horizontally) in a single array
		jointMeans = np.hstack([PE.computeMean() for PE in self._PEs])
		
		return jointMeans.mean(axis=1)[:,np.newaxis]


class TargetTrackingParticleFilterWithDRNA(DistributedTargetTrackingParticleFilter):
	
	def __init__(self,exchangePeriod,topology,aggregatedWeightsUpperBound,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
			  PFsClass=EmbeddedTargetTrackingParticleFilter):
		
		super().__init__(topology.getNumberOfPEs(),nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
				   PFsClass=PFsClass,PFsInitialAggregatedWeight=1.0/topology.getNumberOfPEs())
		
		# a exchange of particles among PEs will happen every...
		self._exchangePeriod = exchangePeriod

		# period for the normalization of the aggregated weights
		self._normalizationPeriod = normalizationPeriod
		
		# how the PEs are interconnected
		self._topology = topology

		# we get a unique exchange map from this network
		self._exchangeMap,_ = self._topology.getExchangeTuples()

	def step(self,observations):
		
		super().step(observations)
		
		# if it is exchanging particles time
		if (self._n % self._exchangePeriod == 0):
			
			self.exchangeParticles()
			
			# after the exchange, the aggregated weight of every PE must be updated
			for PE in self._PEs:
				
				PE.updateAggregatedWeight()
		
		# needed to perform the normalization below
		aggregatedWeightsSum = self.getAggregatedWeights().sum()
		
		# if every aggregated weight is zero...
		if np.isclose(aggregatedWeightsSum,0):
			
			# ...we reinitialize the weights for all the particles of all the PEs
			self.resetWeights()
			
			# ...and skip the normalization code below
			return
		
		# the aggregated weights must be normalized every now and then to avoid computer precision issues
		if self._n % self._normalizationPeriod == 0:
			
			# ...to scale all the weights within ALL the PEs
			for PE in self._PEs:
				
				PE.scaleWeights(1.0/aggregatedWeightsSum)

	def exchangeParticles(self):

		## we generate a random exchange map
		#self._exchangeMap,_ = self._topology.getExchangeTuples()

		# first, we compile all the particles that are going to be exchanged in an auxiliar variable
		aux = []
		for exchangeTuple in self._exchangeMap:
			aux.append((self._PEs[exchangeTuple.iPE].getParticle(exchangeTuple.iParticleWithinPE),self._PEs[exchangeTuple.iNeighbour].getParticle(exchangeTuple.iParticleWithinNeighbour)))

		# afterwards, we loop through all the exchange tuples performing the real exchange
		for (exchangeTuple,particles) in zip(self._exchangeMap,aux):
			self._PEs[exchangeTuple.iPE].setParticle(exchangeTuple.iParticleWithinPE,particles[1])
			self._PEs[exchangeTuple.iNeighbour].setParticle(exchangeTuple.iParticleWithinNeighbour,particles[0])
	
	def computeMean(self):
		
		# the aggregated weights are not necessarily normalized
		normalizedAggregatedWeights = self.getAggregatedWeights()/self.getAggregatedWeights().sum()
		
		# notice that "computeMean" will return a numpy array the size of the state (rather than a scalar)
		return np.multiply(np.hstack([PE.computeMean() for PE in self._PEs]),normalizedAggregatedWeights).sum(axis=1)[np.newaxis].T
	
	def getAggregatedWeights(self):
		
		return np.array([PE.getAggregatedWeight() for PE in self._PEs])

	def resetWeights(self):
		
		"""It sets every weight of every PE to the same value.
		"""
		
		# every PE will be assigned the same aggregated weight:
		aggregatedWeight = 1.0/self._nPEs
		
		# for every PE in this DPF...
		for PE in self._PEs:
			
			# the aggregated weight is set...
			PE._aggregatedWeight = aggregatedWeight
			
			# ...along with the individual weights within the PE
			PE.weights = np.full(PE._nParticles,aggregatedWeight/PE._nParticles)

class DistributedTargetTrackingParticleFilterWithMposterior(DistributedTargetTrackingParticleFilter):
	
	def __init__(self,topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,findWeiszfeldMedianParameters,estimator=smc.estimator.Mposterior(),
			  PFsClass=CentralizedTargetTrackingParticleFilter):
		
		super().__init__(topology.getNumberOfPEs(),nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,PFsClass=PFsClass)
		
		# how the PEs are interconnected
		self._topology = topology

		# the (R) Mposterior package is imported...
		self._Mposterior = importr('Mposterior')
		
		# ...and the parameters to be passed to the required function are kept
		self._findWeiszfeldMedianParameters = findWeiszfeldMedianParameters
		
		# estimator used to combine the distributions of the different PEs
		self._estimator = estimator
	
	def Mposterior(self,posteriorDistributions):
		
		"""Applies the Mposterior algorithm to weight the samples of a list of "subset posterior distribution"s.
		
		Parameters
		----------
		posteriorDistributions: list of tuples
			A list in which each element is a tuple representing a "subset posterior distribution": the first element are the samples, and the second the associated weights
		
		Returns
		-------
		samples: tuple
			The first element is a 2-D ndarray with all the samples, and the second the corresponding weights.
		"""
		
		# the samples of all the "subset posterior distribution"s are extracted
		samples = [posterior[0] for posterior in posteriorDistributions]
		
		# R function implementing the "M posterior" algorithm is called
		weiszfeldMedian = self._Mposterior.findWeiszfeldMedian(samples,**self._findWeiszfeldMedianParameters)

		# the weights assigned by the algorithm to each "subset posterior distribution"
		weiszfeldWeights = np.array(weiszfeldMedian[1])
		
		# a numpy array containing all the particles (coming from all the PEs)
		jointParticles = np.array(weiszfeldMedian[3]).T
		
		# the weight of each PE is scaled according to the "weiszfeldWeights" and, all of them are stacked together
		jointWeights =	np.hstack([posterior[1]*weight for posterior,weight in zip(posteriorDistributions,weiszfeldWeights)])
		
		return (jointParticles,jointWeights)

	def computeMean(self):
		
		return self._estimator.estimate(self)

class DistributedTargetTrackingParticleFilterWithParticleExchangingMposterior(DistributedTargetTrackingParticleFilterWithMposterior):
	
	def __init__(self,topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,findWeiszfeldMedianParameters,sharingPeriod,
			  estimator=smc.estimator.Mposterior(),exchangeManager=smc.exchange.RandomExchange(),PFsClass=CentralizedTargetTrackingParticleFilter):
		
		super().__init__(topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,findWeiszfeldMedianParameters,estimator=estimator,PFsClass=PFsClass)
		
		self._sharingPeriod = sharingPeriod
		self._nSharedParticles = topology.nParticlesExchangedBetweenTwoNeighbours
		
		# we get a unique exchange map from this network
		self._exchangeMap,self._neighboursWithParticles = self._topology.getExchangeTuples()
		
		# this object is responsible for the sharing step
		self._exchangeManager = exchangeManager
		
	def step(self,observations):
		
		super().step(observations)
		
		# if it is sharing particles time
		if (self._n % self._sharingPeriod == 0):
			
			self._exchangeManager.share(self)