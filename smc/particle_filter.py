import numpy as np
import math

import state

# this is required (due to a bug?) for import rpy2
import readline

import rpy2.robjects as robjects

# in order to load an R package
from rpy2.robjects.packages import importr

# for automatic conversion from numpy arrays to R data types
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

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
			
			# ...then we return an all-zeros estimate, though any should do since this estimate must contribute zero
			return np.zeros((self._state.shape[0],1))
		
		normalizedWeights = self._weights / self._aggregatedWeight

		# element-wise multiplication of the state vectors and their correspondent weights...followed by addition => weighted mean
		return np.multiply(self._state,normalizedWeights).sum(axis=1)[np.newaxis].T

	# this methods encapsulates the parts within the code of "step" which are different in this class and its children
	def avoidWeightDegeneracy(self):
		
		# if all the weights are zero...
		if self._aggregatedWeight==0:
			
			# ...then normalization makes no sense and we just initialize the weights again
			self._weights.fill(1.0/self._nParticles)

		else:
		
			self._weights /= self._aggregatedWeight
			
		# we forced this above
		self._aggregatedWeight = 1.0
		
		# the normalized weights are used to resample
		self.resample(self._weights)
	
	
	def getWeights(self):
		
		return self._weights
# =========================================================================================================

class EmbeddedTargetTrackingParticleFilter(CentralizedTargetTrackingParticleFilter):
	
	def getAggregatedWeight(self):
		
		return self._aggregatedWeight
	
	def scaleWeights(self,factor):
		
		self._weights *= factor
		self._aggregatedWeight *= factor

	def avoidWeightDegeneracy(self):
		
		# if all the weights are zero...
		if self._aggregatedWeight==0:
			
			# ...there is nothing we can do
			return
		
		else:
			# the normalized weights are used to resample
			self.resample(self._weights/self._aggregatedWeight)

# =========================================================================================================

class DistributedTargetTrackingParticleFilter(ParticleFilter):
	
	def __init__(self,topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
			  PFsClass=CentralizedTargetTrackingParticleFilter,PFsInitialAggregatedWeight=1.0):
		
		super().__init__(topology.getNumberOfPEs()*nParticlesPerPE,resamplingAlgorithm,resamplingCriterion)
		
		# it is handy to keep the number of PEs in a variable
		self._nPEs = topology.getNumberOfPEs()
		
		# a list of lists, the first one containing the indices of the sensors "seen" by the first PE...and so on
		self._PEsSensorsConnections = PEsSensorsConnections
		
		# number of particles per Pe
		self._nParticlesPerPE = nParticlesPerPE
		
		# the particle filters are built (each one associated with a different set of sensors)
		self._PEs = [PFsClass(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,
													[s for iSensor,s in enumerate(sensors) if iSensor in PEsSensorsConnections[iPe]],
													aggregatedWeight=PFsInitialAggregatedWeight) for iPe in range(self._nPEs)]
		
		# ...time instants, according to the geometry of the network
		self._topology = topology
	
	def initialize(self):
		
		super().initialize()
		
		# all the PFs are initialized
		for PE in self._PEs:
			
			PE.initialize()

		# because of the normalization period, we must keep track of the elapsed (discreet) time instants
		self._n = 0

	def step(self,observations):
		
		super().step(observations)
		
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


class TargetTrackingParticleFilterWithDRNA(DistributedTargetTrackingParticleFilter):
	
	def __init__(self,exchangePeriod,topology,aggregatedWeightsUpperBound,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
			  PFsClass=EmbeddedTargetTrackingParticleFilter):
		
		super().__init__(topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,PFsClass=PFsClass,PFsInitialAggregatedWeight=1.0/topology.getNumberOfPEs())
		
		# a exchange of particles among PEs will happen every...
		self._exchangePeriod = exchangePeriod

		# useful for checking how well the algorithm is doing
		self._aggregatedWeightsUpperBound = aggregatedWeightsUpperBound
		
		# period for the normalization of the aggregated weights
		self._normalizationPeriod = normalizationPeriod

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
				
			if self.degeneratedAggregatedWeights():
				print('after exchanging, aggregated weights are still degenerated => assumption 4 is not being satisfied!!')
				print(self.getAggregatedWeights() / self.getAggregatedWeights().sum())
		
		# in order to peform some checks...
		aggregatedWeightsSum = self.getAggregatedWeights().sum()
		
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
	
	def degeneratedAggregatedWeights(self):

		if self.getAggregatedWeights().sum()==0:
			
			print('aggregated weights add up to 0!!')
			
			import code
			code.interact(local=dict(globals(), **locals()))

		normalizedWeights = self.getAggregatedWeights() / self.getAggregatedWeights().sum()

		if normalizedWeights.max() > self._aggregatedWeightsUpperBound:
			
			return True
		
		else:
		
			return False

class ActivationsAwareEmbeddedTargetTrackingParticleFilter(EmbeddedTargetTrackingParticleFilter):
	
	def __init__(self,nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,aggregatedWeight=1.0,function=lambda x:2*x+1):
		
		super().__init__(nParticles,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,aggregatedWeight)
		
		# attribute meant to be accessed
		self.function = function
		
	def step(self,observations):
		
		# the number of active sensors is taken into account in the weights
		self._weights *= self.function(observations.sum())
		
		# notice that the method from the superclass is called AFTER updating the weights (this is because some actions in the superclass' method depend on the final weights) 
		super().step(observations)

class ActivationsAwareTargetTrackingParticleFilterWithDRNA(TargetTrackingParticleFilterWithDRNA):
	
	# here the default PF class is "ActivationsAwareEmbeddedTargetTrackingParticleFilter"
	def __init__(self,exchangePeriod,topology,aggregatedWeightsUpperBound,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,function,
			  PFsClass=ActivationsAwareEmbeddedTargetTrackingParticleFilter):
		
		# let the superclass do the hard work
		super().__init__(exchangePeriod,topology,aggregatedWeightsUpperBound,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,PFsClass)
		
		# the given function is set to be used by all the PEs
		for PE in self._PEs:
			
			PE.function = function

class RememberingNumberOfActiveSensorsEmbeddedTargetTrackingParticleFilter(EmbeddedTargetTrackingParticleFilter):
	
	def step(self,observations):
		
		# the number of active observations "seen" by this PE is remembered for later use...
		self.nActiveObservations = observations.sum()
		
		# ...and everything else is the same
		super().step(observations)
		
class OnlyPEsWithMaxActiveSensorsTargetTrackingParticleFilterWithDRNA(TargetTrackingParticleFilterWithDRNA):
	
	# here the default PF class is "RememberingNumberOfActiveSensorsEmbeddedTargetTrackingParticleFilter"
	def __init__(self,exchangePeriod,topology,aggregatedWeightsUpperBound,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
			  PFsClass=RememberingNumberOfActiveSensorsEmbeddedTargetTrackingParticleFilter):
		
		# let the superclass do the hard work
		super().__init__(exchangePeriod,topology,aggregatedWeightsUpperBound,nParticlesPerPE,normalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,PFsClass)
		
	def computeMean(self):
		
		# the number of 1's that each PE "has seen"
		nActiveObservations = np.array([p.nActiveObservations for p in self._PEs])
		
		# we focus on the PE (or PEs) with the maximum number of active sensors
		iRelevantPEs = np.where(nActiveObservations==np.max(nActiveObservations))[0]
		
		# normalization constant for the aggregated weights of the PEs with active sensors
		normConstant = self.getAggregatedWeights()[iRelevantPEs].sum()
		
		# if the above aggregated weights cannot be normalized...
		# NOTE: array "normalizedAggregatedWeights", initialized below, will only contain the normalized weight of the SELECTED (relevant) PEs
		if np.isclose(normConstant,0.0):
			
			# it is assumed that all the PEs are equally "reliable"
			normalizedAggregatedWeights = np.ones_like(iRelevantPEs)/len(iRelevantPEs)
		
		# ...otherwise
		else:
			# ...they are normalized
			normalizedAggregatedWeights = self.getAggregatedWeights()[iRelevantPEs]/normConstant
		
		#if np.isnan(np.multiply(np.hstack([self._PEs[iPE].computeMean() for iPE in iRelevantPEs]),normalizedAggregatedWeights).sum(axis=1)[np.newaxis].T).any():
			
			#import code
			#code.interact(local=dict(globals(), **locals()))
		
		return np.multiply(np.hstack([self._PEs[iPE].computeMean() for iPE in iRelevantPEs]),normalizedAggregatedWeights).sum(axis=1)[np.newaxis].T


class DistributedTargetTrackingParticleFilterWithMposterior(DistributedTargetTrackingParticleFilter):
	
	def __init__(self,topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,
			  PFsClass=CentralizedTargetTrackingParticleFilter):
		
		super().__init__(topology,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,PEsSensorsConnections,PFsClass=PFsClass)

		# the (R) Mposterior package is imported
		self._Mposterior = importr('Mposterior')

	def computeMean(self):
		
		# a list containing the "subset posterior distribution"s; each one is a matrix (numpy array) containing the samples of a PE
		PEsParticles = [PE.getState().T for PE in self._PEs]
		
		#for i,PE in enumerate(PEsParticles):
			#np.savetxt('PE{}.txt'.format(i),PE,delimiter=',')
		
		#import code
		#code.interact(local=dict(globals(), **locals()))
		
		# R function implementing the "M posterior" algorithm is called
		weiszfeldMedian = self._Mposterior.findWeiszfeldMedian(PEsParticles, sigma = 0.1, maxit = 100, tol = 1e-10)

		# the weights assigned by the algorithm to each "subset posterior distribution"
		weiszfeldWeights = np.array(weiszfeldMedian[1])
		
		# a numpy array containing all the particles (coming from all the PEs)
		jointParticles = np.array(weiszfeldMedian[3]).T
		
		# the weight of each PE are scaled according to the "weiszfeldWeights" and, all of them are stacked together
		jointWeights =	np.hstack([PE.getWeights()*weight for PE,weight in zip(self._PEs,weiszfeldWeights)])
		
		return np.multiply(jointParticles,jointWeights).sum(axis=1)[np.newaxis].T