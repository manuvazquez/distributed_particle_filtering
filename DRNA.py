import ParticleFilter

class ParticleFiltersCompoundWithDRNA:
	
	def __init__(self,nPEs,exchangePeriod,nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors):
		
		self._nPEs = nPEs
		self._nParticlesPerPE = nParticlesPerPE

		# for the sake of convenience...
		self._PEsrange = range(nPEs)

		# the particle filters are built
		self._PEs = [ParticleFilter.TrackingParticleFilter(nParticlesPerPE,resamplingAlgorithm,resamplingCriterion,prior,stateTransitionKernel,sensors,1.0/nPEs) for i in self._PEsrange]
		
		# a exchange of particles among PEs will happen every
		self._exchangePeriod = exchangePeriod
		
		# because of the exchange period, we must keep track of the elapsed (discreet) time instants
		self._n = 0
		
	def initialize(self):
		
		# all the PFs are initialized
		for i in self._PEsrange:
			
			self._PEs[i].initialize()
			
	def step(self,observations):
		
		# a step is taken in every PF (ideally, this would occur concurrently)
		for i in self._PEsrange:
			
			self._PEs[i].step(observations)
		
		# a new time instant has elapsed
		self._n += 1