import abc

import numpy as np

class SharingManager(metaclass=abc.ABCMeta):
	
	@abc.abstractmethod
	def share(self,DPF):
		
		pass

class DeterministicExchange(SharingManager):
	
	def share(self,DPF):
		
		# a list of lists, one per PE, each one containing tuples whose first element is the index of the neighbour, and whose second element is a list with the indexes of the particles shared by that neighbour
		_,overall_neighbours_particles = DPF._topology.getExchangeTuples()
		
		for PE,this_PE_neighbours_particles in zip(DPF._PEs,overall_neighbours_particles):
			
			# a list with the subset posterior of each neighbour
			subsetPosteriorDistributions = [(DPF._PEs[neighbour_particles[0]].getSamplesAt(neighbour_particles[1]).T,np.full(DPF._nSharedParticles,1.0/DPF._nSharedParticles)) for neighbour_particles in this_PE_neighbours_particles]
			
			# a subset posterior obtained from this PE is also added: it encompasses its FIRST "DPF._nSharedParticles" particles
			subsetPosteriorDistributions.append((PE.getSamplesAt(range(DPF._nSharedParticles)).T,np.full(DPF._nSharedParticles,1.0/DPF._nSharedParticles)))
			
			# M posterior on the posterior distributions collected above
			jointParticles,jointWeights = DPF.Mposterior(subsetPosteriorDistributions)
			
			# the indexes of the particles to be kept
			iNewParticles = DPF._resamplingAlgorithm.getIndexes(jointWeights,PE._nParticles)
			
			PE.samples = jointParticles[:,iNewParticles]
			PE.logWeights = np.full(PE._nParticles,-np.log(PE._nParticles))
			PE.updateAggregatedWeight()

class RandomExchange(SharingManager):
	
	def share(self,DPF):
		
		# each PE draws a set of samples from its probability measure...to be shared with its neighbours
		samplesToBeShared = [PE.getSamplesAt(DPF._resamplingAlgorithm.getIndexes(np.exp(PE.logWeights),DPF._nSharedParticles)) for PE in DPF._PEs]
		
		# the list of neighbours of each PE
		PEsNeighbours = DPF._topology.getNeighbours()
		
		# for every PE...
		for PE,neighbours,thisPEsamples in zip(DPF._PEs,PEsNeighbours,samplesToBeShared):
			
			# ...the particles shared by its neighbours (assumed to be uniformly distributed) are gathered...
			subsetPosteriorDistributions = [(samplesToBeShared[i].T,np.full(DPF._nSharedParticles,1.0/DPF._nSharedParticles)) for i in neighbours]
			
			# ...along with its own (shared, already sampled) particles
			subsetPosteriorDistributions.append((thisPEsamples.T,np.full(DPF._nSharedParticles,1.0/DPF._nSharedParticles)))
			
			# M posterior on the posterior distributions collected above
			jointParticles,jointWeights = DPF.Mposterior(subsetPosteriorDistributions)
			
			# the indexes of the particles to be kept
			iNewParticles = DPF._resamplingAlgorithm.getIndexes(jointWeights,PE._nParticles)
			
			PE.samples = jointParticles[:,iNewParticles]
			PE.logWeights = np.full(PE._nParticles,-np.log(PE._nParticles))
			PE.updateAggregatedWeight()