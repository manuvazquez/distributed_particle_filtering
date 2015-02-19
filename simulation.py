import abc
import numpy as np
import scipy.io

from smc import particle_filter
import topology
import drnautil
import sensors_PEs_connector
import state
import plot

class Simulation(metaclass=abc.ABCMeta):
	
	@abc.abstractmethod
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs):
		
		# number of particles per processing element (PE)
		self._K = parameters["number of particles per PE"]
		
		# length of the trajectory
		self._nTimeInstants = parameters["number of time instants"]
		
		# name of the file to store the results
		self._outputFile = outputFile
		
		# DRNA related
		self._DRNAsettings = parameters["DRNA"]
		
		# parameters related to plotting
		self._painterSettings = parameters["painter"]
		
		# room  dimensions
		self._roomSettings = parameters["room"]
		
		# different setups for the PEs
		self._topologiesSettings = parameters['topologies']
		
		# sensors positions are gathered
		self._sensorsPositions = np.hstack([s.position for s in sensors])
		
		# so that it equals 0 the first time it is incremented...
		self._iFrame = -1
		
	@abc.abstractmethod
	def processFrame(self,targetPosition,targetVelocity,observations):
		
		self._iFrame += 1
	
	@abc.abstractmethod
	def saveData(self,targetPosition):

		if self._iFrame==0:
			print('saveData: nothing to save...skipping')
			return

class Convergence(Simulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs)
		
		topologies = [getattr(topology,t['class'])(t['number of PEs'],self._K,self._DRNAsettings["exchanged particles maximum percentage"],t['parameters'],
											 PRNG=PRNGs["topology pseudo random numbers generator"]) for t in self._topologiesSettings]
		
		# we compute the upper bound for the supremum of the aggregated weights that should guarante convergence
		self._aggregatedWeightsUpperBounds = [drnautil.supremumUpperBound(t['number of PEs'],self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon']) for t in self._topologiesSettings]
		
		# plain non-parallelized particle filter
		self._PFsForTopologies = [particle_filter.CentralizedTargetTrackingParticleFilter(self._K*t.getNumberOfPEs(),resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors) for t in topologies]
		
		sensorsPEsConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(sensors)

		# distributed particle filter
		self._distributedPFsForTopologies = [particle_filter.TargetTrackingParticleFilterWithDRNA(
			self._DRNAsettings["exchange period"],t,upperBound,self._K,self._DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
			sensors,sensorsPEsConnector.getConnections(t.getNumberOfPEs())
			) for t,upperBound in zip(topologies,self._aggregatedWeightsUpperBounds)]
		
		#------------------------------------------------------------- metrics initialization --------------------------------------------------------------------
		
		# we store the aggregated weights...
		self._distributedPFaggregatedWeights = [np.empty((self._nTimeInstants,t.getNumberOfPEs(),parameters["number of frames"])) for t in topologies]

		# ...and the position estimates
		self._centralizedPF_pos,self._distributedPF_pos = np.empty((2,self._nTimeInstants,parameters["number of frames"],len(topologies))),np.empty((2,self._nTimeInstants,parameters["number of frames"],len(topologies)))
		
	def saveData(self,targetPosition):
		
		# let the super class do its thing...
		super().saveData(targetPosition)
		
		# the mean of the MSE incurred by both PFs
		centralizedPF_MSE = (np.subtract(self._centralizedPF_pos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).mean(axis=0).mean(axis=1)
		distributedPF_MSE = (np.subtract(self._distributedPF_pos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).mean(axis=0).mean(axis=1)
		
		# ...the same for the error (euclidean distance)
		centralizedPF_error = np.sqrt((np.subtract(self._centralizedPF_pos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)
		distributedPF_error = np.sqrt((np.subtract(self._distributedPF_pos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)

		# MSE vs time (only the results for the first topology are plotted)
		plot.distributedPFagainstCentralizedPF(np.arange(self._nTimeInstants),centralizedPF_MSE[:,0],distributedPF_MSE[:,0],
							self._painterSettings["file name prefix for the MSE vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
							centralizedPFparameters={'label':'Centralized PF','color':self._painterSettings["color for the centralized PF"],'marker':self._painterSettings["marker for the centralized PF"]},
							distributedPFparameters={'label':'Distributed PF','color':self._painterSettings["color for the distributed PF"],'marker':self._painterSettings["marker for the distributed PF"]},
							figureId='MSE vs Time')

		# distance vs time (only the results for the first topology are plotted)
		plot.distributedPFagainstCentralizedPF(np.arange(self._nTimeInstants),centralizedPF_error[:,0],distributedPF_error[:,0],
							self._painterSettings["file name prefix for the euclidean distance vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
							centralizedPFparameters={'label':'Centralized PF','color':self._painterSettings["color for the centralized PF"],'marker':self._painterSettings["marker for the centralized PF"]},
							distributedPFparameters={'label':'Distributed PF','color':self._painterSettings["color for the distributed PF"],'marker':self._painterSettings["marker for the distributed PF"]},
							figureId='Euclidean distance vs Time')

		# the aggregated weights are normalized at ALL TIMES, for EVERY frame and EVERY topology
		normalizedAggregatedWeights = [np.rollaxis(np.divide(np.rollaxis(w[:,:,:self._iFrame],2,1),w[:,:,:self._iFrame].sum(axis=1)[:,:,np.newaxis]),2,1) for w in self._distributedPFaggregatedWeights]
		
		# ...the same data structured in a dictionary
		normalizedAggregatedWeightsDic = {'normalizedAggregatedWeights_{}'.format(i):array for i,array in enumerate(normalizedAggregatedWeights)}
		
		# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
		maxWeights = np.array([(w.max(axis=1)**self._DRNAsettings['q']).mean(axis=1) for w in normalizedAggregatedWeights])

		# evolution of the largest aggregated weight over time (only the results for the first topology are plotted)
		plot.aggregatedWeightsSupremumVsTime(maxWeights[0,:],self._aggregatedWeightsUpperBounds[0],
												self._painterSettings["file name prefix for the aggregated weights supremum vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),self._DRNAsettings["exchange period"])

		# if requested, save the trajectory
		if self._painterSettings["display evolution?"]:
			if 'iTime' in globals() and iTime>0:
				painter.save(('trajectory_up_to_iTime={}_' + hostname + '_' + date + '.eps').format(repr(iTime)))

		# a dictionary encompassing all the data to be saved
		dataToBeSaved = dict(
				aggregatedWeightsUpperBounds = self._aggregatedWeightsUpperBounds,
				targetPosition = targetPosition[:,:,:self._iFrame],
				centralizedPF_pos = self._centralizedPF_pos[:,:,:self._iFrame,:],
				distributedPF_pos = self._distributedPF_pos[:,:,:self._iFrame,:],
				**normalizedAggregatedWeightsDic
			)
		
		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**dataToBeSaved)
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))
	
	def processFrame(self,targetPosition,targetVelocity,observations):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity,observations)
		
		for iTopology,(pf,distributedPf) in enumerate(zip(self._PFsForTopologies,self._distributedPFsForTopologies)):
			
			# initialization of the particle filters
			pf.initialize()
			distributedPf.initialize()

			if self._painterSettings['display evolution?']:
				
				# if this is not the first iteration...
				if hasattr(self,'_painter'):
					
					# ...then, the previous figure is closed
					self._painter.close()

				# this object will handle graphics...
				self._painter = plot.RectangularRoomPainter(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],self._sensorsPositions,sleepTime=self._painterSettings["sleep time between updates"])

				# ...e.g., draw the sensors
				self._painter.setup()

			for iTime in range(self._nTimeInstants):

				print('---------- iFrame = {}, iTopology = {}, iTime = {}'.format(repr(self._iFrame),repr(iTopology),repr(iTime)))

				print('position:\n',targetPosition[:,iTime:iTime+1])
				print('velocity:\n',targetVelocity[:,iTime:iTime+1])
				
				# particle filters are updated
				pf.step(observations[iTime])
				distributedPf.step(observations[iTime])
				
				# the mean computed by the centralized and distributed PFs
				centralizedPF_mean,distributedPF_mean = pf.computeMean(),distributedPf.computeMean()
				
				self._centralizedPF_pos[:,iTime:iTime+1,self._iFrame,iTopology],self._distributedPF_pos[:,iTime:iTime+1,self._iFrame,iTopology] = state.position(centralizedPF_mean),state.position(distributedPF_mean)
				
				# the aggregated weights of the different PEs in the distributed PF are stored
				self._distributedPFaggregatedWeights[iTopology][iTime,:,self._iFrame] = distributedPf.getAggregatedWeights()
				
				print('centralized PF\n',centralizedPF_mean)
				print('distributed PF\n',distributedPF_mean)
				
				if self._painterSettings["display evolution?"]:

					# the plot is updated with the position of the target...
					self._painter.updateTargetPosition(targetPosition[:,iTime:iTime+1])
					
					# ...those estimated by the PFs
					self._painter.updateEstimatedPosition(state.position(centralizedPF_mean),identifier='centralized',color=self._painterSettings["color for the centralized PF"])
					self._painter.updateEstimatedPosition(state.position(distributedPF_mean),identifier='distributed',color=self._painterSettings["color for the distributed PF"])

					if self._painterSettings["display particles evolution?"]:

						# ...and those of the particles...
						self._painter.updateParticlesPositions(state.position(pf.getState()),identifier='centralized',color=self._painterSettings["color for the centralized PF"])
						self._painter.updateParticlesPositions(state.position(distributedPf.getState()),identifier='distributed',color=self._painterSettings["color for the distributed PF"])

class ActiveSensors(Simulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs)
		
		self._simulationParameters = parameters['ActiveSensors']
		
		# the FIRST topology in the parameters file is selected
		selectedTopologySettings = self._topologiesSettings[0]
		selectedTopology = getattr(topology,selectedTopologySettings['class'])(selectedTopologySettings['number of PEs'],self._K,self._DRNAsettings["exchanged particles maximum percentage"],selectedTopologySettings['parameters'],
											 PRNG=PRNGs["topology pseudo random numbers generator"])
		
		# plain (centralized) particle filter
		self._PFs = [particle_filter.CentralizedTargetTrackingParticleFilter(self._K*selectedTopology.getNumberOfPEs(),resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)]
		
		# the color and label for the centralized PF are added to the corresponding lists
		self._PFsColors = ['black']
		self._PFsLabels = ['Centralized']
		
		aggregatedWeightsUpperBound = drnautil.supremumUpperBound(selectedTopologySettings['number of PEs'],self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon'])
		
		# a dictionary with the parameters of the selected sensors-PEs connector...
		sensorsPEsConnectorParameters = parameters['sensors-PEs connectors'][parameters['sensors-PEs connectors']['type']]
		
		# ...containing information about which class to instantiate along with its specific parameters
		sensorsPEsConnector = getattr(sensors_PEs_connector,sensorsPEsConnectorParameters['implementing class'])(sensors,sensorsPEsConnectorParameters['parameters'])
		
		# a distributed PF with partial observations, in which the estimates are computed using only the PEs with the higher number of active sensors
		self._PFs.append(
			particle_filter.OnlyPEsWithMaxActiveSensorsTargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],selectedTopology,aggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,
				prior,transitionKernel,sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs())
			)
		)
		self._PFsColors.append('cyan')
		self._PFsLabels.append('DRNA (PEs with maximum #active sensors)')
		
		# a distributed PF is added to the list for each of the functions defined in the parameters file
		for f in self._simulationParameters['functions of #1s']:
			
			self._PFs.append(
				particle_filter.ActivationsAwareTargetTrackingParticleFilterWithDRNA(
					self._DRNAsettings["exchange period"],selectedTopology,aggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,
					prior,transitionKernel,sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()),eval('lambda x:' + f['definition'])
				)
			)
			
			self._PFsColors.append(f['color when plotting'])
			self._PFsLabels.append(f['label'])
		
		# the position estimates
		self._PFs_pos = np.empty((2,self._nTimeInstants,parameters["number of frames"],len(self._PFs)))
		
		assert len(self._PFsColors) == len(self._PFsLabels) == len(self._PFs)
		
	def saveData(self,targetPosition):
		
		# let the super class do its thing...
		super().saveData(targetPosition)
		
		# the mean of the error (euclidean distance) incurred by the PFs
		PF_error = np.sqrt((np.subtract(self._PFs_pos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)
		
		# a dictionary encompassing all the data to be saved
		dataToBeSaved = dict(
				targetPosition = targetPosition[:,:,:self._iFrame],
				PF_pos = self._PFs_pos[:,:,:self._iFrame,:]
			)
		
		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**dataToBeSaved)
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))
		
		plot.PFs(range(self._nTimeInstants),PF_error,
		   self._simulationParameters["file name prefix for the estimation error vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
			[{'label':l,'color':c} for l,c in zip(self._PFsLabels,self._PFsColors)])
		
	def processFrame(self,targetPosition,targetVelocity,observations):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity,observations)
		
		for pf in self._PFs:
			
			# initialization of the particle filters
			pf.initialize()
		
		if self._painterSettings['display evolution?']:
			
			# if this is not the first iteration...
			if hasattr(self,'_painter'):
				
				# ...then, the previous figure is closed
				self._painter.close()

			# this object will handle graphics...
			self._painter = plot.RectangularRoomPainter(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],self._sensorsPositions,sleepTime=self._painterSettings["sleep time between updates"])

			# ...e.g., draw the sensors
			self._painter.setup()

		for iTime in range(self._nTimeInstants):

			print('---------- iFrame = {}, iTime = {}'.format(repr(self._iFrame),repr(iTime)))

			print('position:\n',targetPosition[:,iTime:iTime+1])
			print('velocity:\n',targetVelocity[:,iTime:iTime+1])
			
			# particle filters are updated
			for iPF,(pf,label) in enumerate(zip(self._PFs,self._PFsLabels)):
				
				# initialization of the particle filters
				pf.step(observations[iTime])
				
				self._PFs_pos[:,iTime:iTime+1,self._iFrame,iPF] = state.position(pf.computeMean())
				
				print('position estimated by {}\n'.format(label),self._PFs_pos[:,iTime:iTime+1,self._iFrame,iPF])
			
			if self._painterSettings["display evolution?"]:

				# the plot is updated with the position of the target...
				self._painter.updateTargetPosition(targetPosition[:,iTime:iTime+1])
				
				# ...those estimated by the PFs
				for iPF,(pf,color) in enumerate(zip(self._PFs,self._PFsColors)):
					
					self._painter.updateEstimatedPosition(self._PFs_pos[:,iTime:iTime+1,self._iFrame,iPF],identifier='#{}'.format(iPF),color=color)
					
					if self._painterSettings["display particles evolution?"]:
						
						self._painter.updateParticlesPositions(state.position(pf.getState()),identifier='#{}'.format(iPF),color=color)

class Mposterior(Simulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs)
		
		self._simulationParameters = parameters['Mposterior']
		
		# the FIRST topology in the parameters file is selected
		selectedTopologySettings = self._topologiesSettings[0]
		selectedTopology = getattr(topology,selectedTopologySettings['class'])(selectedTopologySettings['number of PEs'],self._K,self._DRNAsettings["exchanged particles maximum percentage"],selectedTopologySettings['parameters'],
											 PRNG=PRNGs["topology pseudo random numbers generator"])
		
		# plain (centralized) particle filter
		self._PFs = [particle_filter.CentralizedTargetTrackingParticleFilter(self._K*selectedTopology.getNumberOfPEs(),resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)]
		
		# the color and label for the centralized PF are added to the corresponding lists
		self._PFsColors = ['black']
		self._PFsLabels = ['Centralized']
		
		# a dictionary with the parameters of the selected sensors-PEs connector...
		sensorsPEsConnectorParameters = parameters['sensors-PEs connectors'][parameters['sensors-PEs connectors']['type']]
		
		# ...containing information about which class to instantiate along with its specific parameters
		sensorsPEsConnector = getattr(sensors_PEs_connector,sensorsPEsConnectorParameters['implementing class'])(sensors,sensorsPEsConnectorParameters['parameters'])
		
		sensorsPositions = np.hstack([s.position for s in sensors])
		PEsPositions = sensorsPEsConnector.computePEsPositions(sensorsPositions,selectedTopology.getNumberOfPEs(),sensorsPEsConnectorParameters['parameters']['number of uniform samples']*selectedTopology.getNumberOfPEs())
		plot.PEsSensorsConnections(sensorsPositions,PEsPositions,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()))
		
		# several "isolated" PFs whose distributions are combined by means of the M-posterior algorithm
		self._PFs.append(
			particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
				selectedTopology,self._K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
				sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()),self._simulationParameters['findWeiszfeldMedian parameters'],
				PFsClass=particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		self._PFsColors.append('cyan')
		self._PFsLabels.append('M-posterior')
		
		aggregatedWeightsUpperBound = drnautil.supremumUpperBound(selectedTopologySettings['number of PEs'],self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon'])
		
		# a connector that connects every sensor to every PE
		everySensorWithEveryPEConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(sensors)
		
		# a distributed PF with DRNA
		self._PFs.append(
			particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],selectedTopology,aggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,
				prior,transitionKernel,sensors,everySensorWithEveryPEConnector.getConnections(selectedTopology.getNumberOfPEs())
			)
		)
		
		self._PFsColors.append('green')
		self._PFsLabels.append('DRNA')
		
		# a "distributed" PF in which each PE does its computation independently of the rest
		self._PFs.append(
			particle_filter.DistributedTargetTrackingParticleFilter(
				selectedTopology,self._K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
				sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()),PFsClass=particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		self._PFsColors.append('pink')
		self._PFsLabels.append('Plain DPF')
		
		# a "distributed" PF in which each PE does its computation independently of the rest...but every now and then, M posterior is used to combine distributions of neighbours
		self._PFs.append(
			particle_filter.DistributedTargetTrackingParticleFilterWithParticleExchangingMposterior(
				selectedTopology,self._K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
				sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()),self._simulationParameters['findWeiszfeldMedian parameters'],
				self._simulationParameters['sharing period'],self._simulationParameters['number of particles shared by each PE'],
				PFsClass=particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		self._PFsColors.append('brown')
		self._PFsLabels.append('M-posterior with exchange')
		
		# several "isolated" PFs whose distributions are combined by means of the M-posterior algorithm...but only a few particles from each PE are used
		self._PFs.append(
			particle_filter.DistributedTargetTrackingParticleFilterWithComplexityConstrainedMposterior(
				selectedTopology,self._K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
				sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()),self._simulationParameters['findWeiszfeldMedian parameters'],
				self._simulationParameters['number of particles from each PE for computing the final estimate'],
				PFsClass=particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		self._PFsColors.append('magenta')
		self._PFsLabels.append('Complexity-constrained M-posterior')
	

		# a "distributed" PF in which each PE does its computation independently of the rest...but every now and then, M posterior is used to combine distributions of neighbours
		self._PFs.append(
			particle_filter.DistributedTargetTrackingParticleFilterWithComplexityConstrainedParticleExchangingMposterior(
				selectedTopology,self._K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
				sensors,sensorsPEsConnector.getConnections(selectedTopology.getNumberOfPEs()),self._simulationParameters['findWeiszfeldMedian parameters'],
				self._simulationParameters['sharing period'],self._simulationParameters['number of particles shared by each PE'],
				self._simulationParameters['number of particles from each PE for computing the final estimate'],
				PFsClass=particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		self._PFsColors.append('gray')
		self._PFsLabels.append('Complexity-constrained M-posterior with exchange')

		# ---------------------
		
		# the position estimates
		self._PFs_pos = np.empty((2,self._nTimeInstants,parameters["number of frames"],len(self._PFs)))
		
		assert len(self._PFsColors) == len(self._PFsLabels) == len(self._PFs)
		
		# information about the simulated algorithms is added to the parameters...
		parameters['algorithms'] = [{'name':name,'color':color} for name,color in zip(self._PFsLabels,self._PFsColors)]
		
	def saveData(self,targetPosition):
		
		# let the super class do its thing...
		super().saveData(targetPosition)
		
		# the mean of the error (euclidean distance) incurred by the PFs
		PF_error = np.sqrt((np.subtract(self._PFs_pos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)
		
		# a dictionary encompassing all the data to be saved
		dataToBeSaved = dict(
				targetPosition = targetPosition[:,:,:self._iFrame],
				PF_pos = self._PFs_pos[:,:,:self._iFrame,:]
			)
		
		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**dataToBeSaved)
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))
		
		plot.PFs(range(self._nTimeInstants),PF_error,
		   self._simulationParameters["file name prefix for the estimation error vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
			[{'label':l,'color':c} for l,c in zip(self._PFsLabels,self._PFsColors)])
		
	def processFrame(self,targetPosition,targetVelocity,observations):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity,observations)
		
		for pf in self._PFs:
			
			# initialization of the particle filters
			pf.initialize()
		
		if self._painterSettings['display evolution?']:
			
			# if this is not the first iteration...
			if hasattr(self,'_painter'):
				
				# ...then, the previous figure is closed
				self._painter.close()

			# this object will handle graphics...
			self._painter = plot.RectangularRoomPainter(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],self._sensorsPositions,sleepTime=self._painterSettings["sleep time between updates"])

			# ...e.g., draw the sensors
			self._painter.setup()

		for iTime in range(self._nTimeInstants):

			print('---------- iFrame = {}, iTime = {}'.format(repr(self._iFrame),repr(iTime)))

			print('position:\n',targetPosition[:,iTime:iTime+1])
			print('velocity:\n',targetVelocity[:,iTime:iTime+1])
			
			# particle filters are updated
			for iPF,(pf,label) in enumerate(zip(self._PFs,self._PFsLabels)):
				
				# initialization of the particle filters
				pf.step(observations[iTime])
				
				self._PFs_pos[:,iTime:iTime+1,self._iFrame,iPF] = state.position(pf.computeMean())
				
				print('position estimated by {}\n'.format(label),self._PFs_pos[:,iTime:iTime+1,self._iFrame,iPF])
			
			if self._painterSettings["display evolution?"]:

				# the plot is updated with the position of the target...
				self._painter.updateTargetPosition(targetPosition[:,iTime:iTime+1])
				
				# ...those estimated by the PFs
				for iPF,(pf,color) in enumerate(zip(self._PFs,self._PFsColors)):
					
					self._painter.updateEstimatedPosition(self._PFs_pos[:,iTime:iTime+1,self._iFrame,iPF],identifier='#{}'.format(iPF),color=color)
					
					if self._painterSettings["display particles evolution?"]:
						
						self._painter.updateParticlesPositions(state.position(pf.getState()),identifier='#{}'.format(iPF),color=color)