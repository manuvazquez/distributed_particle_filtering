import abc
import numpy as np
import scipy.io
import math
import h5py

from smc import particle_filter
import smc.exchange_recipe
import smc.estimator
import PEs_topology
import drnautil
import sensor
import sensors_PEs_connector
import state
import plot
import network_nodes


class Simulation(metaclass=abc.ABCMeta):
	
	@abc.abstractmethod
	def __init__(self, parameters, resamplingAlgorithm, resamplingCriterion, prior, transitionKernel, outputFile, PRNGs):
		
		# these parameters are kept for later use
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		self._prior = prior
		self._transitionKernel = transitionKernel
		self._PRNGs = PRNGs
		
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
		
		# the settings for the topology or topologies given...if it is a list...
		if isinstance(parameters['topologies']['type'],list):
			# ...we have a list of settings
			self._topologiesSettings = [parameters['topologies'][i] for i in parameters['topologies']['type']]
		# otherwise...
		else:
			# the "topology settings" object is just a dictionary
			self._topologiesSettings = parameters['topologies'][parameters['topologies']['type']]
		
		# so that it equals 0 the first time it is incremented...
		self._iFrame = -1
		
		# the parameters for this particular simulation are obtained
		self._simulationParameters = parameters['simulations'][parameters['simulations']['type']]
		
	@abc.abstractmethod
	def processFrame(self,targetPosition,targetVelocity):
		
		self._iFrame += 1
	
	# TODO: remove targetPosition as argument?
	
	@abc.abstractmethod
	def saveData(self,targetPosition):

		if self._iFrame==0:
			print('saveData: nothing to save...skipping')
			return

class SimpleSimulation(Simulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs,nPEs=None,nSensors=None):
		
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs)
		
		# for the sake of convenience
		sensorsSettings = parameters["sensors"]

		if nSensors==None:
			
			nSensors = parameters["sensors"]['number']
		
		if nPEs==None:

			# we try to extract the "number of PEs" from the topology settings...
			try:
				
				nPEs = self._topologiesSettings['number of PEs']
			
			# ...it it's not possible, it is because there are multiple topology settings => Convergence simulation
			except TypeError:
				
				# nPEs = None
				pass
		
		# for the sake of convenience below...
		networkNodesSettings = parameters['network nodes'][self._simulationParameters['network']]
		networkNodesClass = getattr(network_nodes, networkNodesSettings['implementing class'])
		
		# the appropriate class is instantiated with the given parameters
		self._network = networkNodesClass(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],nPEs,nSensors,**networkNodesSettings['parameters'])
		
		# the positions of the PEs and the sensors are collected from the network just built
		self._sensorsPositions,self._PEsPositions = self._network.sensorsPositions,self._network.PEsPositions
		
		# the class to be instantiated is figured out from the settings for that particular sensor type
		sensorClass = getattr(sensor,sensorsSettings[sensorsSettings['type']]['implementing class'])
		
		self._sensors = [sensorClass(pos[:,np.newaxis],PRNG=PRNGs['Sensors and Monte Carlo pseudo random numbers generator'],**sensorsSettings[sensorsSettings['type']]['parameters']) for pos in self._sensorsPositions.T]
		
	def processFrame(self,targetPosition,targetVelocity):
		
		super().processFrame(targetPosition,targetVelocity)
		
		# observations for all the sensors at every time instant (each list)
		# NOTE: conversion to float is done so that the observations (either 1 or 0) are amenable to be used in later computations
		self._observations = [np.array([sensor.detect(state.position(s[:,np.newaxis])) for sensor in self._sensors],dtype=float) for s in targetPosition.T]

class Convergence(SimpleSimulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs)
		
		topologies = [getattr(PEs_topology,t['implementing class'])(t['number of PEs'],t['parameters']) for t in self._topologiesSettings]
		
		exchangeRecipes = [smc.exchange_recipe.DRNAExchangeRecipe(t,self._K,self._simulationParameters["exchanged particles"],PRNG=self._PRNGs["topology pseudo random numbers generator"]) for t in topologies]
		
		# we compute the upper bound for the supremum of the aggregated weights that should guarante convergence
		self._aggregatedWeightsUpperBounds = [drnautil.supremumUpperBound(t['number of PEs'],self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon']) for t in self._topologiesSettings]

		# plain non-parallelized particle filter
		self._PFsForTopologies = [particle_filter.CentralizedTargetTrackingParticleFilter(self._K*t.getNumberOfPEs(),resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,self._sensors) for t in topologies]
		
		sensorsPEsConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensorsPositions)

		# distributed particle filter
		self._distributedPFsForTopologies = [particle_filter.TargetTrackingParticleFilterWithDRNA(
			self._DRNAsettings["exchange period"],e,self._K,self._DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,
			self._sensors,sensorsPEsConnector.getConnections(e.getNumberOfPEs())
			) for e in exchangeRecipes]
		
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
							outputFile=self._painterSettings["file name prefix for the MSE vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
							centralizedPFparameters={'label':'Centralized PF','color':self._painterSettings["color for the centralized PF"],'marker':self._painterSettings["marker for the centralized PF"]},
							distributedPFparameters={'label':'Distributed PF','color':self._painterSettings["color for the distributed PF"],'marker':self._painterSettings["marker for the distributed PF"]},
							figureId='MSE vs Time')

		# distance vs time (only the results for the first topology are plotted)
		plot.distributedPFagainstCentralizedPF(np.arange(self._nTimeInstants),centralizedPF_error[:,0],distributedPF_error[:,0],
							outputFile=self._painterSettings["file name prefix for the euclidean distance vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
							centralizedPFparameters={'label':'Centralized PF','color':self._painterSettings["color for the centralized PF"],'marker':self._painterSettings["marker for the centralized PF"]},
							distributedPFparameters={'label':'Distributed PF','color':self._painterSettings["color for the distributed PF"],'marker':self._painterSettings["marker for the distributed PF"]},
							figureId='Euclidean distance vs Time')

		# the aggregated weights are normalized at ALL TIMES, for EVERY frame and EVERY topology
		normalizedAggregatedWeights = [np.divide(w[:,:,:self._iFrame],w[:,:,:self._iFrame].sum(axis=1)[:,np.newaxis,:]) for w in self._distributedPFaggregatedWeights]
		
		# ...the same data structured in a dictionary
		normalizedAggregatedWeightsDic = {'normalizedAggregatedWeights_{}'.format(i):array for i,array in enumerate(normalizedAggregatedWeights)}
		
		# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
		maxWeights = np.array([(w.max(axis=1)**self._DRNAsettings['q']).mean(axis=1) for w in normalizedAggregatedWeights])

		# evolution of the largest aggregated weight over time (only the results for the first topology are plotted)
		plot.aggregatedWeightsSupremumVsTime(maxWeights[0,:],self._aggregatedWeightsUpperBounds[0],
												self._painterSettings["file name prefix for the aggregated weights supremum vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),self._DRNAsettings["exchange period"])

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
	
	def processFrame(self,targetPosition,targetVelocity):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity)
		
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
				pf.step(self._observations[iTime])
				distributedPf.step(self._observations[iTime])
				
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


class MultipleMposterior(Simulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs)
		
		# HDF5 output file
		self._f = h5py.File('res_' + self._outputFile + '.hdf5','w')
		
		# we will build several "Mposterior" objects...
		self._simulations = []
		
		# ...and each one will have a different set of sensors
		self._sensors = []
		
		# for every pair nPEs-nSensors we aim to simulate...
		for (nPEs,nSensors) in self._simulationParameters["nPEs-nSensors pairs"]:
			
			self._simulations.append(Mposterior(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs,nPEs,nSensors,
									   h5pyFile=self._f,h5pyFilePrefix='{} PEs,{} sensors/'.format(nPEs,nSensors)))

	def processFrame(self,targetPosition,targetVelocity):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity)
		
		for sim in self._simulations:
		
			sim.processFrame(targetPosition,targetVelocity)
		
	def saveData(self,targetPosition):
		
		# let the super class do its thing...
		super().saveData(targetPosition)
		
		self._f.close()


class Mposterior(SimpleSimulation):
	
	# TODO: a method of the object is called from within "__init__" (allowed in python...but weird)
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs,nPEs=None,nSensors=None,h5pyFile=None,h5pyFilePrefix=''):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,outputFile,PRNGs,nPEs,nSensors)
		
		# for saving the data in HDF5
		self._h5pyFile = h5pyFile
		self._h5pyFilePrefix = h5pyFilePrefix
		
		self._simulationParameters = parameters['simulations'][parameters['simulations']['type']]
		self._MposteriorSettings = parameters['Mposterior']
		self._LCDPFsettings = parameters['Likelihood Consensus']
		
		# if the number of PEs is not received...
		if nPEs is None:
			# ...it is looked up in "parameters"
			self._nPEs = self._topologiesSettings['number of PEs']
		# otherwise...
		else:
			self._nPEs = nPEs
		
		# a connector that connects every sensor to every PE
		self._everySensorWithEveryPEConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensorsPositions)
		
		# the settings for the selected "sensors-PES connector"
		sensorsPEsConnectorSettings = parameters['sensors-PEs connectors'][self._simulationParameters['sensors-PEs connector']]
		
		# the positions of the PEs are added as a parameters...technically they are "derived" parameters since they are completely determined by: 
		#	- the corners of the room
		#	- the positions of the sensors which, in turn, also depend on the corners of the room and the number of sensors
		#	- the number of PEs
		self._topologiesSettings['parameters']['PEs positions'] = self._PEsPositions
		
		# ...are used to build a connector, from which the links between PEs and sensors are obtained
		self._PEsSensorsConnections = getattr(sensors_PEs_connector,sensorsPEsConnectorSettings['implementing class'])(
			self._sensorsPositions,self._PEsPositions,sensorsPEsConnectorSettings['parameters']).getConnections(self._nPEs)

		# network topology, which describes the connection among PEs, as well as the exact particles exchanged/shared
		self._PEsTopology = getattr(PEs_topology,self._topologiesSettings['implementing class'])(self._nPEs,self._topologiesSettings['parameters'])

		# import code
		# code.interact(local=dict(globals(), **locals()))
		
		# ...are plot the connections between them		
		sensorsNetworkPlot = plot.TightRectangularRoomPainterWithPEs(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],
														  self._sensorsPositions,self._PEsPositions,self._PEsSensorsConnections,
														  self._PEsTopology.getNeighbours(),sleepTime=self._painterSettings["sleep time between updates"])
		sensorsNetworkPlot.setup()		
		sensorsNetworkPlot.save(outputFile='network_topology_{}_PEs.pdf'.format(self._nPEs))
		
		# the lists of PFs, estimators, colors and labels are initialized...
		self._PFs = []
		self._estimators = []
		self._estimatorsColors = []
		self._estimatorsLabels = []
		
		# ...and algorithms are added
		self.addAlgorithms()
		
		# the position estimates
		self._estimatedPos = np.empty((2,self._nTimeInstants,parameters["number of frames"],len(self._estimators)))
		
		assert len(self._estimatorsColors) == len(self._estimatorsLabels) == len(self._estimators)
		
		# information about the simulated algorithms is added to the parameters...
		parameters['algorithms'] = [{'name':name,'color':color} for name,color in zip(self._estimatorsLabels,self._estimatorsColors)]
		
		# HDF5
		
		# if a reference to an HDF5 file was not received...
		if self._h5pyFile == None:
			# ...a new HDF5 file is created
			self._f = h5py.File('res_' + self._outputFile + '.hdf5','w')
		# otherwise...
		else:
			# the value received is assumed to be a reference to an already open file
			self._f = self._h5pyFile
		
		# this is the number of digits needed to express the frame number
		self._nFramesWidth = math.ceil(math.log10(parameters["number of frames"]))
		
		# the names of the algorithms are also stored
		h5algorithms = self._f.create_dataset(self._h5pyFilePrefix + 'algorithms/names',shape=(len(self._estimators),),dtype=h5py.special_dtype(vlen=str))
		for il,l in enumerate(self._estimatorsLabels):
			h5algorithms[il] = l
		
		# the position and connected sensors of each PE
		for iPE,(pos,sens) in enumerate(zip(self._PEsPositions.T,self._PEsSensorsConnections)):
			self._f.create_dataset(self._h5pyFilePrefix + 'PEs/{}/position'.format(iPE),shape=(2,),data=pos)
			self._f.create_dataset(self._h5pyFilePrefix + 'PEs/{}/connected sensors'.format(iPE),shape=(len(sens),),data=sens)
		
		# the positions of the sensors
		self._f.create_dataset(self._h5pyFilePrefix + 'sensors/positions',shape=self._sensorsPositions.shape,data=self._sensorsPositions)

		algorithms_messages = []

		for estimator,label in zip(self._estimators,self._estimatorsLabels):

			messages_during_estimation = estimator.messages(self._PEsTopology)

			messages_algorithm_operation = estimator.DPF.messages(self._PEsTopology,self._PEsSensorsConnections)

			algorithms_messages.append(messages_during_estimation+messages_algorithm_operation)

			print('{}: messages = {}'.format(label,algorithms_messages[-1]))

			# try:
			#
			# 	messages_during_exchange = estimator.DPF.exchange_recipe.messages()
			#
			# except AttributeError:
			#
			# 	messages_during_exchange = 0
			#
			# print('{}: messages\n\t during estimation = {}\n\t during exchange = {}'.format(label,messages_during_estimation,messages_during_exchange))

		# the messages (per iteration) required by each algorithm
		self._f.create_dataset(self._h5pyFilePrefix + 'algorithms/messages',shape=(len(algorithms_messages),),data=algorithms_messages)
	
	def addAlgorithms(self):
		
		"""Adds the algorithms to be tested by this simulation, defining the required parameters.
		
		"""

		# a copy of the required PRNG is built...so that the exchange particles map is the same for both DRNA and Mposterior
		# TODO: is this really necesary? a better approach?
		import copy
		PRNGcopy = copy.deepcopy(self._PRNGs["topology pseudo random numbers generator"])

		DRNA_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(
			self._PEsTopology, self._K, self._simulationParameters["exchanged particles"],
			PRNG=self._PRNGs["topology pseudo random numbers generator"])
		# Mposterior_exchange_recipe = smc.exchange_recipe.MposteriorExchangeRecipe(
		# 	self._PEsTopology,self._K,self._simulationParameters["exchanged particles"],PRNG=PRNGcopy)
		Mposterior_exchange_recipe = smc.exchange_recipe.IteratedMposteriorExchangeRecipe(
			self._PEsTopology, self._K, self._simulationParameters["exchanged particles"],
			self._MposteriorSettings["number of iterations"], PRNG=PRNGcopy)
		likelihood_consensus_exchange_recipe = smc.exchange_recipe.LikelihoodConsensusExchangeRecipe(self._PEsTopology,
		                                            self._LCDPFsettings['number of consensus iterations'],self._LCDPFsettings['degree of the polynomial approximation'])

		nMessages = DRNA_exchange_recipe.messages()
		print('number of messages DRNA = {}'.format(nMessages))
		nMessages = Mposterior_exchange_recipe.messages()
		print('number of messages Mposterior = {}'.format(nMessages))
		nMessages = likelihood_consensus_exchange_recipe.messages()
		print('number of messages LC = {}'.format(nMessages))
		
		# consensus
		self._PFs.append(
			particle_filter.LikelihoodConsensusDistributedTargetTrackingParticleFilter(
				likelihood_consensus_exchange_recipe,self._nPEs,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._PEsSensorsConnections,self._LCDPFsettings['degree of the polynomial approximation'],
				PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilterWithConsensusCapabilities
				)
		)
		
		# the estimator just delegates the calculus of the estimate to one of the PEs
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1],0))
		
		self._estimatorsColors.append('brown')
		self._estimatorsLabels.append('LC DPF')
		
		# ------------

		# a single PE (with the number of particles of any other PE) that has access to all the observations
		self._PFs.append(
			particle_filter.CentralizedTargetTrackingParticleFilter(
				self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,self._sensors
				)
		)
			
		# the estimator just delegates the calculus of the estimate to the PF
		self._estimators.append(smc.estimator.Delegating(self._PFs[-1]))
		
		self._estimatorsColors.append('indigo')
		self._estimatorsLabels.append('Single know-it-all PE')
		
		# ------------

		# centralized PF
		self._PFs.append(
			particle_filter.CentralizedTargetTrackingParticleFilter(
				self._K*self._nPEs,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,self._sensors
				)
		)
			
		# the estimator just delegates the calculus of the estimate to the PF
		self._estimators.append(smc.estimator.Delegating(self._PFs[-1]))
		
		self._estimatorsColors.append('lawngreen')
		self._estimatorsLabels.append('Centralized')
		
		# ------------
		
		# a distributed PF with DRNA
		self._PFs.append(
			smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],DRNA_exchange_recipe,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
				self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
			)
		)
		
		
		# the estimator is the mean
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('black')
		self._estimatorsLabels.append('DRNA')

		print('{}: {}'.format(self._estimatorsLabels[-1],self._PFs[-1].messages(self._PEsTopology,self._PEsSensorsConnections)))
		
		# ------------
		
		# a distributed PF using a variation of DRNA in which each PE only sees a subset of the observations
		self._PFs.append(
			smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],DRNA_exchange_recipe,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
				self._prior,self._transitionKernel,self._sensors,self._PEsSensorsConnections,PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
			)
		)
		
		# the estimator is still the mean
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('magenta')
		self._estimatorsLabels.append('DRNA (partial observations)')

		print('{}: {}'.format(self._estimatorsLabels[-1],self._PFs[-1].messages(self._PEsTopology,self._PEsSensorsConnections)))
		
		# ------------
		
		# a "distributed" PF in which each PE does its computation independently of the rest
		self._PFs.append(
			smc.particle_filter.DistributedTargetTrackingParticleFilter(
				self._nPEs,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._PEsSensorsConnections,PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		# yes...still the mean
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('blue')
		self._estimatorsLabels.append('Plain DPF')

		# ------------

		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(self._PFs[-1],maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
														tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
		self._estimatorsColors.append('seagreen')
		self._estimatorsLabels.append('Plain DPF (geometric median with 1 particle from each PE)')
		
		# ------------

		# DPF with M-posterior-based exchange
		self._PFs.append(
			smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
				Mposterior_exchange_recipe,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._PEsSensorsConnections,self._MposteriorSettings['findWeiszfeldMedian parameters'],
				self._MposteriorSettings['sharing period'],PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter)
		)
		
		## an estimator combining all the particles from all the PEs through M-posterior to give a distribution whose mean is the estimate
		#self._estimators.append(smc.estimator.Mposterior(self._PFs[-1]))
		
		#self._estimatorsColors.append('darkred')
		#self._estimatorsLabels.append('M-posterior (M-posterior with ALL particles - mean)')
		
		# ------------
		
		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(self._PFs[-1],maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
														tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
		self._estimatorsColors.append('green')
		self._estimatorsLabels.append('M-posterior (geometric median with 1 particle from each PE)')
		
		# ------------
		
		# an estimator which yields the mean of ALL the particles
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('red')
		self._estimatorsLabels.append('M-posterior (Mean)')
		
		# ------------
		
		# DPF with M-posterior-based exchange that gets its estimates from the mean of the particles in the first PE
		iPE,color = 0,'olive'
		
		# an estimator which yields the mean of the particles in the "iPE"-th PE
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1],iPE))
		
		self._estimatorsColors.append(color)
		self._estimatorsLabels.append('M-posterior (mean with particles from PE \#{})'.format(iPE))
		
		# ------------
		
		# DPF with M-posterior-based exchange that gets its estimates from the geometric median of the particles in the first PE
		iPE,color = 0,'crimson'
		
		# an estimator which yields the geometric median of the particles in the "iPE"-th PE
		self._estimators.append(smc.estimator.SinglePEGeometricMedian(self._PFs[-1],iPE))
		
		self._estimatorsColors.append(color)
		self._estimatorsLabels.append('M-posterior (geometric median with particles from PE \#{})'.format(iPE))

		# ------------

		# DPF with M-posterior-based exchange that gets its estimates from the geometric median of the particles in the first PE
		iPE= 0

		for radius,color in zip([1,2,3,4,5],['deeppink','cornsilk','sienna','coral','orchid']):

			# an estimator which yields the geometric median of the particles in the "iPE"-th PE
			self._estimators.append(smc.estimator.SinglePEGeometricMedianWithinRadius(self._PFs[-1],iPE,self._PEsTopology,radius))

			print('messages with radius {} = {}'.format(radius,self._estimators[-1].messages(self._PEsTopology)))

			self._estimatorsColors.append(color)
			self._estimatorsLabels.append('M-posterior ({} hops geometric median with particles from PE \#{})'.format(radius,iPE))
		
		# ------------

	def saveData(self,targetPosition):
		
		# let the super class do its thing...
		super().saveData(targetPosition)
		
		# a dictionary encompassing all the data to be saved
		dataToBeSaved = dict(
				targetPosition = targetPosition[:,:,:self._iFrame],
				PF_pos = self._estimatedPos[:,:,:self._iFrame,:]
			)
		
		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**dataToBeSaved)
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))
		
		# the mean of the error (euclidean distance) incurred by the PFs
		PF_error = np.sqrt((np.subtract(self._estimatedPos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)
		
		plot.PFs(range(self._nTimeInstants),PF_error,
		   self._simulationParameters["file name prefix for the estimation error vs time plot"] + '_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
			[{'label':l,'color':c} for l,c in zip(self._estimatorsLabels,self._estimatorsColors)])
		
		print(self._estimatedPos)
		
		# if a reference to an HDF5 was not received, that means the file was created by this object,
		# and hence it is responsible of closing it...
		if self._h5pyFile is None:
			
			# ...in order to make sure the HDF5 file is valid...
			self._f.close()
		
	def processFrame(self,targetPosition,targetVelocity):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity)
		
		# a reference to the "group" for the current frame (notice the prefix in the name given "self._h5pyFilePrefix")...
		h5thisFrame = self._f.create_group(self._h5pyFilePrefix + 'frames/{num:0{width}}'.format(num=self._iFrame, width=self._nFramesWidth))
		
		# ...where a new dataset (initialized with NaN's) is created for the "actual position" of the target...
		h5actualPos = h5thisFrame.create_dataset('actual position',shape=(2,self._nTimeInstants),dtype=float,data=np.full((2,self._nTimeInstants),np.nan))
		
		# ...and another one (also initialized with NaN's) for the "estimated position"
		h5estimatedPos = h5thisFrame.create_dataset('estimated position',shape=(2,self._nTimeInstants,len(self._estimators)),dtype=float,data=np.full((2,self._nTimeInstants,len(self._estimators)),np.nan))
		
		# for every PF (different from estimator)...
		for pf in self._PFs:
			
			# ...initialization
			pf.initialize()
		
		if self._painterSettings['display evolution?']:
			
			# if this is not the first iteration...
			if hasattr(self,'_painter'):
				
				# ...then, the previous figure is closed
				self._painter.close()

			# this object will handle graphics...
			self._painter = plot.RectangularRoomPainter(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],self._sensorsPositions,sleepTime=self._painterSettings["sleep time between updates"])

			# ...e.g., draw the self._sensors
			self._painter.setup()

		for iTime in range(self._nTimeInstants):

			print('---------- iFrame = {}, iTime = {}'.format(repr(self._iFrame),repr(iTime)))

			print('position:\n',targetPosition[:,iTime:iTime+1])
			print('velocity:\n',targetVelocity[:,iTime:iTime+1])
			
			# the actual position of the target is written to the HDF5 file
			h5actualPos[:,iTime:iTime+1] = targetPosition[:,iTime:iTime+1]
			
			# for every PF (different from estimator)...
			for pf in self._PFs:
				
				# ...a step is taken
				pf.step(self._observations[iTime])
			
			# for every estimator, along with its corresponding label,...
			for iEstimator,(estimator,label) in enumerate(zip(self._estimators,self._estimatorsLabels)):
				
				self._estimatedPos[:,iTime:iTime+1,self._iFrame,iEstimator] = state.position(estimator.estimate())
				
				# the position given by this estimator at the current time instant is written to the HDF5 file
				h5estimatedPos[:,iTime:iTime+1,iEstimator] = state.position(estimator.estimate())
				
				print('position estimated by {}\n'.format(label),self._estimatedPos[:,iTime:iTime+1,self._iFrame,iEstimator])
			
			if self._painterSettings["display evolution?"]:

				# the plot is updated with the position of the target...
				self._painter.updateTargetPosition(targetPosition[:,iTime:iTime+1])
				
				# ...those estimated by the PFs
				for iEstimator,(pf,color) in enumerate(zip(self._estimators,self._estimatorsColors)):
					
					self._painter.updateEstimatedPosition(self._estimatedPos[:,iTime:iTime+1,self._iFrame,iEstimator],identifier='#{}'.format(iEstimator),color=color)
					
					if self._painterSettings["display particles evolution?"]:
						
						self._painter.updateParticlesPositions(state.position(pf.getState()),identifier='#{}'.format(iEstimator),color=color)

		# in order to make sure the HDF5 files is valid...
		self._f.flush()

class MposteriorExchange(Mposterior):
		
	def addAlgorithms(self):

		# available colors
		colors = ['red','blue','green','goldenrod','cyan','crimson','lime','cadetblue','magenta']

		# topology of the network
		topology = getattr(PEs_topology,self._topologiesSettings['implementing class'])(self._nPEs,self._topologiesSettings['parameters'])

		for iPercentage,(percentage,color) in enumerate(zip(self._simulationParameters["exchanged particles"],colors)):

			DRNA_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(topology,self._K,percentage,PRNG=self._PRNGs["topology pseudo random numbers generator"])

			# a distributed PF with DRNA
			self._PFs.append(
				smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
					self._DRNAsettings["exchange period"],DRNA_exchange_recipe,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
					self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
				)
			)

			self._estimators.append(smc.estimator.Mean(self._PFs[-1]))

			self._estimatorsColors.append('black')
			self._estimatorsLabels.append('DRNA {}'.format(percentage))

			# ------------

			Mposterior_exchange_recipe = smc.exchange_recipe.MposteriorExchangeRecipe(self._PEsTopology,self._K,percentage,PRNG=self._PRNGs["topology pseudo random numbers generator"])

			self._PFs.append(
				smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
					Mposterior_exchange_recipe,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
					self._sensors,self._PEsSensorsConnections,self._MposteriorSettings['findWeiszfeldMedian parameters'],
					self._MposteriorSettings['sharing period'],PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter
				)
			)

			self._estimators.append(smc.estimator.GeometricMedian(self._PFs[-1],
			                        maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
			                            tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

			self._estimatorsColors.append(color)
			self._estimatorsLabels.append('M-posterior {}'.format(percentage))


	def saveData(self,targetPosition):
		
		# the method from the grandparent
		SimpleSimulation.saveData(self,targetPosition)

		# FIXME: this shoudn't happen every time save is called
		del self._f['algorithms/names']
		h5algorithms = self._f.create_dataset('algorithms/names',shape=(2,),dtype=h5py.special_dtype(vlen=str))
		h5algorithms[0] = 'DRNA'
		h5algorithms[1] = 'Mposterior'

		h5_algorithms_colors = self._f.create_dataset('algorithms/colors',shape=(2,),dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_colors[0] = 'black'
		h5_algorithms_colors[1] = 'blue'

		# for every frame previously stored
		for frame_number in self._f['frames']:

			# for the sake of convenience
			frame = self._f['frames'][frame_number]

			# the estimated position for every algorithm and every "exchange value" is stored in an auxiliar variable...
			frame['aux'] = frame['estimated position']

			# ...so that we can delete the corresponding dataset and reuse the name
			del frame['estimated position']

			# for every "exchange value"...
			for i,exchanged_particles in enumerate(self._simulationParameters["exchanged particles"]):

				# ...the results for all the algorithms are stored in the appropriate place
				frame['estimated position/exchanged particles/{}'.format(exchanged_particles)] = frame['aux'][...,i*2:(i+1)*2]

			# we don't need this anymore
			del frame['aux']

		# if a reference to an HDF5 was not received, that means the file was created by this object, and hence it is responsibility to close it...
		if self._h5pyFile is None:

			# ...in order to make sure the HDF5 file is valid...
			self._f.close()

class MposteriorGeometricMedian(Mposterior):
	
	def addAlgorithms(self):
		
		# a copy of the required PRNG is built...so that the exchange particles map is the same for both DRNA and Mposterior
		# TODO: is this really necesary? a better approach?
		import copy
		PRNGcopy = copy.deepcopy(self._PRNGs["topology pseudo random numbers generator"])

		DRNA_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(self._PEsTopology,self._K,self._simulationParameters["exchanged particles"],PRNG=self._PRNGs["topology pseudo random numbers generator"])
		Mposterior_exchange_recipe = smc.exchange_recipe.MposteriorExchangeRecipe(self._PEsTopology,self._K,self._simulationParameters["exchanged particles"],PRNG=PRNGcopy)
		
		# a distributed PF with DRNA
		self._PFs.append(
			smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],DRNA_exchange_recipe,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
				self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
			)
		)
		
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('black')
		self._estimatorsLabels.append('DRNA')
		
		# ------------
		
		# available colors
		colors = ['red','blue','green','goldenrod','cyan','crimson','lime','cadetblue','magenta']
		
		# DPF with M-posterior-based exchange
		self._PFs.append(
			smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
				Mposterior_exchange_recipe,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._PEsSensorsConnections,self._MposteriorSettings['findWeiszfeldMedian parameters'],
				self._MposteriorSettings['sharing period'],PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter,
			)
		)
		
		for nParticles,col in zip(self._simulationParameters['number of particles for estimation'],colors):
			
			self._estimators.append(smc.estimator.StochasticGeometricMedian(
				self._PFs[-1],nParticles,maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
			
			self._estimatorsColors.append(col)
			self._estimatorsLabels.append('M-posterior (Stochastic Geometric Median with {} particles from each PE)'.format(nParticles))
