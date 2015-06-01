import abc
import numpy as np
import scipy.io
import math

from smc import particle_filter
import smc.estimator
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
		
		# the settings for the topology or topologies given...if it is a list...
		if isinstance(parameters['topologies']['type'],list):
			# ...we have a list of settings
			self._topologiesSettings = [parameters['topologies'][i] for i in parameters['topologies']['type']]
		# otherwise...
		else:
			# the "topology settings" object is just a dictionary
			self._topologiesSettings = parameters['topologies'][parameters['topologies']['type']]
		
		# sensors positions are gathered
		self._sensorsPositions = np.hstack([s.position for s in sensors])
		
		# so that it equals 0 the first time it is incremented...
		self._iFrame = -1
		
	@abc.abstractmethod
	def processFrame(self,targetPosition,targetVelocity,observations):
		
		self._iFrame += 1
	
	# TODO: remove targetPosition as argument?
	
	@abc.abstractmethod
	def saveData(self,targetPosition):

		if self._iFrame==0:
			print('saveData: nothing to save...skipping')
			return

class Convergence(Simulation):
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs)
		
		topologies = [getattr(topology,t['implementing class'])(t['number of PEs'],self._K,self._DRNAsettings["exchanged particles maximum percentage"],t['parameters'],
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

class Mposterior(Simulation):
	
	# TODO: a method of the object is called from within "__init__" (allowed in python...but weird)
	
	def __init__(self,parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors,outputFile,PRNGs)
		
		self._simulationParameters = parameters['simulations'][parameters['simulations']['type']]
		self._MposteriorSettings = parameters['Mposterior']
		
		# these parameters are kept for use in the "addAlgorithms" method below
		self._resamplingAlgorithm = resamplingAlgorithm
		self._resamplingCriterion = resamplingCriterion
		self._prior = prior
		self._transitionKernel = transitionKernel
		self._sensors = sensors
		self._PRNGs = PRNGs
		
		# for the sake of clarity...since this will be used many times
		self._nPEs = self._topologiesSettings['number of PEs']
		
		# a connector that connects every sensor to every PE
		self._everySensorWithEveryPEConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(sensors)
		
		# the parameters given by the user for a connector that connects every sensor to the closest PE
		sensorWithTheClosestPEConnectorSettings = parameters['sensors-PEs connectors']['sensors connected to the closest PEs']
		
		# the positions of the sensors...
		sensorsPositions = np.hstack([s.position for s in sensors])
		
		# ...and those of the PEs...
		PEsPositions = sensors_PEs_connector.computePEsPositions(self._roomSettings["bottom left corner"],self._roomSettings["top right corner"],
														   self._nPEs,sensorWithTheClosestPEConnectorSettings['parameters']['number of uniform samples']*self._nPEs)
		
		# the positions of the PEs are added as a parameters...technically they are "derived" parameters since they are completely determined by: 
		#	- the corners of the room
		#	- the positions of the sensors which, in turn, also depend on the corners of the room and the number of sensors
		#	- the number of PEs
		self._topologiesSettings['parameters']['PEs positions'] = PEsPositions
		
		# ...are used to build the corresponding connector
		self._sensorWithTheClosestPEConnector = getattr(sensors_PEs_connector,sensorWithTheClosestPEConnectorSettings['implementing class'])(sensors,PEsPositions,sensorWithTheClosestPEConnectorSettings['parameters'])
		
		# network topology, which describes the connection among PEs, as well as the exact particles exchanged/shared
		self._networkTopology = getattr(topology,self._topologiesSettings['implementing class'])(self._nPEs,self._K,self._simulationParameters["exchanged particles maximum percentage"],
																					 self._topologiesSettings['parameters'],PRNG=self._PRNGs["topology pseudo random numbers generator"])
		
		# ...are plot the connections between them
		#plot.PEsSensorsConnections(sensorsPositions,PEsPositions,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs))
		plot.PEsSensorsAndPEsPEsConnections(sensorsPositions,PEsPositions,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs),self._networkTopology.getNeighbours())
		
		# the list of algorithms (objects) to be executed, along with their corresponding colors and labels are empty before...
		self._PFs = []
		self._estimators = []
		self._estimatorsColors = []
		self._estimatorsLabels = []
		
		# ...algorithms are added
		self.addAlgorithms()
		
		# the position estimates
		self._estimatedPos = np.empty((2,self._nTimeInstants,parameters["number of frames"],len(self._estimators)))
		
		assert len(self._estimatorsColors) == len(self._estimatorsLabels) == len(self._estimators)
		
		# information about the simulated algorithms is added to the parameters...
		parameters['algorithms'] = [{'name':name,'color':color} for name,color in zip(self._estimatorsLabels,self._estimatorsColors)]
		
		# HDF5
		import h5py
		
		# a reference to the HDF5 file
		self._f = h5py.File('res_' + self._outputFile + '.hdf5','w')
		
		# this is the number of digits needed to express the frame number
		self._nFramesWidth = math.ceil(math.log10(parameters["number of frames"]))
		
		## the names of the algorithms are also stored
		#h5algorithms = self._f.create_dataset('algorithms',shape=(len(self._estimators),),dtype=h5py.special_dtype(vlen=str))
		#for il,l in enumerate(self._estimatorsLabels):
			#h5algorithms[il] = l
		
	
	def addAlgorithms(self):
		
		"""Adds the algorithms to be tested by this simulation, defining the required parameters.
		
		"""

		# centralized PF
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

		# a parameter for the DRNA algorithm
		DRNAaggregatedWeightsUpperBound = drnautil.supremumUpperBound(self._nPEs,self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon'])
		
		# centralized PF
		self._PFs.append(
			particle_filter.CentralizedTargetTrackingParticleFilter(
				self._K*self._networkTopology.getNumberOfPEs(),self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,self._sensors
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
				self._DRNAsettings["exchange period"],self._networkTopology,DRNAaggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
				self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
			)
		)
		
		
		# the estimator is the mean
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('black')
		self._estimatorsLabels.append('DRNA')
		
		# ------------
		
		# a distributed PF using a variation of DRNA in which each PE only sees a subset of the observations
		self._PFs.append(
			smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],self._networkTopology,DRNAaggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
				self._prior,self._transitionKernel,self._sensors,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
			)
		)
		
		# the estimator is still the mean
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('magenta')
		self._estimatorsLabels.append('DRNA (partial observations)')
		
		# ------------
		
		# a "distributed" PF in which each PE does its computation independently of the rest
		self._PFs.append(
			smc.particle_filter.DistributedTargetTrackingParticleFilter(
				self._nPEs,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)
		
		# yes...still the mean
		self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
		
		self._estimatorsColors.append('blue')
		self._estimatorsLabels.append('Plain DPF')
		
		# ------------

		# DPF with M-posterior-based exchange
		self._PFs.append(
			smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
				self._networkTopology,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs),self._MposteriorSettings['findWeiszfeldMedian parameters'],
				self._MposteriorSettings['sharing period'],exchangeManager=smc.mposterior.share.DeterministicExchange(),
				PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter)
		)
		
		# an estimator combining all the particles from all the PEs through M-posterior to give a distribution whose mean is the estimate
		self._estimators.append(smc.estimator.Mposterior(self._PFs[-1]))
		
		self._estimatorsColors.append('darkred')
		self._estimatorsLabels.append('M-posterior (M-posterior with ALL particles - mean)')
		
		# ------------
		
		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(self._PFs[-1],
														maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
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
		self._estimators.append(smc.estimator.SinglePEmean(self._PFs[-1],iPE))
		
		self._estimatorsColors.append(color)
		self._estimatorsLabels.append('M-posterior (mean with particles from PE \#{})'.format(iPE))
		
		# ------------
		
		# DPF with M-posterior-based exchange that gets its estimates from the geometric median of the particles in the first PE
		iPE,color = 0,'crimson'
		
		# an estimator which yields the geometric median of the particles in the "iPE"-th PE
		self._estimators.append(smc.estimator.SinglePEgeometricMedian(self._PFs[-1],iPE))
		
		self._estimatorsColors.append(color)
		self._estimatorsLabels.append('M-posterior (geometric median with particles from PE \#{})'.format(iPE))
		
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
		
		# in order to make sure the hdf5 files is valid...
		self._f.close()
		
	def processFrame(self,targetPosition,targetVelocity,observations):
		
		# let the super class do its thing...
		super().processFrame(targetPosition,targetVelocity,observations)
		
		# HDF5
		
		# a reference to the group for the current frame...
		h5thisFrame = self._f.create_group('frame {num:0{width}}'.format(num=self._iFrame, width=self._nFramesWidth))
		
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
				pf.step(observations[iTime])
			
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

class MposteriorExchangePercentage(Mposterior):
		
	def addAlgorithms(self):
		
		# the coordinates associated with a given estimator for summarizing purposes
		self._estimatorsCoordinates = []
		
		# a parameter for the DRNA algorithm
		DRNAaggregatedWeightsUpperBound = drnautil.supremumUpperBound(self._nPEs,self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon'])
		
		# available colors
		colors = ['red','blue','green','goldenrod','cyan','crimson','lime','cadetblue','magenta']

		for iPercentage,(percentage,color) in enumerate(zip(self._simulationParameters["exchanged particles maximum percentage"],colors)):
			
			# topologies of the network, which includes the percentage of particles exchanged
			self._networkTopology = getattr(topology,self._topologiesSettings['implementing class'])(self._nPEs,self._K,percentage,self._topologiesSettings['parameters'],PRNG=self._PRNGs["topology pseudo random numbers generator"])
			
			# a distributed PF with DRNA
			self._PFs.append(
				smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
					self._DRNAsettings["exchange period"],self._networkTopology,DRNAaggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
					self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),PFsClass=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
				)
			)
			
			self._estimators.append(smc.estimator.Mean(self._PFs[-1]))
			
			self._estimatorsColors.append('black')
			self._estimatorsLabels.append('DRNA {}'.format(percentage))
			
			self._estimatorsCoordinates.append((0,iPercentage))
			
			# ------------
			
			self._PFs.append(
				smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
					self._networkTopology,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
					self._sensors,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs),self._MposteriorSettings['findWeiszfeldMedian parameters'],
					self._MposteriorSettings['sharing period'],exchangeManager=smc.mposterior.share.DeterministicExchange(),
					PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter
				)
			)
			
			self._estimators.append(smc.estimator.GeometricMedian(self._PFs[-1],
														maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
			
			self._estimatorsColors.append(color)
			self._estimatorsLabels.append('M-posterior {}'.format(percentage))
			
			self._estimatorsCoordinates.append((1,iPercentage))
			

	def saveData(self,targetPosition):
		
		# the method from the grandparent
		Simulation.saveData(self,targetPosition)
		
		# the mean of the error (euclidean distance) incurred by the PFs
		error_vs_time = np.sqrt((np.subtract(self._estimatedPos[:,:,:self._iFrame,:],targetPosition[:,:,:self._iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)
		
		estimators_summaries = error_vs_time[self._simulationParameters['starting time instant as percentage of the frame length']*self._nTimeInstants:].sum(axis=0)
		
		# the number of percentages tested yields the number of columns, and the number of rows is inferred from that and the number of tuples in "self._estimatorsCoordinates"
		error_vs_percentage = np.empty((len(self._estimatorsCoordinates)/len(self._simulationParameters['exchanged particles maximum percentage']),len(self._simulationParameters['exchanged particles maximum percentage'])))
		
		for summary,coordinates in zip(estimators_summaries,self._estimatorsCoordinates):
			
			error_vs_percentage[coordinates] = summary

		# a dictionary encompassing all the data to be saved
		dataToBeSaved = dict(
				percentages = self._simulationParameters['exchanged particles maximum percentage'],
				mean_error = error_vs_percentage
			)
		
		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**dataToBeSaved)
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))

		import code
		code.interact(local=dict(globals(), **locals()))

class MposteriorGeometricMedian(Mposterior):
	
	def addAlgorithms(self):
		
		# a parameter for the DRNA algorithm
		DRNAaggregatedWeightsUpperBound = drnautil.supremumUpperBound(self._nPEs,self._DRNAsettings['c'],self._DRNAsettings['q'],self._DRNAsettings['epsilon'])
		
		# a distributed PF with DRNA
		self._PFs.append(
			smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],self._networkTopology,DRNAaggregatedWeightsUpperBound,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
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
				self._networkTopology,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
				self._sensors,self._sensorWithTheClosestPEConnector.getConnections(self._nPEs),self._MposteriorSettings['findWeiszfeldMedian parameters'],
				self._MposteriorSettings['sharing period'],exchangeManager=smc.mposterior.share.DeterministicExchange(),
				PFsClass=smc.particle_filter.CentralizedTargetTrackingParticleFilter,
			)
		)
		
		for nParticles,col in zip(self._simulationParameters['number of particles for estimation'],colors):
			
			self._estimators.append(smc.estimator.StochasticGeometricMedian(
				self._PFs[-1],nParticles,maxIterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
			
			self._estimatorsColors.append(col)
			self._estimatorsLabels.append('M-posterior (Stochastic Geometric Median with {} particles from each PE)'.format(nParticles))
