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
	def __init__(self, parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs):
		
		# these parameters are kept for later use
		self._resamplingAlgorithm = resampling_algorithm
		self._resamplingCriterion = resampling_criterion
		self._prior = prior
		self._transitionKernel = transition_kernel
		self._PRNGs = PRNGs
		
		# number of particles per processing element (PE)
		self._K = parameters["number of particles per PE"]
		
		# length of the trajectory
		self._nTimeInstants = parameters["number of time instants"]
		
		# name of the file to store the results
		self._outputFile = output_file
		
		# DRNA related
		self._DRNAsettings = parameters["DRNA"]
		
		# parameters related to plotting
		self._painterSettings = parameters["painter"]
		
		# room  dimensions
		self._roomSettings = parameters["room"]
		
		# the settings for the topology or topologies given...if it is a list...
		if isinstance(parameters['topologies']['type'],list):
			# ...we have a list of settings
			self._settings_topologies = [parameters['topologies'][i] for i in parameters['topologies']['type']]
		# otherwise...
		else:
			# the "topology settings" object is just a dictionary
			self._settings_topologies = parameters['topologies'][parameters['topologies']['type']]
		
		# so that it equals 0 the first time it is incremented...
		self._iFrame = -1
		
		# the parameters for this particular simulation are obtained
		self._simulationParameters = parameters['simulations'][parameters['simulations']['type']]
		
	@abc.abstractmethod
	def process_frame(self, target_position, target_velocity):
		
		self._iFrame += 1
	
	# TODO: remove targetPosition as argument?
	
	@abc.abstractmethod
	def save_data(self, target_position):

		if self._iFrame == 0:
			print('save_data: still in the first frame...maybe nothing will be saved')


class SimpleSimulation(Simulation):
	
	def __init__(self, parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs, h5py_file, h5py_prefix, n_PEs=None, n_sensors=None):
		
		super().__init__(parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs)

		# for saving the data in HDF5
		self._h5py_file = h5py_file
		self._h5py_prefix = h5py_prefix

		# if a reference to an HDF5 file was not received...
		if h5py_file is None:
			# ...a new HDF5 file is created
			self._f = h5py.File('res_' + self._outputFile + '.hdf5', 'w', driver='core', libver='latest')
			# self._f = h5py.File('res_' + self._outputFile + '.hdf5', 'w')
		# otherwise...
		else:
			# the value received is assumed to be a reference to an already open file
			self._f = self._h5py_file

		# this is the number of digits needed to express the frame number
		self._nFramesWidth = math.ceil(math.log10(parameters["number of frames"]))

		# for the sake of convenience
		sensors_settings = parameters["sensors"]

		if n_sensors is None:
			
			n_sensors = parameters["sensors"]['number']
		
		if n_PEs is None:

			# we try to extract the "number of PEs" from the topology settings...
			try:
				
				n_PEs = self._settings_topologies['number of PEs']
			
			# ...it it's not possible, it is because there are multiple topology settings => Convergence simulation
			except TypeError:
				
				# n_PEs = None
				pass
		
		# for the sake of convenience below...
		network_nodes_settings = parameters['network nodes'][self._simulationParameters['network']]
		network_nodes_class = getattr(network_nodes, network_nodes_settings['implementing class'])
		
		# the appropriate class is instantiated with the given parameters
		self._network = network_nodes_class(
			self._roomSettings["bottom left corner"], self._roomSettings["top right corner"], n_PEs, n_sensors,
			**network_nodes_settings['parameters'])
		
		# the positions of the PEs and the sensors are collected from the network just built
		self._sensorsPositions, self._PEsPositions = self._network.sensorsPositions, self._network.PEsPositions
		
		# the class to be instantiated is figured out from the settings for that particular sensor type
		sensor_class = getattr(sensor,sensors_settings[sensors_settings['type']]['implementing class'])
		
		self._sensors = [sensor_class(
			pos[:, np.newaxis],
			pseudo_random_numbers_generator=PRNGs['Sensors and Monte Carlo pseudo random numbers generator'],
			**sensors_settings[sensors_settings['type']]['parameters']
		) for pos in self._sensorsPositions.T]

		# these are going to be set/used by other methods
		self._observations = None
		self._h5_current_frame = None
		
	def process_frame(self, target_position, target_velocity):
		
		super().process_frame(target_position, target_velocity)
		
		# observations for all the sensors at every time instant (each list)
		# NOTE: conversion to float is done so that the observations (1 or 0) are amenable to be used in later computations
		self._observations = [np.array(
			[sensor.detect(state.position(s[:, np.newaxis])) for sensor in self._sensors], dtype=float
		) for s in target_position.T]
		
		# a reference to the "group" for the current frame (notice the prefix in the name given "self._h5py_prefix")...
		self._h5_current_frame = self._f.create_group(
			self._h5py_prefix + 'frames/{num:0{width}}'.format(num=self._iFrame, width=self._nFramesWidth))
		
		# ...where a new dataset is created for the "actual position" of the target...
		self._h5_current_frame.create_dataset(
			'actual position', shape=(2, self._nTimeInstants), dtype=float, data=target_position)

	def save_data(self, target_position):

		super().save_data(target_position)

		# if a reference to an HDF5 was not received, that means the file was created by this object,
		# and hence it is responsible of closing it...
		if self._h5py_file is None:

			# ...in order to make sure the HDF5 file is valid...
			self._f.close()

	def save_this_frame_pseudo_random_numbers_generators(self, pseudo_random_numbers_generators):

		self.save_pseudo_random_numbers_generators_in_hdf5_group(pseudo_random_numbers_generators, self._h5_current_frame)

	def save_initial_pseudo_random_numbers_generators(self, pseudo_random_numbers_generators):

		self.save_pseudo_random_numbers_generators_in_hdf5_group(pseudo_random_numbers_generators, self._f)

	@staticmethod
	def save_pseudo_random_numbers_generators_in_hdf5_group(pseudo_random_numbers_generators, group):

		for key, value in pseudo_random_numbers_generators.items():

			prng_state = value.get_state()

			group.create_dataset(
				'pseudo random numbers generators/{}/1'.format(key), shape=(1,), dtype=h5py.special_dtype(vlen=str))
			group['pseudo random numbers generators/{}/1'.format(key)][0] = 'MT19937'

			group.create_dataset(
				'pseudo random numbers generators/{}/2'.format(key), shape=prng_state[1].shape, dtype=np.uint,
				data=prng_state[1])
			group.create_dataset(
				'pseudo random numbers generators/{}/3'.format(key), shape=(1,), dtype=int, data=prng_state[2])
			group.create_dataset(
				'pseudo random numbers generators/{}/4'.format(key), shape=(1,), dtype=int, data=prng_state[3])
			group.create_dataset(
				'pseudo random numbers generators/{}/5'.format(key), shape=(1,), dtype=float, data=prng_state[4])

	@staticmethod
	def pseudo_random_numbers_generators_from_file(filename, i_frame):

		with h5py.File(filename,'r') as data_file:

			prngs = data_file['frames/{}/pseudo random numbers generators'.format(i_frame)]

			res = {}

			for p in prngs:

				state = [None]*5

				state[0] = prngs[p]['1'][0]
				state[1] = prngs[p]['2'][...]
				state[2] = prngs[p]['3'][0]
				state[3] = prngs[p]['4'][0]
				state[4] = prngs[p]['5'][0]

				res[p] = np.random.RandomState()
				res[p].set_state(tuple(state))

		return res


class Convergence(SimpleSimulation):

	@staticmethod
	def parse_hdf5(data_file):

		n_state, n_time_instants, n_algorithms = data_file['frames/0/topology/0/estimated position'].shape
		n_topologies = len(data_file['frames/0/topology'])
		n_frames = len(data_file['frames'])

		estimated_position = np.empty((n_state, n_time_instants, n_algorithms, n_frames, n_topologies))

		for i_frame, frame in enumerate(data_file['frames']):

			for i_topology, topology in enumerate(data_file['frames/{}/topology'.format(i_frame)]):
				estimated_position[..., i_frame, i_topology] = data_file[
					'frames/{}/topology/{}/estimated position'.format(i_frame, i_topology)]

		actual_position = np.concatenate(
			[data_file['frames/{}/actual position'.format(i)][...][..., np.newaxis] for i in data_file['frames']],
			axis=2)

		return actual_position, estimated_position

	def __init__(
			self, parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			PRNGs, h5py_file=None, h5py_prefix=''):

		# let the super class do its thing...
		super().__init__(
			parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			PRNGs, h5py_file, h5py_prefix)

		topologies = [getattr(PEs_topology,t['implementing class'])(
			t['number of PEs'], t['parameters']) for t in self._settings_topologies]

		exchange_recipes = [smc.exchange_recipe.DRNAExchangeRecipe(
			t, self._K, self._simulationParameters["exchanged particles"],
			PRNG=self._PRNGs["topology pseudo random numbers generator"]) for t in topologies]

		# we compute the upper bound for the supremum of the aggregated weights that should guarante convergence
		self._aggregatedWeightsUpperBounds = [drnautil.supremum_upper_bound(
			t['number of PEs'], self._DRNAsettings['c'], self._DRNAsettings['q'], self._DRNAsettings['epsilon']
		) for t in self._settings_topologies]

		# plain non-parallelized particle filter
		self._PFsForTopologies = [particle_filter.CentralizedTargetTrackingParticleFilter(
			self._K*t.n_processing_elements, resampling_algorithm, resampling_criterion, prior,
			transition_kernel, self._sensors) for t in topologies]

		PEs_sensors_requirements = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensorsPositions)

		# distributed particle filter
		self._distributedPFsForTopologies = [particle_filter.TargetTrackingParticleFilterWithDRNA(
			self._DRNAsettings["exchange period"], e, self._K, self._DRNAsettings["normalization period"],
			resampling_algorithm,resampling_criterion,prior,transition_kernel, self._sensors,
			PEs_sensors_requirements.getConnections(e.n_processing_elements) ) for e in exchange_recipes]

		#------------------------------------------------------------- metrics initialization --------------------------------------------------------------------

		# we store the aggregated weights...
		self._distributedPFaggregatedWeights = [np.empty((self._nTimeInstants,t.n_processing_elements,parameters["number of frames"])) for t in topologies]

		# ...and the position estimates
		self._centralizedPF_pos = np.empty((2, self._nTimeInstants, parameters["number of frames"], len(topologies)))
		self._distributedPF_pos = np.empty((2, self._nTimeInstants, parameters["number of frames"], len(topologies)))

		# HDF5

		# the names of the algorithms are also stored
		h5_algorithms_names = self._f.create_dataset(self._h5py_prefix + 'algorithms/names', shape=(2,), dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_names[0] = 'Centralized PF'
		h5_algorithms_names[1] = 'Distributed PF'

		# the colors
		h5_algorithms_colors = self._f.create_dataset(self._h5py_prefix + 'algorithms/plot/colors', shape=(2,), dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_colors[0] = self._painterSettings["color for the centralized PF"]
		h5_algorithms_colors[1] = self._painterSettings["color for the distributed PF"]

		# markers
		h5_algorithms_markers = self._f.create_dataset(self._h5py_prefix + 'algorithms/plot/markers', shape=(2,), dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_markers[0] = self._painterSettings["marker for the centralized PF"]
		h5_algorithms_markers[1] = self._painterSettings["marker for the distributed PF"]

		# saving of the aggregated weights upper bounds for each topology
		self._f.create_dataset(
			self._h5py_prefix + 'upper bounds for the aggregated weights', shape=(len(self._settings_topologies),),
			data=self._aggregatedWeightsUpperBounds)

	def save_data(self, target_position):

		# let the super class do its thing...
		super().save_data(target_position)

		# so that the last frame is also saved
		# FIXME: this method should only be called after completing a frame (never in the middle)
		self._iFrame += 1

		# the mean of the MSE incurred by both PFs
		centralizedPF_MSE = (np.subtract(self._centralizedPF_pos[:, :, :self._iFrame, :], target_position[:, :, :self._iFrame, np.newaxis])**2).mean(axis=0).mean(axis=1)
		distributedPF_MSE = (np.subtract(self._distributedPF_pos[:, :, :self._iFrame, :], target_position[:, :, :self._iFrame, np.newaxis])**2).mean(axis=0).mean(axis=1)

		# ...the same for the error (euclidean distance)
		centralizedPF_error = np.sqrt((np.subtract(self._centralizedPF_pos[:, :, :self._iFrame, :], target_position[:, :, :self._iFrame, np.newaxis])**2).sum(axis=0)).mean(axis=1)
		distributedPF_error = np.sqrt((np.subtract(self._distributedPF_pos[:, :, :self._iFrame, :], target_position[:, :, :self._iFrame, np.newaxis])**2).sum(axis=0)).mean(axis=1)

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
				targetPosition = target_position[:,:,:self._iFrame],
				centralizedPF_pos = self._centralizedPF_pos[:,:,:self._iFrame,:],
				distributedPF_pos = self._distributedPF_pos[:,:,:self._iFrame,:],
				**normalizedAggregatedWeightsDic
			)

		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**dataToBeSaved)
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))

		# the above fix is undone
		self._iFrame -= 1

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		for iTopology, (pf, distributed_pf) in enumerate(zip(self._PFsForTopologies, self._distributedPFsForTopologies)):

			n_PEs = self._settings_topologies[iTopology]['number of PEs']

			# the last dimension is for the number of algorithms (centralized and distributed)
			estimated_pos = np.full((state.n_elements_position, self._nTimeInstants, 2), np.nan)

			aggregated_weights = np.full((self._nTimeInstants, n_PEs), np.nan)

			# initialization of the particle filters
			pf.initialize()
			distributed_pf.initialize()

			if self._painterSettings['display evolution?']:

				# if this is not the first iteration...
				if hasattr(self, '_painter'):

					# ...then, the previous figure is closed
					self._painter.close()

				# this object will handle graphics...
				self._painter = plot.RectangularRoomPainter(
					self._roomSettings["bottom left corner"], self._roomSettings["top right corner"],
					self._sensorsPositions, sleepTime=self._painterSettings["sleep time between updates"])

				# ...e.g., draw the sensors
				self._painter.setup()

			for iTime in range(self._nTimeInstants):

				print('---------- iFrame = {}, iTopology = {}, iTime = {}'.format(repr(self._iFrame),repr(iTopology),repr(iTime)))

				print('position:\n',target_position[:, iTime:iTime+1])
				print('velocity:\n',target_velocity[:, iTime:iTime+1])

				# particle filters are updated
				pf.step(self._observations[iTime])
				distributed_pf.step(self._observations[iTime])

				# the mean computed by the centralized and distributed PFs
				centralizedPF_mean, distributedPF_mean = pf.compute_mean(),distributed_pf.compute_mean()

				estimated_pos[:, iTime:iTime+1, 0] = state.position(centralizedPF_mean)
				estimated_pos[:, iTime:iTime+1, 1] = state.position(distributedPF_mean)

				self._centralizedPF_pos[:, iTime:iTime+1, self._iFrame, iTopology] = state.position(centralizedPF_mean)
				self._distributedPF_pos[:, iTime:iTime+1, self._iFrame, iTopology] = state.position(distributedPF_mean)

				# the aggregated weights of the different PEs in the distributed PF are stored
				self._distributedPFaggregatedWeights[iTopology][iTime, :, self._iFrame] = distributed_pf.getAggregatedWeights()
				aggregated_weights[iTime, :] = distributed_pf.getAggregatedWeights()

				print('centralized PF\n',centralizedPF_mean)
				print('distributed PF\n',distributedPF_mean)

				if self._painterSettings["display evolution?"]:

					# the plot is updated with the position of the target...
					self._painter.updateTargetPosition(target_position[:,iTime:iTime+1])

					# ...those estimated by the PFs
					self._painter.updateEstimatedPosition(state.position(centralizedPF_mean),identifier='centralized',color=self._painterSettings["color for the centralized PF"])
					self._painter.updateEstimatedPosition(state.position(distributedPF_mean),identifier='distributed',color=self._painterSettings["color for the distributed PF"])

					if self._painterSettings["display particles evolution?"]:

						# ...and those of the particles...
						self._painter.updateParticlesPositions(state.position(pf.get_state()),identifier='centralized',color=self._painterSettings["color for the centralized PF"])
						self._painter.updateParticlesPositions(state.position(distributed_pf.get_state()),identifier='distributed',color=self._painterSettings["color for the distributed PF"])


			# data is saved
			h5_estimated_pos = self._h5_current_frame.create_dataset(
				'topology/{}/estimated position'.format(iTopology), shape=estimated_pos.shape, dtype=float,
				data=estimated_pos)

			h5_estimated_pos.attrs['M'] = n_PEs

			self._h5_current_frame.create_dataset(
				'topology/{}/DPF aggregated weights'.format(iTopology), aggregated_weights.shape, dtype=float,
				data=aggregated_weights)

class MultipleMposterior(Simulation):
	
	def __init__(self, parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs):
		
		# let the super class do its thing...
		super().__init__(parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs)
		
		# HDF5 output file
		self._f = h5py.File('res_' + self._outputFile + '.hdf5','w')
		
		# we will build several "Mposterior" objects...
		self._simulations = []
		
		# ...and each one will have a different set of sensors
		self._sensors = []
		
		# for every pair nPEs-nSensors we aim to simulate...
		for (nPEs, nSensors) in self._simulationParameters["nPEs-nSensors pairs"]:
			
			self._simulations.append(Mposterior(
				parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs,
				self._f, '{} PEs,{} sensors/'.format(nPEs, nSensors), nPEs, nSensors))

	def process_frame(self, target_position, target_velocity):
		
		# let the super class do its thing...
		super().process_frame(target_position,target_velocity)
		
		for sim in self._simulations:
		
			sim.process_frame(target_position,target_velocity)
		
	def save_data(self,target_position):
		
		# let the super class do its thing...
		super().save_data(target_position)
		
		self._f.close()


class Mposterior(SimpleSimulation):

	@staticmethod
	def parse_hdf5(data_file, prefix=''):

		h5_frames = data_file[prefix + '/frames']

		n_frames = len(h5_frames)

		any_frame = list(h5_frames.keys())[0]

		n_state, n_time_instants, n_algorithms = h5_frames['{}/estimated position'.format(any_frame)].shape

		estimated_position = np.empty((n_state, n_time_instants, n_algorithms, n_frames))
		actual_position = np.empty((n_state, n_time_instants, n_frames))

		for i_frame, frame in enumerate(h5_frames):

			estimated_position[..., i_frame] = h5_frames['{}/estimated position'.format(frame)]
			actual_position[...,i_frame] = h5_frames['{}/actual position'.format(frame)]

		return actual_position, estimated_position, n_frames
	
	# TODO: a method of the object is called from within "__init__" (allowed in python...but weird)
	
	def __init__(
			self, parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			PRNGs, h5py_file=None, h5py_prefix='', n_PEs=None, n_sensors=None):
		
		# let the super class do its thing...
		super().__init__(
			parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file, PRNGs,
			h5py_file, h5py_prefix, n_PEs, n_sensors)
		
		self._simulationParameters = parameters['simulations'][parameters['simulations']['type']]
		self._MposteriorSettings = parameters['Mposterior']
		self._LCDPFsettings = parameters['Likelihood Consensus']
		
		# if the number of PEs is not received...
		if n_PEs is None:
			# ...it is looked up in "parameters"
			self._nPEs = self._settings_topologies['number of PEs']
		# otherwise...
		else:
			self._nPEs = n_PEs
		
		# a connector that connects every sensor to every PE
		self._everySensorWithEveryPEConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensorsPositions)
		
		# the settings for the selected "sensors-PES connector"
		settings_sensors_PEs_connector = parameters['sensors-PEs connectors'][self._simulationParameters['sensors-PEs connector']]
		
		# the positions of the PEs are added as a parameters...
		# technically they are "derived" parameters since they are completely determined by:
		# 	- the corners of the room
		# 	- the positions of the sensors which, in turn, also depend on the corners of the room and the number of sensors
		# 	- the number of PEs
		self._settings_topologies['parameters']['PEs positions'] = self._PEsPositions
		
		# ...are used to build a connector, from which the links between PEs and sensors are obtained
		self._PEsSensorsConnections = getattr(sensors_PEs_connector,settings_sensors_PEs_connector['implementing class'])(
			self._sensorsPositions, self._PEsPositions, settings_sensors_PEs_connector['parameters']).getConnections(self._nPEs)

		# network topology, which describes the connection among PEs, as well as the exact particles exchanged/shared
		self._PEsTopology = getattr(PEs_topology, self._settings_topologies['implementing class'])(
			self._nPEs, self._settings_topologies['parameters'])

		# ...are plot the connections between them
		sensors_network_plot = plot.TightRectangularRoomPainterWithPEs(
			self._roomSettings["bottom left corner"], self._roomSettings["top right corner"], self._sensorsPositions,
			self._PEsPositions,self._PEsSensorsConnections, self._PEsTopology.get_neighbours(),
			sleepTime=self._painterSettings["sleep time between updates"])

		sensors_network_plot.setup()
		sensors_network_plot.save(outputFile='network_topology_{}_PEs.pdf'.format(self._nPEs))
		
		# the lists of PFs, estimators, colors and labels are initialized...
		self._PFs = []
		self._estimators = []
		self._estimatorsColors = []
		self._estimatorsLabels = []
		
		# ...and algorithms are added
		self.add_algorithms()
		
		# the position estimates
		self._estimatedPos = np.empty((2,self._nTimeInstants,parameters["number of frames"], len(self._estimators)))
		
		assert len(self._estimatorsColors) == len(self._estimatorsLabels) == len(self._estimators)
		
		# information about the simulated algorithms is added to the parameters...
		parameters['algorithms'] = [{'name': name, 'color': color} for name, color in zip(
			self._estimatorsLabels, self._estimatorsColors)]
		
		# HDF5

		# the names of the algorithms are also stored
		h5algorithms = self._f.create_dataset(
			self._h5py_prefix + 'algorithms/names', shape=(len(self._estimators),), dtype=h5py.special_dtype(vlen=str))
		for il, l in enumerate(self._estimatorsLabels):
			h5algorithms[il] = l
		
		# the position and connected sensors of each PE
		for iPE, (pos, sens) in enumerate(zip(self._PEsPositions.T, self._PEsSensorsConnections)):
			self._f.create_dataset(self._h5py_prefix + 'PEs/{}/position'.format(iPE), shape=(2,), data=pos)
			self._f.create_dataset(
				self._h5py_prefix + 'PEs/{}/connected sensors'.format(iPE), shape=(len(sens),), data=sens)
		
		# the positions of the sensors
		self._f.create_dataset(
			self._h5py_prefix + 'sensors/positions', shape=self._sensorsPositions.shape, data=self._sensorsPositions)

		# a list with the messages required by each estimator at a single time instant
		algorithms_messages = []

		for estimator, label in zip(self._estimators, self._estimatorsLabels):

			messages_during_estimation = estimator.messages(self._PEsTopology)

			messages_algorithm_operation = estimator.DPF.messages(self._PEsTopology,self._PEsSensorsConnections)

			algorithms_messages.append(messages_during_estimation+messages_algorithm_operation)

			print('{}: messages = {}'.format(label,algorithms_messages[-1]))

		# the messages (per iteration) required by each algorithm
		self._f.create_dataset(
			self._h5py_prefix + 'algorithms/messages', shape=(len(algorithms_messages),), data=algorithms_messages)
	
	def add_algorithms(self):
		
		"""Adds the algorithms to be tested by this simulation, defining the required parameters.
		
		"""

		DRNA_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(
			self._PEsTopology, self._K, self._simulationParameters["exchanged particles"],
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		Mposterior_exchange_recipe = smc.exchange_recipe.IteratedMposteriorExchangeRecipe(
			self._PEsTopology, self._K, self._simulationParameters["exchanged particles"],
			self._MposteriorSettings["number of iterations"], PRNG=self._PRNGs["topology pseudo random numbers generator"])

		# ------------

		for n_consensus_iter, color in zip(
				[self._LCDPFsettings['number of consensus iterations'], 10], ['brown', 'yellowgreen']):

			likelihood_consensus_exchange_recipe = smc.exchange_recipe.LikelihoodConsensusExchangeRecipe(
				self._PEsTopology, n_consensus_iter,
				self._LCDPFsettings['degree of the polynomial approximation'])

			# consensus
			self._PFs.append(
				particle_filter.LikelihoodConsensusDistributedTargetTrackingParticleFilter(
					likelihood_consensus_exchange_recipe, self._nPEs, self._K, self._resamplingAlgorithm,
					self._resamplingCriterion, self._prior, self._transitionKernel, self._sensors, self._PEsSensorsConnections,
					self._LCDPFsettings['degree of the polynomial approximation'],
					PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilterWithConsensusCapabilities
					)
			)

			# the estimator just delegates the calculus of the estimate to one of the PEs
			self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1], 0))

			self._estimatorsColors.append(color)
			self._estimatorsLabels.append('LC DPF with {} iterations'.format(n_consensus_iter))
		
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
				self._K*self._nPEs, self._resamplingAlgorithm, self._resamplingCriterion, self._prior,
				self._transitionKernel, self._sensors
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
				self._DRNAsettings["exchange period"], DRNA_exchange_recipe, self._K,
				self._DRNAsettings["normalization period"], self._resamplingAlgorithm, self._resamplingCriterion,
				self._prior, self._transitionKernel, self._sensors,
				self._everySensorWithEveryPEConnector.getConnections(self._nPEs),
				PFs_class=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
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
				self._DRNAsettings["exchange period"], DRNA_exchange_recipe, self._K,
				self._DRNAsettings["normalization period"], self._resamplingAlgorithm, self._resamplingCriterion,
				self._prior, self._transitionKernel, self._sensors, self._PEsSensorsConnections,
				PFs_class=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
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
				self._nPEs, self._K, self._resamplingAlgorithm, self._resamplingCriterion, self._prior, self._transitionKernel,
				self._sensors, self._PEsSensorsConnections, PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilter
			)
		)

		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(
			self._PFs[-1], max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
			tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

		self._estimatorsColors.append('seagreen')
		self._estimatorsLabels.append('Plain DPF (geometric median with 1 particle from each PE)')
		
		# ------------

		# DPF with M-posterior-based exchange
		self._PFs.append(
			smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
				Mposterior_exchange_recipe, self._K, self._resamplingAlgorithm, self._resamplingCriterion, self._prior,
				self._transitionKernel, self._sensors, self._PEsSensorsConnections,
				self._MposteriorSettings['findWeiszfeldMedian parameters'], self._MposteriorSettings['sharing period'],
				PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilter)
		)
		
		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(
			self._PFs[-1], max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
			tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

		self._estimatorsColors.append('green')
		self._estimatorsLabels.append('M-posterior (geometric median with 1 particle from each PE)')
		
		# ------------

		# DPF with M-posterior-based exchange that gets its estimates from the geometric median
		#  of the particles in the first PE
		iPE = 0

		for radius, color in zip([1, 4, 5], ['deeppink', 'coral', 'orchid']):

			# an estimator which yields the geometric median of the particles in the "iPE"-th PE
			self._estimators.append(
				smc.estimator.SinglePEGeometricMedianWithinRadius(self._PFs[-1], iPE, self._PEsTopology, radius))

			self._estimatorsColors.append(color)
			self._estimatorsLabels.append(
				'M-posterior ({} hops geometric median with particles from PE \#{})'.format(radius, iPE))
		
		# ------------

		for i_PE in range(self._nPEs):

			# an estimator which yields the mean of the particles in the "iPE"-th PE
			self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1], i_PE))

			self._estimatorsColors.append('olive')
			self._estimatorsLabels.append('M-posterior (mean with particles from PE \#{})'.format(i_PE))

		# ------------

		for radius, color, exchange in zip(
				[2, 2], ['blue', 'red'], [self._simulationParameters["exchanged particles"], 10]):

			Mposterior_within_radius_exchange_recipe = smc.exchange_recipe.MposteriorWithinRadiusExchangeRecipe(
			self._PEsTopology, self._K, exchange, radius,
				PRNG=self._PRNGs["topology pseudo random numbers generator"])

			# DPF with M-posterior-based exchange, using a certain radius
			self._PFs.append(
				smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
					Mposterior_within_radius_exchange_recipe, self._K, self._resamplingAlgorithm, self._resamplingCriterion,
					self._prior, self._transitionKernel, self._sensors, self._PEsSensorsConnections,
					self._MposteriorSettings['findWeiszfeldMedian parameters'], self._MposteriorSettings['sharing period'],
					PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilter)
			)

			# an estimator computing the geometric median with 1 particle taken from each PE
			self._estimators.append(smc.estimator.GeometricMedian(
				self._PFs[-1], max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
				tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

			self._estimatorsColors.append(color)
			self._estimatorsLabels.append(
				'M-posterior - depth {} exchanging {}'.format(radius,exchange))

			for estimator_radius, color in zip([1, 4, 5], ['rosybrown', 'khaki', 'darkred']):

				# an estimator which yields the geometric median of the particles in the "iPE"-th PE
				self._estimators.append(
					smc.estimator.SinglePEGeometricMedianWithinRadius(self._PFs[-1], iPE, self._PEsTopology, estimator_radius))

				self._estimatorsColors.append(color)
				self._estimatorsLabels.append(
					'M-posterior - depth {} exchanging {} ({} hops)'.format(radius, exchange, estimator_radius))

		# ------------

	def save_data(self, target_position):
		
		# let the super class do its thing...
		super().save_data(target_position)
		
		# a dictionary encompassing all the data to be saved
		dataToBeSaved = dict(
				targetPosition = target_position[:, :, :self._iFrame],
				PF_pos = self._estimatedPos[:, :, :self._iFrame, :]
			)
		
		# data is saved
		scipy.io.savemat('res_' + self._outputFile,dataToBeSaved)
		print('results saved in "{}"'.format('res_' + self._outputFile))
		
		# the mean of the error (euclidean distance) incurred by the PFs
		PF_error = np.sqrt(
			(np.subtract(self._estimatedPos[:, :, :self._iFrame, :], target_position[:, :, :self._iFrame, np.newaxis])**2)
				.sum(axis=0)).mean(axis=1)
		
		plot.PFs(
			range(self._nTimeInstants),PF_error,
			self._simulationParameters["file name prefix for the estimation error vs time plot"] +
			'_' + self._outputFile + '_nFrames={}.eps'.format(repr(self._iFrame)),
			[{'label':l,'color':c} for l,c in zip(self._estimatorsLabels,self._estimatorsColors)])
		
		print(self._estimatedPos)
		
	def process_frame(self, target_position, target_velocity):
		
		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		# this array will store the results before they are saved
		estimated_pos = np.full((state.n_elements_position, self._nTimeInstants, len(self._estimators)), np.nan)

		# for every PF (different from estimator)...
		for pf in self._PFs:

			# ...initialization
			pf.initialize()

		if self._painterSettings['display evolution?']:

			# if this is not the first iteration...
			if hasattr(self, '_painter'):

				# ...then, the previous figure is closed
				self._painter.close()

			# this object will handle graphics...
			self._painter = plot.RectangularRoomPainter(
				self._roomSettings["bottom left corner"], self._roomSettings["top right corner"], self._sensorsPositions,
				sleepTime=self._painterSettings["sleep time between updates"])

			# ...e.g., draw the self._sensors
			self._painter.setup()

		for iTime in range(self._nTimeInstants):

			print('---------- iFrame = {}, iTime = {}'.format(repr(self._iFrame),repr(iTime)))

			print('position:\n',target_position[:, iTime:iTime+1])
			print('velocity:\n',target_velocity[:, iTime:iTime+1])

			# for every PF (different from estimator)...
			for pf in self._PFs:

				# ...a step is taken
				pf.step(self._observations[iTime])

			# for every estimator, along with its corresponding label,...
			for iEstimator, (estimator, label) in enumerate(zip(self._estimators, self._estimatorsLabels)):

				self._estimatedPos[:, iTime:iTime+1, self._iFrame, iEstimator] = state.position(estimator.estimate())

				# the position given by this estimator at the current time instant is written to the HDF5 file
				estimated_pos[:, iTime:iTime+1, iEstimator] = state.position(estimator.estimate())

				print('position estimated by {}\n'.format(label), self._estimatedPos[:, iTime:iTime+1, self._iFrame, iEstimator])

			if self._painterSettings["display evolution?"]:

				# the plot is updated with the position of the target...
				self._painter.updateTargetPosition(target_position[:, iTime:iTime+1])

				# ...those estimated by the PFs
				for iEstimator,(pf,color) in enumerate(zip(self._estimators, self._estimatorsColors)):

					self._painter.updateEstimatedPosition(
						self._estimatedPos[:, iTime:iTime+1, self._iFrame, iEstimator],
						identifier='#{}'.format(iEstimator), color=color)

					if self._painterSettings["display particles evolution?"]:

						self._painter.updateParticlesPositions(
							state.position(pf.get_state()), identifier='#{}'.format(iEstimator), color=color)

		# the results (estimated positions) are saved
		self._h5_current_frame.create_dataset(
			'estimated position', shape=estimated_pos.shape, dtype=float, data=estimated_pos)

		# in order to make sure the HDF5 files is valid...
		self._f.flush()


class MposteriorExchange(Mposterior):
		
	def add_algorithms(self):

		# available colors
		colors = ['red','blue','green','goldenrod','cyan','crimson','lime','cadetblue','magenta']

		# topology of the network
		topology = getattr(PEs_topology,self._settings_topologies['implementing class'])(self._nPEs,self._settings_topologies['parameters'])

		for iPercentage,(percentage,color) in enumerate(zip(self._simulationParameters["exchanged particles"],colors)):

			DRNA_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(topology,self._K,percentage,PRNG=self._PRNGs["topology pseudo random numbers generator"])

			# a distributed PF with DRNA
			self._PFs.append(
				smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
					self._DRNAsettings["exchange period"],DRNA_exchange_recipe,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
					self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),
					PFs_class=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
				)
			)

			self._estimators.append(smc.estimator.Mean(self._PFs[-1]))

			self._estimatorsColors.append('black')
			self._estimatorsLabels.append('DRNA exchanging {}'.format(percentage))

			# ------------

			Mposterior_exchange_recipe = smc.exchange_recipe.MposteriorExchangeRecipe(
				self._PEsTopology, self._K, percentage, PRNG=self._PRNGs["topology pseudo random numbers generator"])

			self._PFs.append(
				smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
					Mposterior_exchange_recipe,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
					self._sensors,self._PEsSensorsConnections,self._MposteriorSettings['findWeiszfeldMedian parameters'],
					self._MposteriorSettings['sharing period'], PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilter
				)
			)

			self._estimators.append(
				smc.estimator.GeometricMedian(self._PFs[-1],
				max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
				tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

			self._estimatorsColors.append(color)
			self._estimatorsLabels.append('M-posterior {}'.format(percentage))

	def save_data(self,target_position):
		
		# the method from the grandparent
		SimpleSimulation.save_data(self,target_position)

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
			for i, exchanged_particles in enumerate(self._simulationParameters["exchanged particles"]):

				# ...the results for all the algorithms are stored in the appropriate place
				frame['estimated position/exchanged particles/{}'.format(exchanged_particles)] = frame['aux'][...,i*2:(i+1)*2]

			# we don't need this anymore
			del frame['aux']

		# if a reference to an HDF5 was not received, that means the file was created by this object, and hence it is responsibility to close it...
		if self._h5py_file is None:

			# ...in order to make sure the HDF5 file is valid...
			self._f.close()


class MposteriorGeometricMedian(Mposterior):
	
	def add_algorithms(self):
		
		# a copy of the required PRNG is built...so that the exchange particles map is the same for both DRNA and Mposterior
		# TODO: is this really necesary? a better approach?
		import copy
		PRNGcopy = copy.deepcopy(self._PRNGs["topology pseudo random numbers generator"])

		DRNA_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(self._PEsTopology, self._K, self._simulationParameters["exchanged particles"], PRNG=self._PRNGs["topology pseudo random numbers generator"])
		Mposterior_exchange_recipe = smc.exchange_recipe.MposteriorExchangeRecipe(self._PEsTopology,self._K,self._simulationParameters["exchanged particles"], PRNG=PRNGcopy)
		
		# a distributed PF with DRNA
		self._PFs.append(
			smc.particle_filter.TargetTrackingParticleFilterWithDRNA(
				self._DRNAsettings["exchange period"],DRNA_exchange_recipe,self._K,self._DRNAsettings["normalization period"],self._resamplingAlgorithm,self._resamplingCriterion,
				self._prior,self._transitionKernel,self._sensors,self._everySensorWithEveryPEConnector.getConnections(self._nPEs),
				PFs_class=smc.particle_filter.EmbeddedTargetTrackingParticleFilter
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
				self._MposteriorSettings['sharing period'], PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilter,
			)
		)
		
		for nParticles,col in zip(self._simulationParameters['number of particles for estimation'],colors):
			
			self._estimators.append(smc.estimator.StochasticGeometricMedian(
				self._PFs[-1],nParticles,
				max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
			
			self._estimatorsColors.append(col)
			self._estimatorsLabels.append('M-posterior (Stochastic Geometric Median with {} particles from each PE)'.format(nParticles))


class MposteriorIterative(Mposterior):

	def add_algorithms(self):

		colors = ['red','blue','green','goldenrod','cyan','crimson','lime','cadetblue','magenta']

		for n_iterations,color in zip(self._simulationParameters["number of iterations"],colors):

			exchange_recipe = smc.exchange_recipe.IteratedMposteriorExchangeRecipe(
				self._PEsTopology, self._K, self._simulationParameters["exchanged particles"],
				n_iterations, PRNG=self._PRNGs["topology pseudo random numbers generator"])

			self._PFs.append(
				smc.particle_filter.DistributedTargetTrackingParticleFilterWithMposterior(
					exchange_recipe,self._K,self._resamplingAlgorithm,self._resamplingCriterion,self._prior,self._transitionKernel,
					self._sensors,self._PEsSensorsConnections,self._MposteriorSettings['findWeiszfeldMedian parameters'],
					self._MposteriorSettings['sharing period'], PFs_class=smc.particle_filter.CentralizedTargetTrackingParticleFilter,
				)
			)

			self._estimators.append(smc.estimator.GeometricMedian(self._PFs[-1], max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
															tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))
			self._estimatorsColors.append(color)
			self._estimatorsLabels.append('{} iterations'.format(n_iterations))