import abc
import numpy as np
import scipy.io
import colorama
import h5py
import re

import smc.particle_filter.centralized as centralized
import smc.particle_filter.distributed as distributed
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
	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators):

		self._parameters = parameters
		self._room = room
		
		# these parameters are kept for later use
		self._resampling_algorithm = resampling_algorithm
		self._resampling_criterion = resampling_criterion
		self._prior = prior
		self._transition_kernel = transition_kernel
		self._PRNGs = pseudo_random_numbers_generators
		
		# number of particles per processing element (PE)
		self._n_particles_per_PE = parameters["number of particles per PE"]
		
		# length of the trajectory
		self._n_time_instants = parameters["number of time instants"]
		
		# name of the file to store the results
		self._output_file = output_file
		
		# DRNA related
		self._settings_DRNA = parameters["DRNA"]
		
		# parameters related to plotting
		self._settings_painter = parameters["painter"]
		
		# room  dimensions
		self._settings_room = parameters["room"]
		
		# the settings for the topology or topologies given...if it is a list...
		if isinstance(parameters['topologies types'], list):
			# ...we have a list of settings
			self._settings_topologies = [parameters['topologies'][i] for i in parameters['topologies types']]
		# otherwise...
		else:
			# the "topology settings" object is just a dictionary
			self._settings_topologies = parameters['topologies'][parameters['topologies types']]
		
		# so that it equals 0 the first time it is incremented...
		self._i_current_frame = -1
		
		# the parameters for this particular simulation are obtained
		self._simulation_parameters = parameters['simulations'][parameters['simulation type']]
		
	@abc.abstractmethod
	def process_frame(self, target_position, target_velocity):
		
		self._i_current_frame += 1
	
	# TODO: remove target_position as argument?
	
	@abc.abstractmethod
	def save_data(self, target_position):

		if self._i_current_frame == 0:
			print('save_data: still in the first frame...maybe nothing will be saved')


class SimpleSimulation(Simulation):
	
	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements=None, n_sensors=None):
		
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators)

		# for saving the data in HDF5
		self._h5py_file = h5py_file
		self._h5py_prefix = h5py_prefix

		# if a reference to an HDF5 file was not received...
		if h5py_file is None:

			# ...a new HDF5 file is created
			self._f = h5py.File('res_' + self._output_file + '.hdf5', 'w', driver='core', libver='latest')

		# otherwise...
		else:

			# the value received is assumed to be a reference to an already open file
			self._f = self._h5py_file

		# for the sake of convenience
		sensors_settings = parameters["sensors"]

		if n_sensors is None:
			
			n_sensors = parameters["number of sensors"]
		
		if n_processing_elements is None:

			# we try to extract the "number of PEs" from the topology settings...
			try:
				
				n_processing_elements = self._settings_topologies['number of PEs']
			
			# ...it it's not possible, it is because there are multiple topology settings => Convergence simulation
			except TypeError:
				
				# n_PEs = None
				pass
		
		# for the sake of convenience below...
		network_nodes_settings = parameters['network nodes'][self._simulation_parameters['network']]
		network_nodes_class = getattr(network_nodes, network_nodes_settings['implementing class'])
		
		# the appropriate class is instantiated with the given parameters
		self._network = network_nodes_class(
			self._settings_room["bottom left corner"], self._settings_room["top right corner"], n_processing_elements,
			n_sensors, **network_nodes_settings['parameters'])
		
		# the positions of the PEs and the sensors are collected from the network just built
		self._sensorsPositions, self._PEsPositions = self._network.sensors_positions, self._network.PEs_positions
		
		# the class to be instantiated is figured out from the settings for that particular sensor type
		sensor_class = getattr(sensor, sensors_settings[parameters['sensors type']]['implementing class'])
		
		self._sensors = [sensor_class(
			pos[:, np.newaxis],
			pseudo_random_numbers_generator=pseudo_random_numbers_generators[
				'Sensors and Monte Carlo pseudo random numbers generator'],
			**sensors_settings[parameters['sensors type']]['parameters']
		) for pos in self._sensorsPositions.T]

		self._f.create_dataset(
			self._h5py_prefix + 'room/bottom left corner',
			parameters['room']['bottom left corner'].shape, data=parameters['room']['bottom left corner'])

		self._f.create_dataset(
			self._h5py_prefix + 'room/top right corner',
			parameters['room']['top right corner'].shape, data=parameters['room']['top right corner'])

		# these are going to be set/used by other methods
		self._observations = None
		self._h5_current_frame = None
		self._painter = None
		
	def process_frame(self, target_position, target_velocity):
		
		super().process_frame(target_position, target_velocity)

		# a call to the method in charge of "filling" self._observations
		self.build_observations(target_position)
		
		# a reference to the "group" for the current frame (notice the prefix in the name given "self._h5py_prefix")...
		self._h5_current_frame = self._f.create_group(
			self._h5py_prefix + 'frames/{}'.format(self._i_current_frame))
		
		# ...where a new dataset is created for the "actual position" of the target...
		self._h5_current_frame.create_dataset(
			'actual position', shape=(2, self._n_time_instants), dtype=float, data=target_position)

	def build_observations(self, target_position):

		# observations for all the sensors at every time instant (each list)
		# REMARK: conversion to float is done so that the observations (when 1 or 0) are amenable to be used in later
		# computations
		self._observations = [np.array(
			[sens.detect(state.to_position(s[:, np.newaxis])) for sens in self._sensors], dtype=float
		) for s in target_position.T]

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
	def pseudo_random_numbers_generators_from_file(filename, i_frame=None):

		if i_frame:
			path_within_file = 'frames/{}/pseudo random numbers generators'.format(i_frame)
		else:
			path_within_file = 'pseudo random numbers generators'

		with h5py.File(filename, 'r') as data_file:

			prngs = data_file[path_within_file]

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
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file=None, h5py_prefix=''):

		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix)

		topologies = [getattr(PEs_topology, t['implementing class'])(
			t['number of PEs'], t['parameters']) for t in self._settings_topologies]

		exchange_recipes = [smc.exchange_recipe.DRNAExchangeRecipe(
			t, self._n_particles_per_PE, self._simulation_parameters["exchanged particles"],
			PRNG=self._PRNGs["topology pseudo random numbers generator"]) for t in topologies]

		# we compute the upper bound for the supremum of the aggregated weights that should guarantee convergence
		self._aggregatedWeightsUpperBounds = [drnautil.supremum_upper_bound(
			t['number of PEs'], self._settings_DRNA['c'], self._settings_DRNA['q'], self._settings_DRNA['epsilon']
		) for t in self._settings_topologies]

		# plain non-parallelized particle filter
		self._PFsForTopologies = [centralized.TargetTrackingParticleFilter(
			self._n_particles_per_PE*t.n_processing_elements, resampling_algorithm, resampling_criterion, prior,
			transition_kernel, self._sensors) for t in topologies]

		PEs_sensors_requirements = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensorsPositions)

		# distributed particle filter
		self._distributedPFsForTopologies = [distributed.TargetTrackingParticleFilterWithDRNA(
			self._settings_DRNA["exchange period"], e, self._n_particles_per_PE, self._settings_DRNA["normalization period"],
			resampling_algorithm, resampling_criterion, prior, transition_kernel, self._sensors,
			PEs_sensors_requirements.get_connections(e.n_processing_elements)) for e in exchange_recipes]

		# ------------------------------------------ metrics initialization --------------------------------------------

		# we store the aggregated weights...
		self._distributedPFaggregatedWeights = [np.empty(
			(self._n_time_instants, t.n_processing_elements, parameters["number of frames"])
		) for t in topologies]

		# ...and the position estimates
		self._centralizedPF_pos = np.empty((2, self._n_time_instants, parameters["number of frames"], len(topologies)))
		self._distributedPF_pos = np.empty((2, self._n_time_instants, parameters["number of frames"], len(topologies)))

		# HDF5

		# the names of the algorithms are also stored
		h5_algorithms_names = self._f.create_dataset(
			self._h5py_prefix + 'algorithms/names', shape=(2,), dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_names[0] = 'Centralized PF'
		h5_algorithms_names[1] = 'Distributed PF'

		# the colors
		h5_algorithms_colors = self._f.create_dataset(
			self._h5py_prefix + 'algorithms/plot/colors', shape=(2,), dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_colors[0] = self._settings_painter["color for the centralized PF"]
		h5_algorithms_colors[1] = self._settings_painter["color for the distributed PF"]

		# markers
		h5_algorithms_markers = self._f.create_dataset(
			self._h5py_prefix + 'algorithms/plot/markers', shape=(2,), dtype=h5py.special_dtype(vlen=str))
		h5_algorithms_markers[0] = self._settings_painter["marker for the centralized PF"]
		h5_algorithms_markers[1] = self._settings_painter["marker for the distributed PF"]

		# saving of the aggregated weights upper bounds for each topology
		self._f.create_dataset(
			self._h5py_prefix + 'upper bounds for the aggregated weights', shape=(len(self._settings_topologies),),
			data=self._aggregatedWeightsUpperBounds)

	def save_data(self, target_position):

		# let the super class do its thing...
		super().save_data(target_position)

		# so that the last frame is also saved
		# FIXME: this method should only be called after completing a frame (never in the middle)
		self._i_current_frame += 1

		# the mean of the MSE incurred by both PFs
		centralized_particle_filter_MSE = (
			(self._centralizedPF_pos[:, :, :self._i_current_frame, :] - target_position[:, :, :self._i_current_frame, np.newaxis])**2
		).mean(axis=0).mean(axis=1)
		distributed_particle_filter_MSE = (
			(self._distributedPF_pos[:, :, :self._i_current_frame, :] - target_position[:, :, :self._i_current_frame, np.newaxis])**2
		).mean(axis=0).mean(axis=1)

		# ...the same for the error (euclidean distance)
		centralizedPF_error = np.sqrt(
			((self._centralizedPF_pos[:, :, :self._i_current_frame, :] - target_position[:, :, :self._i_current_frame, np.newaxis])**2).sum(
				axis=0)).mean(axis=1)
		distributedPF_error = np.sqrt(
			((self._distributedPF_pos[:, :, :self._i_current_frame, :] - target_position[:, :, :self._i_current_frame, np.newaxis])**2).sum(
				axis=0)).mean(axis=1)

		# MSE vs time (only the results for the first topology are plotted)
		plot.distributed_against_centralized_particle_filter(
			np.arange(self._n_time_instants), centralized_particle_filter_MSE[:, 0], distributed_particle_filter_MSE[:, 0],
			output_file='{}_{}_nFrames={}.eps'.format(
				self._settings_painter["file name prefix for the MSE vs time plot"], self._output_file, repr(self._i_current_frame)
			), centralized_particle_filter_parameters={
				'label': 'Centralized PF', 'color': self._settings_painter["color for the centralized PF"],
				'marker': self._settings_painter["marker for the centralized PF"]
			}, distributed_particle_filter_parameters={
				'label': 'Distributed PF', 'color': self._settings_painter["color for the distributed PF"],
				'marker': self._settings_painter["marker for the distributed PF"]
			}, figure_id='MSE vs Time')

		# distance vs time (only the results for the first topology are plotted)
		plot.distributed_against_centralized_particle_filter(
			np.arange(self._n_time_instants), centralizedPF_error[:, 0], distributedPF_error[:, 0],
			output_file='{}_{}_nFrames={}.eps'.format(
				self._settings_painter["file name prefix for the euclidean distance vs time plot"], self._output_file,
				repr(self._i_current_frame)
			), centralized_particle_filter_parameters={
				'label': 'Centralized PF', 'color': self._settings_painter["color for the centralized PF"],
				'marker': self._settings_painter["marker for the centralized PF"]
			}, distributed_particle_filter_parameters={
				'label': 'Distributed PF', 'color': self._settings_painter["color for the distributed PF"],
				'marker': self._settings_painter["marker for the distributed PF"]
			}, figure_id='Euclidean distance vs Time')

		# the aggregated weights are normalized at ALL TIMES, for EVERY frame and EVERY topology
		normalized_aggregated_weights = [
			w[:, :, :self._i_current_frame] / w[:, :, :self._i_current_frame].sum(axis=1)[:, np.newaxis, :]
			for w in self._distributedPFaggregatedWeights]

		# ...the same data structured in a dictionary
		dic_normalized_aggregated_weights = {
			'normalizedAggregatedWeights_{}'.format(i): array for i, array in enumerate(normalized_aggregated_weights)}

		# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
		max_weights = np.array(
			[(w.max(axis=1)**self._settings_DRNA['q']).mean(axis=1) for w in normalized_aggregated_weights])

		# evolution of the largest aggregated weight over time (only the results for the first topology are plotted)
		plot.aggregated_weights_supremum_vs_time(
			max_weights[0, :], self._aggregatedWeightsUpperBounds[0], '{}_{}_nFrames={}.eps'.format(
				self._settings_painter["file name prefix for the aggregated weights supremum vs time plot"],
				self._output_file, repr(self._i_current_frame)
			), self._settings_DRNA["exchange period"])

		# a dictionary encompassing all the data to be saved
		data_to_be_saved = dict(
				aggregatedWeightsUpperBounds=self._aggregatedWeightsUpperBounds,
				targetPosition=target_position[:, :, :self._i_current_frame],
				centralizedPF_pos=self._centralizedPF_pos[:, :, :self._i_current_frame, :],
				distributedPF_pos=self._distributedPF_pos[:, :, :self._i_current_frame, :],
				**dic_normalized_aggregated_weights
			)

		# data is saved
		#np.savez('res_' + self._outputFile + '.npz',**data_to_be_saved)
		scipy.io.savemat('res_' + self._output_file, data_to_be_saved)
		print('results saved in "{}"'.format('res_' + self._output_file))

		# the above fix is undone
		self._i_current_frame -= 1

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		for iTopology, (pf, distributed_pf) in enumerate(zip(self._PFsForTopologies, self._distributedPFsForTopologies)):

			n_PEs = self._settings_topologies[iTopology]['number of PEs']

			# the last dimension is for the number of algorithms (centralized and distributed)
			estimated_pos = np.full((state.n_elements_position, self._n_time_instants, 2), np.nan)

			aggregated_weights = np.full((self._n_time_instants, n_PEs), np.nan)

			# initialization of the particle filters
			pf.initialize()
			distributed_pf.initialize()

			if self._settings_painter['display evolution?'] and self._settings_painter["use display server if available?"]:

				# if this is not the first frame...
				if self._painter:

					# ...then, the previous figure is closed
					self._painter.close()

				# this object will handle graphics...
				self._painter = plot.RectangularRoomPainter(
					self._settings_room["bottom left corner"], self._settings_room["top right corner"],
					self._sensorsPositions, sleepTime=self._settings_painter["sleep time between updates"])

				# ...e.g., draw the sensors
				self._painter.setup()

			for iTime in range(self._n_time_instants):

				print('---------- iFrame = {}, iTopology = {}, iTime = {}'.format(self._i_current_frame, iTopology, iTime))

				print('position:\n', target_position[:, iTime:iTime+1])
				print('velocity:\n', target_velocity[:, iTime:iTime+1])

				# particle filters are updated
				pf.step(self._observations[iTime])
				distributed_pf.step(self._observations[iTime])

				# the mean computed by the centralized and distributed PFs
				centralizedPF_mean, distributedPF_mean = pf.compute_mean(), distributed_pf.compute_mean()

				estimated_pos[:, iTime:iTime+1, 0] = state.to_position(centralizedPF_mean)
				estimated_pos[:, iTime:iTime+1, 1] = state.to_position(distributedPF_mean)

				self._centralizedPF_pos[:, iTime:iTime+1, self._i_current_frame, iTopology] = state.to_position(centralizedPF_mean)
				self._distributedPF_pos[:, iTime:iTime+1, self._i_current_frame, iTopology] = state.to_position(distributedPF_mean)

				# the aggregated weights of the different PEs in the distributed PF are stored
				self._distributedPFaggregatedWeights[iTopology][iTime, :, self._i_current_frame] = distributed_pf.aggregated_weights
				aggregated_weights[iTime, :] = distributed_pf.aggregated_weights

				print('centralized PF\n', centralizedPF_mean)
				print('distributed PF\n', distributedPF_mean)

				if self._settings_painter["display evolution?"] and self._settings_painter["use display server if available?"]:

					# the plot is updated with the position of the target...
					self._painter.updateTargetPosition(target_position[:, iTime:iTime+1])

					# ...those estimated by the PFs
					self._painter.updateEstimatedPosition(state.to_position(centralizedPF_mean), identifier='centralized', color=self._settings_painter["color for the centralized PF"])
					self._painter.updateEstimatedPosition(state.to_position(distributedPF_mean), identifier='distributed', color=self._settings_painter["color for the distributed PF"])

					if self._settings_painter["display particles evolution?"]:

						# ...and those of the particles...
						self._painter.updateParticlesPositions(state.to_position(pf.get_state()), identifier='centralized', color=self._settings_painter["color for the centralized PF"])
						self._painter.updateParticlesPositions(state.to_position(distributed_pf.get_state()), identifier='distributed', color=self._settings_painter["color for the distributed PF"])

			# data is saved
			h5_estimated_pos = self._h5_current_frame.create_dataset(
				'topology/{}/estimated position'.format(iTopology), shape=estimated_pos.shape, dtype=float,
				data=estimated_pos)

			h5_estimated_pos.attrs['M'] = n_PEs

			self._h5_current_frame.create_dataset(
				'topology/{}/DPF aggregated weights'.format(iTopology), aggregated_weights.shape, dtype=float,
				data=aggregated_weights)


class MultipleMposterior(Simulation):
	
	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators):
		
		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators)
		
		# HDF5 output file
		self._f = h5py.File('res_' + self._output_file + '.hdf5', 'w')
		
		# we will build several "Mposterior" objects...
		self._simulations = []
		
		# ...and each one will have a different set of sensors
		self._sensors = []
		
		# for every pair nPEs-nSensors we aim to simulate...
		for (nPEs, nSensors) in self._simulation_parameters["nPEs-nSensors pairs"]:
			
			self._simulations.append(Mposterior(
				parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
				pseudo_random_numbers_generators, self._f, '{} PEs,{} sensors/'.format(nPEs, nSensors), nPEs, nSensors))

	def process_frame(self, target_position, target_velocity):
		
		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)
		
		for sim in self._simulations:
		
			sim.process_frame(target_position, target_velocity)
		
	def save_data(self, target_position):
		
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
			actual_position[..., i_frame] = h5_frames['{}/actual position'.format(frame)]

		return actual_position, estimated_position, n_frames
	
	# TODO: a method of the object is called from within "__init__" (allowed in python...but weird)
	
	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='', n_processing_elements=None, n_sensors=None):
		
		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements, n_sensors)
		
		self._MposteriorSettings = parameters['Mposterior']
		self._LCDPFsettings = parameters['Likelihood Consensus']
		
		# if the number of PEs is not received...
		if n_processing_elements is None:
			# ...it is looked up in "parameters"
			self._nPEs = self._settings_topologies['number of PEs']
		# otherwise...
		else:
			self._nPEs = n_processing_elements
		
		# a connector that connects every sensor to every PE
		self._everySensorWithEveryPEConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensorsPositions)
		
		# the settings for the selected "sensors-PES connector"
		settings_sensors_PEs_connector = parameters['sensors-PEs connectors'][
			self._simulation_parameters['sensors-PEs connector']]
		
		# ...are used to build a connector, from which the links between PEs and sensors are obtained
		self._PEsSensorsConnections = getattr(sensors_PEs_connector, settings_sensors_PEs_connector['implementing class'])(
			self._sensorsPositions, self._PEsPositions, settings_sensors_PEs_connector['parameters']).get_connections(self._nPEs)

		# network topology, which describes the connection among PEs, as well as the exact particles exchanged/shared
		self._PEsTopology = getattr(PEs_topology, self._settings_topologies['implementing class'])(
			self._nPEs, self._settings_topologies['parameters'], PEs_positions=self._PEsPositions)

		# the maximum number of hops between any pair of PEs
		max_hops_number = self._PEsTopology.distances_between_processing_elements.max()

		print('maximum number of hops between PEs = {}'.format(max_hops_number))

		# ...are plot the connections between them
		sensors_network_plot = plot.TightRectangularRoomPainterWithPEs(
			self._settings_room["bottom left corner"], self._settings_room["top right corner"], self._sensorsPositions,
			self._PEsPositions, self._PEsSensorsConnections, self._PEsTopology.get_neighbours(),
			sleepTime=self._settings_painter["sleep time between updates"])

		sensors_network_plot.setup()
		sensors_network_plot.save(outputFile='network_topology_{}_PEs.pdf'.format(self._nPEs))

		# ------------------ parameters to be used by add_algorithms

		# this is initialized here so that its value can be manipulated by subclasses
		self._exchanged_particles = self._simulation_parameters["exchanged particles"]

		# the maximum number of hops that a PE can be to exchange particles with a given PE
		self._mposterior_exchange_step_depth = parameters["Mposterior"]["sharing step depth"]

		# the same at estimation time
		self._mposterior_estimator_radius = [parameters["Mposterior"]["estimation step radius"]]

		self._mposterior_exchange_particles_more_than_once = parameters["Mposterior"][
			"allow sharing each particle more than once"]

		# DPF with M-posterior-based exchange gets its estimates from this PE
		self._i_PE_estimation = self._simulation_parameters["index of reference PE"]

		# number of particles gathered from each PE within the estimation radius
		self._mposterior_n_part_estimation = parameters["Mposterior"]["number of particles from each PE for estimation"]

		# -----------------------------------------------------------
		
		# the lists of PFs, estimators, colors and labels are initialized...
		self._PFs = []
		self._estimators = []
		self._estimators_colors = []
		self._estimators_labels = []

		# ================================== ...and algorithms are added =============================

		self.add_algorithms()

		# ============================================================================================

		# the position estimates
		self._estimated_pos = np.empty((2, self._n_time_instants, parameters["number of frames"], len(self._estimators)))
		
		assert len(self._estimators_colors) == len(self._estimators_labels) == len(self._estimators)
		
		# information about the simulated algorithms is added to the parameters...
		parameters['algorithms'] = [{'name': name, 'color': color} for name, color in zip(
			self._estimators_labels, self._estimators_colors)]
		
		# HDF5

		# the names of the algorithms are also stored
		h5algorithms = self._f.create_dataset(
			self._h5py_prefix + 'algorithms/names', shape=(len(self._estimators),), dtype=h5py.special_dtype(vlen=str))
		for il, l in enumerate(self._estimators_labels):
			h5algorithms[il] = l
		
		# the position, connected sensors, and neighbours of each PE
		for iPE, (pos, sens, neighbours) in enumerate(zip(
				self._PEsPositions.T, self._PEsSensorsConnections, self._PEsTopology.get_neighbours())):
			self._f.create_dataset(self._h5py_prefix + 'PEs/{}/position'.format(iPE), shape=(2,), data=pos)
			self._f.create_dataset(
				self._h5py_prefix + 'PEs/{}/connected sensors'.format(iPE), shape=(len(sens),), data=sens)
			self._f.create_dataset(
				self._h5py_prefix + 'PEs/{}/neighbours'.format(iPE), shape=(len(neighbours),), data=neighbours)

		self._f[self._h5py_prefix + 'PEs'].attrs['max number of hops'] = max_hops_number

		# the positions of the sensors
		self._f.create_dataset(
			self._h5py_prefix + 'sensors/positions', shape=self._sensorsPositions.shape, data=self._sensorsPositions)

		# a list with the messages required by every algorithm at a single time instant
		algorithms_messages = []

		for estimator, label in zip(self._estimators, self._estimators_labels):

			# number of messages due to the particular estimator used
			messages_during_estimation = estimator.messages(self._PEsTopology)

			# number of messages related to the algorithm
			messages_algorithm_operation = estimator.DPF.messages(self._PEsTopology, self._PEsSensorsConnections)

			algorithms_messages.append(messages_during_estimation+messages_algorithm_operation)

			print(colorama.Fore.GREEN + '{}'.format(label) + colorama.Style.RESET_ALL +
			      ': messages = {}'.format(algorithms_messages[-1]))

		# the messages (per iteration) required by each algorithm
		self._f.create_dataset(
			self._h5py_prefix + 'algorithms/messages', shape=(len(algorithms_messages),), data=algorithms_messages)
	
	def add_algorithms(self):
		
		"""Adds the algorithms to be tested by this simulation, defining the required parameters.
		
		"""

		drna_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(
			self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		# mposterior_exchange_recipe = smc.exchange_recipe.IteratedExchangeRecipe(
		# 	smc.exchange_recipe.MposteriorExchangeRecipe(
		# 		self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
		# 		PRNG=self._PRNGs["topology pseudo random numbers generator"],
		# 		allow_exchange_one_particle_more_than_once=self._mposterior_exchange_particles_more_than_once),
		# 	self._MposteriorSettings["number of iterations"])
		#
		# mposterior_within_radius_exchange_recipe = smc.exchange_recipe.IteratedExchangeRecipe(
		# 	smc.exchange_recipe.MposteriorWithinRadiusExchangeRecipe(
		# 		self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
		# 		self._mposterior_exchange_step_depth, PRNG=self._PRNGs["topology pseudo random numbers generator"],
		# 		allow_exchange_one_particle_more_than_once=self._mposterior_exchange_particles_more_than_once),
		# 	self._MposteriorSettings["number of iterations"])

		mposterior_exchange_recipe = smc.exchange_recipe.IteratedExchangeRecipe(
			smc.exchange_recipe.SameParticlesMposteriorWithinRadiusExchangeRecipe(
				self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
					self._MposteriorSettings['findWeiszfeldMedian parameters'], 1,
					PRNG=self._PRNGs["topology pseudo random numbers generator"]),
			self._MposteriorSettings["number of iterations"])

		mposterior_within_radius_exchange_recipe = smc.exchange_recipe.IteratedExchangeRecipe(
			smc.exchange_recipe.SameParticlesMposteriorWithinRadiusExchangeRecipe(
					self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
					self._MposteriorSettings['findWeiszfeldMedian parameters'], self._mposterior_exchange_step_depth,
					PRNG=self._PRNGs["topology pseudo random numbers generator"]),
			self._MposteriorSettings["number of iterations"])

		# ------------

		for n_consensus_iter, color in zip(
				[self._LCDPFsettings['number of consensus iterations'], 10, 5], ['brown', 'yellowgreen', 'fuchsia']):

			likelihood_consensus_exchange_recipe = smc.exchange_recipe.LikelihoodConsensusExchangeRecipe(
				self._PEsTopology, n_consensus_iter,
				self._LCDPFsettings['degree of the polynomial approximation'])

			# consensus
			self._PFs.append(
				distributed.LikelihoodConsensusTargetTrackingParticleFilter(
					likelihood_consensus_exchange_recipe, self._nPEs, self._n_particles_per_PE, self._resampling_algorithm,
					self._resampling_criterion, self._prior, self._transition_kernel, self._sensors,
					self._PEsSensorsConnections, self._LCDPFsettings['degree of the polynomial approximation']
					)
			)

			# the estimator just delegates the calculus of the estimate to one of the PEs
			self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1], 0))

			self._estimators_colors.append(color)
			self._estimators_labels.append('Likelihood Consensus DPF with {} iterations'.format(n_consensus_iter))
		
		# ------------

		# a single PE (with the number of particles of any other PE) that has access to all the observations
		self._PFs.append(
			centralized.TargetTrackingParticleFilterWithFusionCenter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._i_PE_estimation
				)
		)

		# the estimator just delegates the calculus of the estimate to the PF
		self._estimators.append(smc.estimator.Delegating(self._PFs[-1]))

		self._estimators_colors.append('indigo')
		self._estimators_labels.append('Single know-it-all PE')
		
		# ------------

		# centralized PF
		self._PFs.append(
			centralized.TargetTrackingParticleFilterWithFusionCenter(
				self._n_particles_per_PE*self._nPEs, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._i_PE_estimation
				)
		)
			
		# the estimator just delegates the calculus of the estimate to the PF
		self._estimators.append(smc.estimator.Delegating(self._PFs[-1]))
		
		self._estimators_colors.append('lawngreen')
		self._estimators_labels.append('Centralized')
		
		# ------------
		
		# a distributed PF with DRNA
		self._PFs.append(
			distributed.TargetTrackingParticleFilterWithDRNA(
				self._settings_DRNA["exchange period"], drna_exchange_recipe, self._n_particles_per_PE,
				self._settings_DRNA["normalization period"], self._resampling_algorithm, self._resampling_criterion,
				self._prior, self._transition_kernel, self._sensors,
				self._everySensorWithEveryPEConnector.get_connections(self._nPEs)
			)
		)
		
		# the estimator is the mean
		self._estimators.append(smc.estimator.WeightedMean(self._PFs[-1]))
		
		self._estimators_colors.append('black')
		self._estimators_labels.append('DRNA exch. {}'.format(self._exchanged_particles))
		
		# ------------

		# DPF with M-posterior-based exchange
		self._PFs.append(
			distributed.TargetTrackingParticleFilterWithMposterior(
				mposterior_exchange_recipe, self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion,
				self._prior, self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				self._MposteriorSettings['sharing period'])
		)
		
		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(
			self._PFs[-1], max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
			tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

		self._estimators_colors.append('green')
		self._estimators_labels.append('M-posterior exch. {} ({} particle(s) from each PE)'.format(
			self._exchanged_particles, self._mposterior_n_part_estimation))
		
		# ------------

		for mposterior_estimator_radius in self._mposterior_estimator_radius:

			# an estimator which yields the geometric median of the particles in the "self._i_PE_estimation"-th PE
			self._estimators.append(smc.estimator.SinglePEGeometricMedianWithinRadius(
				self._PFs[-1], self._i_PE_estimation, self._PEsTopology, mposterior_estimator_radius,
				n_particles=self._mposterior_n_part_estimation, radius_lower_bound=self._mposterior_exchange_step_depth)
			)

			self._estimators_colors.append('coral')
			self._estimators_labels.append('M-posterior exch. {} ({} hops, {} particle(s))'.format(
				self._exchanged_particles, mposterior_estimator_radius, self._mposterior_n_part_estimation))
		
		# ------------

		for i_PE in range(self._nPEs):

			# an estimator which yields the mean of the particles in the "self._i_PE_estimation"-th PE
			self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1], i_PE))

			self._estimators_colors.append('olive')
			self._estimators_labels.append('M-posterior (mean with particles from PE \#{})'.format(i_PE))

		# ------------

		# DPF with M-posterior-based exchange within a certain depth
		self._PFs.append(
			distributed.TargetTrackingParticleFilterWithMposterior(
				mposterior_within_radius_exchange_recipe, self._n_particles_per_PE, self._resampling_algorithm,
				self._resampling_criterion, self._prior, self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				self._MposteriorSettings['sharing period'])
		)

		# an estimator computing the geometric median with 1 particle taken from each PE
		self._estimators.append(smc.estimator.GeometricMedian(
			self._PFs[-1], max_iterations=self._MposteriorSettings['findWeiszfeldMedian parameters']['maxit'],
			tolerance=self._MposteriorSettings['findWeiszfeldMedian parameters']['tol']))

		self._estimators_colors.append('blue')
		self._estimators_labels.append('M-posterior exch. {} - depth {} ({} particle(s) from each PE)'.format(
			self._exchanged_particles, self._mposterior_exchange_step_depth, self._mposterior_n_part_estimation))

		# ------------

		for mposterior_estimator_radius in self._mposterior_estimator_radius:

			# an estimator which yields the geometric median of the particles in the "self._i_PE_estimation"-th PE
			self._estimators.append(smc.estimator.SinglePEGeometricMedianWithinRadius(
				self._PFs[-1], self._i_PE_estimation, self._PEsTopology, mposterior_estimator_radius,
				n_particles=self._mposterior_n_part_estimation, radius_lower_bound=self._mposterior_exchange_step_depth)
			)

			self._estimators_colors.append('khaki')
			self._estimators_labels.append(
				'M-posterior exch. {} - depth {} ({} hops, {} particle(s))'.format(
					self._exchanged_particles, self._mposterior_exchange_step_depth, mposterior_estimator_radius,
					self._mposterior_n_part_estimation))

	def save_data(self, target_position):
		
		# let the super class do its thing...
		super().save_data(target_position)
		
		# a dictionary encompassing all the data to be saved
		data_to_be_saved = dict(
				targetPosition=target_position[:, :, :self._i_current_frame],
				PF_pos=self._estimated_pos[:, :, :self._i_current_frame, :]
			)
		
		# data is saved
		scipy.io.savemat('res_' + self._output_file, data_to_be_saved)
		print('results saved in "{}"'.format('res_' + self._output_file))
		
		# the mean of the error (euclidean distance) incurred by the PFs
		pf_error = np.sqrt(
			((self._estimated_pos[:, :, :self._i_current_frame, :] - target_position[:, :, :self._i_current_frame, np.newaxis]) ** 2).sum(
				axis=0)).mean(axis=1)
		
		plot.particle_filters(
			range(self._n_time_instants), pf_error,
			self._simulation_parameters["file name prefix for the estimation error vs time plot"] +
			'_' + self._output_file + '_nFrames={}.eps'.format(repr(self._i_current_frame)),
			[{'label': l, 'color': c} for l, c in zip(self._estimators_labels, self._estimators_colors)])
		
		print(self._estimated_pos)
		
	def process_frame(self, target_position, target_velocity):
		
		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		# this array will store the results before they are saved
		estimated_pos = np.full((state.n_elements_position, self._n_time_instants, len(self._estimators)), np.nan)

		# for every PF (different from estimator)...
		for pf in self._PFs:

			# ...initialization
			pf.initialize()

		if self._settings_painter['display evolution?'] and self._settings_painter["use display server if available?"]:

			# if this is not the first frame...
			if self._painter:

				# ...then, the previous figure is closed
				self._painter.close()

			# this object will handle graphics...
			self._painter = plot.RectangularRoomPainter(
				self._settings_room["bottom left corner"], self._settings_room["top right corner"], self._sensorsPositions,
				sleepTime=self._settings_painter["sleep time between updates"])

			# ...e.g., draw the self._sensors
			self._painter.setup()

		for iTime in range(self._n_time_instants):

			print(colorama.Fore.LIGHTWHITE_EX + '---------- iFrame = {}, iTime = {}'.format(self._i_current_frame, iTime) + colorama.Style.RESET_ALL)

			print(colorama.Fore.CYAN + 'position:\n' + colorama.Style.RESET_ALL, target_position[:, iTime:iTime+1])
			print(colorama.Fore.YELLOW + 'velocity:\n' + colorama.Style.RESET_ALL, target_velocity[:, iTime:iTime+1])

			# for every PF (different from estimator)...
			for pf in self._PFs:

				# ...a step is taken
				pf.step(self._observations[iTime])

			# for every estimator, along with its corresponding label,...
			for iEstimator, (estimator, label) in enumerate(zip(self._estimators, self._estimators_labels)):

				# for the sake of efficiency
				current_estimated_pos = state.to_position(estimator.estimate())

				self._estimated_pos[:, iTime:iTime + 1, self._i_current_frame, iEstimator] = current_estimated_pos

				# the position given by this estimator at the current time instant is written to the HDF5 file
				estimated_pos[:, iTime:iTime+1, iEstimator] = current_estimated_pos

				print('position estimated by {}\n'.format(label), self._estimated_pos[:, iTime:iTime + 1, self._i_current_frame, iEstimator])

			if self._settings_painter["display evolution?"] and self._settings_painter["use display server if available?"]:

				# the plot is updated with the position of the target...
				self._painter.updateTargetPosition(target_position[:, iTime:iTime+1])

				# ...those estimated by the PFs
				for iEstimator, (pf, color) in enumerate(zip(self._estimators, self._estimators_colors)):

					self._painter.updateEstimatedPosition(
							self._estimated_pos[:, iTime:iTime + 1, self._i_current_frame, iEstimator],
						identifier='#{}'.format(iEstimator), color=color)

					if self._settings_painter["display particles evolution?"]:

						self._painter.updateParticlesPositions(
							state.to_position(pf.get_state()), identifier='#{}'.format(iEstimator), color=color)

		# the results (estimated positions) are saved
		self._h5_current_frame.create_dataset(
			'estimated position', shape=estimated_pos.shape, dtype=float, data=estimated_pos)

		# in order to make sure the HDF5 files is valid...
		self._f.flush()

	def drop_duplicated_estimators(self):

		# the list with the names of the algorithms is turned into a numpy array, so that the indexes of the unique
		# elements can be easily obtained
		_, i_unique = np.unique(np.array(self._estimators_labels), return_index=True)

		# they are sorted (just for keeping the original order whenever possible)
		i_to_keep = sorted(i_unique)

		# these will contain the relevant information for the surviving estimators
		new_estimator_labels = []
		new_estimator_colors = []
		new_estimators = []

		# for every index that gives a list with unique elements....
		for i in i_to_keep:

			# relevant stuff is added to the lists...
			new_estimator_labels.append(self._estimators_labels[i])
			new_estimator_colors.append(self._estimators_colors[i])
			new_estimators.append(self._estimators[i])

		# ...that are later to replace the original lists
		self._estimators_labels = new_estimator_labels
		self._estimators_colors = new_estimator_colors
		self._estimators = new_estimators


class MposteriorExchange(Mposterior):

	def add_algorithms(self):

		for exchange in self._exchanged_particles:

			# the value set is used by "add_algorithms"
			self._exchanged_particles = exchange

			super().add_algorithms()

		self.drop_duplicated_estimators()


class MposteriorNumberOfParticles(Mposterior):

	def add_algorithms(self):

		for n_particles in self._simulation_parameters['number of particles']:

			# how many algorithms were queued so far
			n_algorithms = len(self._estimators)

			self._n_particles_per_PE = n_particles

			super().add_algorithms()

			# the number of particles used is prepended to the names of the algorithms
			self._estimators_labels[n_algorithms:] = [
				'[{} particles per PE] '.format(n_particles) + l for l in self._estimators_labels[n_algorithms:]]


class MposteriorEstimationRadius(Mposterior):

	def add_algorithms(self):

		self._mposterior_estimator_radius = self._simulation_parameters['radius']

		super().add_algorithms()


class MposteriorNumberOfParticlesForEstimation(Mposterior):

	def add_algorithms(self):

		for n_particles in self._simulation_parameters['number of particles']:

			self._mposterior_n_part_estimation = n_particles

			super().add_algorithms()

		self.drop_duplicated_estimators()


class MposteriorRevisited(Mposterior):

	def build_observations(self, target_position):

		if "malfunctioning PEs" not in self._simulation_parameters:

			super().build_observations(target_position)

			return

		# for convenience (notation)
		sensor_to_proc_elem = sensors_PEs_connector.sensors_PEs_mapping(self._PEsSensorsConnections)

		# ...idem
		failing_proc_elem = self._simulation_parameters["malfunctioning PEs"]

		self._observations = []

		for t, s in enumerate(target_position.T):

			# a list with the observations for all the sensors at the current time instant
			current_obs = []

			for i_sens, sens in enumerate(self._sensors):

				# the PE associated with this sensor
				i_pe = str(sensor_to_proc_elem[i_sens])

				# if the PE is in the list *and* marked to fail at this specific time instant...
				if (i_pe in failing_proc_elem) and (t in failing_proc_elem[i_pe]):

					if self._simulation_parameters["malfunctioning PEs deed"] == "pure noise":

						# ...the method yielding just noise is called
						current_obs.append(sens.measurement_noise())

					elif re.match('additive noise with variance (\d+)', self._simulation_parameters["malfunctioning PEs deed"]):

						m = re.match('additive noise with variance (\d+)', self._simulation_parameters["malfunctioning PEs deed"])

						# ...a large noise is added
						current_obs.append(
							sens.detect(state.to_position(s.reshape(-1, 1))) +
							self._PRNGs["Sensors and Monte Carlo pseudo random numbers generator"].randn()*np.sqrt(
								float(m.group(1))
							)
						)

					else:

						raise Exception("unknown deed for malfunctioning PEs")

				else:

					current_obs.append(sens.detect(state.to_position(s.reshape(-1, 1))))

			# all the observations for the current time instant are added to the final result
			self._observations.append(np.array(current_obs))

	def add_algorithms(self):
		"""Adds the algorithms to be tested by this simulation, defining the required parameters."""

		drna_exchange_recipe = smc.exchange_recipe.DRNAExchangeRecipe(
			self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		mposterior_within_radius_exchange_recipe = smc.exchange_recipe.IteratedExchangeRecipe(
			smc.exchange_recipe.SameParticlesMposteriorWithinRadiusExchangeRecipe(
					self._PEsTopology, self._n_particles_per_PE, self._exchanged_particles,
					self._MposteriorSettings['findWeiszfeldMedian parameters'], self._mposterior_exchange_step_depth,
					PRNG=self._PRNGs["topology pseudo random numbers generator"]),
			self._MposteriorSettings["number of iterations"])

		gaussian_exchange_recipe = smc.exchange_recipe.GaussianExchangeRecipe(
			self._PEsTopology, self._n_particles_per_PE, self._parameters["Gaussian products"], self._room,
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		perfect_gaussian_exchange_recipe = smc.exchange_recipe.PerfectConsensusGaussianExchangeRecipe(
			self._PEsTopology, self._n_particles_per_PE, self._parameters["Gaussian products"], self._room,
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		setmembership_constrained_exchange_recipe = smc.exchange_recipe.SetMembershipConstrainedExchangeRecipe(
			self._PEsTopology, self._parameters["Set-Membership constrained"], self._n_particles_per_PE,
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		alt_setmembership_constrained_exchange_recipe = smc.exchange_recipe.SetMembershipConstrainedExchangeRecipe(
			self._PEsTopology, self._parameters["Set-Membership constrained Alt."], self._n_particles_per_PE,
			PRNG=self._PRNGs["topology pseudo random numbers generator"])

		selective_gossip_exchange_recipe = smc.exchange_recipe.SelectiveGossipExchangeRecipe(
			self._PEsTopology, self._parameters["Selective Gossip"], self._PRNGs["topology pseudo random numbers generator"])

		perfect_selective_gossip_exchange_recipe = smc.exchange_recipe.PerfectSelectiveGossipExchangeRecipe(
			self._PEsTopology, self._parameters["Selective Gossip"],
			self._PRNGs["topology pseudo random numbers generator"])

		# ------------

		# a distributed PF with DRNA
		self._PFs.append(
			distributed.TargetTrackingParticleFilterWithDRNA(
				self._settings_DRNA["exchange period"], drna_exchange_recipe, self._n_particles_per_PE,
				self._settings_DRNA["normalization period"], self._resampling_algorithm, self._resampling_criterion,
				self._prior, self._transition_kernel, self._sensors,
				self._everySensorWithEveryPEConnector.get_connections(self._nPEs)
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.WeightedMean(self._PFs[-1]))

		self._estimators_colors.append('black')
		self._estimators_labels.append('DRNA exch. {}'.format(self._exchanged_particles))

		# ------------

		# DPF with M-posterior-based exchange within a certain depth
		self._PFs.append(
			distributed.TargetTrackingParticleFilterWithMposterior(
				mposterior_within_radius_exchange_recipe, self._n_particles_per_PE, self._resampling_algorithm,
				self._resampling_criterion, self._prior, self._transition_kernel, self._sensors,
				self._PEsSensorsConnections,
				self._MposteriorSettings['sharing period'])
		)

		self._estimators.append(smc.estimator.SinglePEMeansGeometricMedianWithinRadius(
			self._PFs[-1], self._i_PE_estimation, self._PEsTopology, self._mposterior_estimator_radius[0],
			radius_lower_bound=self._mposterior_exchange_step_depth)
		)

		self._estimators_colors.append('khaki')
		self._estimators_labels.append(
			'M-posterior exch. {} - depth {} ({} hops, {} particle(s))'.format(
				self._exchanged_particles, self._mposterior_exchange_step_depth, self._mposterior_estimator_radius[0],
				self._mposterior_n_part_estimation))

		# ------------

		for n_consensus_iter, color in zip(
				[self._LCDPFsettings['number of consensus iterations'], 75], ['brown', 'yellowgreen', 'fuchsia']):

			likelihood_consensus_exchange_recipe = smc.exchange_recipe.LikelihoodConsensusExchangeRecipe(
				self._PEsTopology, n_consensus_iter,
				self._LCDPFsettings['degree of the polynomial approximation'])

			# consensus
			self._PFs.append(
				distributed.LikelihoodConsensusTargetTrackingParticleFilter(
					likelihood_consensus_exchange_recipe, self._nPEs, self._n_particles_per_PE, self._resampling_algorithm,
					self._resampling_criterion, self._prior, self._transition_kernel, self._sensors,
					self._PEsSensorsConnections, self._LCDPFsettings['degree of the polynomial approximation']
					)
			)

			# the estimator just delegates the calculus of the estimate to one of the PEs
			self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1], 0))

			self._estimators_colors.append(color)
			self._estimators_labels.append('Likelihood Consensus DPF with {} iterations'.format(n_consensus_iter))

		# ------------

		# asynchronous DPF via decentralized
		self._PFs.append(
			distributed.TargetTrackingGaussianParticleFilter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				gaussian_exchange_recipe, self._parameters["Gaussian products"], self._room,
				PRNG=self._PRNGs["Sensors and Monte Carlo pseudo random numbers generator"],
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1]))

		self._estimators_colors.append('magenta')
		self._estimators_labels.append('Gaussian')

		# ------------

		# asynchronous DPF via decentralized...with perfect consensus
		self._PFs.append(
			distributed.TargetTrackingGaussianParticleFilter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				perfect_gaussian_exchange_recipe, self._parameters["Gaussian products"], self._room,
				PRNG=self._PRNGs["Sensors and Monte Carlo pseudo random numbers generator"],
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1]))

		self._estimators_colors.append('red')
		self._estimators_labels.append('Perfect Gaussian')

		# ------------

		# Set-membership Constrained DPF
		self._PFs.append(
			distributed.TargetTrackingSetMembershipConstrainedParticleFilter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				setmembership_constrained_exchange_recipe, self._parameters["Set-Membership constrained"],
				self._PRNGs["Sensors and Monte Carlo pseudo random numbers generator"]
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1]))

		self._estimators_colors.append('brown')
		self._estimators_labels.append('Set-Membership constrained')

		# ------------

		# Alternative (exchange with different parameters) Set-membership Constrained DPF
		self._PFs.append(
			distributed.TargetTrackingSetMembershipConstrainedParticleFilter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				alt_setmembership_constrained_exchange_recipe, self._parameters["Set-Membership constrained Alt."],
				self._PRNGs["Sensors and Monte Carlo pseudo random numbers generator"]
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1]))

		self._estimators_colors.append('yellowgreen')
		self._estimators_labels.append('Alt. Set-Membership constrained')

		# ------------

		# Selective gossip
		self._PFs.append(
			distributed.TargetTrackingSelectiveGossipParticleFilter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				selective_gossip_exchange_recipe, self._parameters["Selective Gossip"]
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1]))

		self._estimators_colors.append('fuchsia')
		self._estimators_labels.append('Selective gossip')

		# ------------

		# Selective gossip with perfect consensus
		self._PFs.append(
			distributed.TargetTrackingSelectiveGossipParticleFilter(
				self._n_particles_per_PE, self._resampling_algorithm, self._resampling_criterion, self._prior,
				self._transition_kernel, self._sensors, self._PEsSensorsConnections,
				perfect_selective_gossip_exchange_recipe, self._parameters["Selective Gossip"]
			)
		)

		# the estimator is the mean
		self._estimators.append(smc.estimator.SinglePEMean(self._PFs[-1]))

		self._estimators_colors.append('blue')
		self._estimators_labels.append('Perfect Selective gossip')