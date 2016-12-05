import abc
import numpy as np
import h5py

import sensor
import state
import network_nodes


class Simulation(metaclass=abc.ABCMeta):
	@abc.abstractmethod
	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
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
		self._output_file_basename = output_file_basename

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
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements=None, n_sensors=None):

		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators)

		# for saving the data in HDF5
		self._h5py_file = h5py_file
		self._h5py_prefix = h5py_prefix

		# if a reference to an HDF5 file was not received...
		if h5py_file is None:

			# ...a new HDF5 file is created
			self._f = h5py.File(self._output_file_basename + '.hdf5', 'w', driver='core', libver='latest')

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
		self._sensors_positions, self._PEs_positions = self._network.sensors_positions, self._network.PEs_positions

		# the class to be instantiated is figured out from the settings for that particular sensor type
		sensor_class = getattr(sensor, sensors_settings[parameters['sensors type']]['implementing class'])

		self._sensors = [sensor_class(
			pos[:, np.newaxis],
			pseudo_random_numbers_generator=pseudo_random_numbers_generators[
				'Sensors and Monte Carlo pseudo random numbers generator'],
			**sensors_settings[parameters['sensors type']]['parameters']
		) for pos in self._sensors_positions.T]

		self._f.create_dataset(
			self._h5py_prefix + 'room/bottom left corner',
			parameters['room']['bottom left corner'].shape, data=parameters['room']['bottom left corner'])

		self._f.create_dataset(
			self._h5py_prefix + 'room/top right corner',
			parameters['room']['top right corner'].shape, data=parameters['room']['top right corner'])

		# these are going to be set/used by other methods
		self._observations = None
		self._h5_current_frame = None

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

		self.save_pseudo_random_numbers_generators_in_hdf5_group(
			pseudo_random_numbers_generators, self._h5_current_frame)

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
				state = [None] * 5

				state[0] = prngs[p]['1'][0]
				state[1] = prngs[p]['2'][...]
				state[2] = prngs[p]['3'][0]
				state[3] = prngs[p]['4'][0]
				state[4] = prngs[p]['5'][0]

				res[p] = np.random.RandomState()
				res[p].set_state(tuple(state))

		return res
