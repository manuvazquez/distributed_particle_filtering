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
import sensors_PEs_connector
import state
import simulations.base


class MultipleMposterior(simulations.base.Simulation):
	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators):

		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators)

		# HDF5 output file
		self._f = h5py.File(self._output_file_basename + '.hdf5', 'w')

		# we will build several "Mposterior" objects...
		self._simulations = []

		# ...and each one will have a different set of sensors
		self._sensors = []

		# for every pair nPEs-nSensors we aim to simulate...
		for (nPEs, nSensors) in self._simulation_parameters["nPEs-nSensors pairs"]:
			self._simulations.append(Mposterior(
				parameters, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
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


class Mposterior(simulations.base.SimpleSimulation):
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
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='', n_processing_elements=None,
			n_sensors=None):

		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements, n_sensors)

		# DRNA-related settings
		self._settings_DRNA = parameters["DRNA"]

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
		self._everySensorWithEveryPEConnector = sensors_PEs_connector.EverySensorWithEveryPEConnector(
			self._sensors_positions)

		# the settings for the selected "sensors-PES connector"
		settings_sensors_PEs_connector = parameters['sensors-PEs connectors'][
			self._simulation_parameters['sensors-PEs connector']]

		# ...are used to build a connector, from which the links between PEs and sensors are obtained
		self._PEsSensorsConnections = getattr(sensors_PEs_connector,
		                                      settings_sensors_PEs_connector['implementing class'])(
			self._sensors_positions, self._PEs_positions, settings_sensors_PEs_connector['parameters']).get_connections(
			self._nPEs)

		# network topology, which describes the connection among PEs, as well as the exact particles exchanged/shared
		self._PEsTopology = getattr(PEs_topology, self._settings_topologies['implementing class'])(
			self._nPEs, self._settings_topologies['parameters'], PEs_positions=self._PEs_positions)

		# the maximum number of hops between any pair of PEs
		max_hops_number = self._PEsTopology.distances_between_processing_elements.max()

		print('maximum number of hops between PEs = {}'.format(max_hops_number))

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
		self._estimated_pos = np.empty(
			(2, self._n_time_instants, parameters["number of frames"], len(self._estimators)))

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
				self._PEs_positions.T, self._PEsSensorsConnections, self._PEsTopology.get_neighbours())):
			self._f.create_dataset(self._h5py_prefix + 'PEs/{}/position'.format(iPE), shape=(2,), data=pos)
			self._f.create_dataset(
				self._h5py_prefix + 'PEs/{}/connected sensors'.format(iPE), shape=(len(sens),), data=sens)
			self._f.create_dataset(
				self._h5py_prefix + 'PEs/{}/neighbours'.format(iPE), shape=(len(neighbours),), data=neighbours)

		self._f[self._h5py_prefix + 'PEs'].attrs['max number of hops'] = max_hops_number

		# the positions of the sensors
		self._f.create_dataset(
			self._h5py_prefix + 'sensors/positions', shape=self._sensors_positions.shape, data=self._sensors_positions)

		# a list with the messages required by every algorithm at a single time instant
		algorithms_messages = []

		for estimator, label in zip(self._estimators, self._estimators_labels):
			# number of messages due to the particular estimator used
			messages_during_estimation = estimator.messages(self._PEsTopology)

			# number of messages related to the algorithm
			messages_algorithm_operation = estimator.DPF.messages(self._PEsTopology, self._PEsSensorsConnections)

			algorithms_messages.append(messages_during_estimation + messages_algorithm_operation)

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
					likelihood_consensus_exchange_recipe, self._nPEs, self._n_particles_per_PE,
					self._resampling_algorithm,
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
				self._n_particles_per_PE * self._nPEs, self._resampling_algorithm, self._resampling_criterion,
				self._prior,
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
				mposterior_exchange_recipe, self._n_particles_per_PE, self._resampling_algorithm,
				self._resampling_criterion,
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
				self._resampling_criterion, self._prior, self._transition_kernel, self._sensors,
				self._PEsSensorsConnections,
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
		scipy.io.savemat('res_' + self._output_file_basename, data_to_be_saved)
		print('results saved in "{}"'.format(self._output_file_basename))

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

		for iTime in range(self._n_time_instants):

			print(colorama.Fore.LIGHTWHITE_EX + '---------- iFrame = {}, iTime = {}'.format(self._i_current_frame,
			                                                                                iTime) + colorama.Style.RESET_ALL)

			print(colorama.Fore.CYAN + 'position:\n' + colorama.Style.RESET_ALL, target_position[:, iTime:iTime + 1])
			print(colorama.Fore.YELLOW + 'velocity:\n' + colorama.Style.RESET_ALL, target_velocity[:, iTime:iTime + 1])

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
				estimated_pos[:, iTime:iTime + 1, iEstimator] = current_estimated_pos

				print('position estimated by {}\n'.format(label),
				      self._estimated_pos[:, iTime:iTime + 1, self._i_current_frame, iEstimator])

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

					elif re.match('additive noise with variance (\d+)',
					              self._simulation_parameters["malfunctioning PEs deed"]):

						m = re.match('additive noise with variance (\d+)',
						             self._simulation_parameters["malfunctioning PEs deed"])

						# ...a large noise is added
						current_obs.append(
							sens.detect(state.to_position(s.reshape(-1, 1))) +
							self._PRNGs["Sensors and Monte Carlo pseudo random numbers generator"].randn() * np.sqrt(
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
			self._PEsTopology, self._parameters["Selective Gossip"],
			self._PRNGs["topology pseudo random numbers generator"])

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
					likelihood_consensus_exchange_recipe, self._nPEs, self._n_particles_per_PE,
					self._resampling_algorithm,
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