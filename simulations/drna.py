import numpy as np
import scipy.io
import h5py

import smc.particle_filter.centralized as centralized
import smc.particle_filter.distributed as distributed
import smc.exchange_recipe
import smc.estimator
import PEs_topology
import drnautil
import sensors_PEs_connector
import state
import simulations.base


class Convergence(simulations.base.SimpleSimulation):

	@staticmethod
	def parse_hdf5(data_file):

		n_state, n_time_instants, n_algorithms = data_file['frames/0/topology/0/estimated position'].shape
		n_topologies = len(data_file['frames/0/topology'])
		n_frames = len(data_file['frames'])

		estimated_position = np.empty((n_state, n_time_instants, n_algorithms, n_frames, n_topologies))

		for i_frame in range(len(data_file['frames'])):

			for i_topology, topology in enumerate(data_file['frames/{}/topology'.format(i_frame)]):
				estimated_position[..., i_frame, i_topology] = data_file[
					'frames/{}/topology/{}/estimated position'.format(i_frame, i_topology)]

		actual_position = np.concatenate(
			[
				data_file['frames/{}/actual position'.format(i)][...][..., np.newaxis]
				for i in range(len(data_file['frames']))], axis=2)

		return actual_position, estimated_position

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file=None, h5py_prefix=''):

		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix)

		# DRNA-related settings
		self._settings_DRNA = parameters["DRNA"]

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

		PEs_sensors_requirements = sensors_PEs_connector.EverySensorWithEveryPEConnector(self._sensors_positions)

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

		# the aggregated weights are normalized at ALL TIMES, for EVERY frame and EVERY topology
		normalized_aggregated_weights = [
			w[:, :, :self._i_current_frame] / w[:, :, :self._i_current_frame].sum(axis=1)[:, np.newaxis, :]
			for w in self._distributedPFaggregatedWeights]

		# ...the same data structured in a dictionary
		dic_normalized_aggregated_weights = {
			'normalizedAggregatedWeights_{}'.format(i): array for i, array in enumerate(normalized_aggregated_weights)}

		# a dictionary encompassing all the data to be saved
		data_to_be_saved = dict(
				aggregatedWeightsUpperBounds=self._aggregatedWeightsUpperBounds,
				targetPosition=target_position[:, :, :self._i_current_frame],
				centralizedPF_pos=self._centralizedPF_pos[:, :, :self._i_current_frame, :],
				distributedPF_pos=self._distributedPF_pos[:, :, :self._i_current_frame, :],
				**dic_normalized_aggregated_weights
			)

		# data is saved
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

			# data is saved
			h5_estimated_pos = self._h5_current_frame.create_dataset(
				'topology/{}/estimated position'.format(iTopology), shape=estimated_pos.shape, dtype=float,
				data=estimated_pos)

			h5_estimated_pos.attrs['M'] = n_PEs

			self._h5_current_frame.create_dataset(
				'topology/{}/DPF aggregated weights'.format(iTopology), aggregated_weights.shape, dtype=float,
				data=aggregated_weights)
