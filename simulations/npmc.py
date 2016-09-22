import sys
import os

import numpy as np
import colorama
import h5py

import smc.particle_filter.centralized as centralized
import smc.exchange_recipe
import smc.estimator
import state
import simulations.base

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util


class NPMC(simulations.base.SimpleSimulation):

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='', n_processing_elements=None,
			n_sensors=None):

		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements, n_sensors)

		self._n_particles_likelihood = self._simulation_parameters["number of particles for approximating the likelihood"]
		self._n_particles = self._simulation_parameters["number of particles"]

		# the pseudo random numbers generator this class will be using
		self._prng = self._PRNGs['Sensors and Monte Carlo pseudo random numbers generator']

		# algorithm is added
		self._pf_state = centralized.TargetTrackingParticleFilter(
			self._n_particles_likelihood, self._resampling_algorithm, self._resampling_criterion,
			self._prior, self._transition_kernel, self._sensors
		)

		self._estimator = smc.estimator.Delegating(self._pf_state)

		# the position estimates
		self._estimated_pos = np.empty(
			(state.n_elements_position, self._n_time_instants, parameters["number of frames"]))

		# -------------

		# # self._min_power = self._simulation_parameters["prior"]["minimum amount of power"]
		# self._min_power = dict()
		# self._min_power['loc'] = self._simulation_parameters["prior"]["minimum amount of power"]["mean"]
		# self._min_power['scale'] = np.sqrt(self._simulation_parameters["prior"]["minimum amount of power"]["variance"])
		#
		# # self._path_loss_exponent = self._simulation_parameters["prior"]["path loss exponent"]
		# self._path_loss_exponent = dict()
		# self._path_loss_exponent['loc'] = self._simulation_parameters["prior"]["path loss exponent"]["mean"]
		# self._path_loss_exponent['scale'] = np.sqrt(self._simulation_parameters["prior"]["path loss exponent"]["variance"])

		self._prior_mean = np.array([
			self._simulation_parameters["prior"]["minimum amount of power"]["mean"],
			self._simulation_parameters["prior"]["path loss exponent"]["mean"]])

		self._prior_covar = np.diag([
			self._simulation_parameters["prior"]["minimum amount of power"]["variance"],
			self._simulation_parameters["prior"]["path loss exponent"]["variance"]])

		# ------------- HDF5

		# the positions of the sensors
		self._f.create_dataset(
			self._h5py_prefix + 'sensors/positions', shape=self._sensors_positions.shape, data=self._sensors_positions)

	def save_data(self, target_position):

		# let the super class do its thing...
		super().save_data(target_position)

		print(self._estimated_pos)

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		# this array will store the results before they are saved
		estimated_pos = np.full((state.n_elements_position, self._n_time_instants, 1), np.nan)

		# the initial mean and covariance are given by the prior
		mean = self._prior_mean
		covar = self._prior_covar

		# 1st component (column) is "minimum amount of power" and 2nd is "path loss exponent"
		samples = self._prng.multivariate_normal(mean, covar, size=self._n_particles)

		# min_power_seq = np.random.normal(**self._min_power, size=self._n_particles)
		# path_loss_exponent_seq = np.random.normal(**self._path_loss_exponent, size=self._n_particles)

		# likelihoods = np.ones(self._n_particles)
		loglikelihoods = np.zeros(self._n_particles)

		for i_particle, (min_power, path_loss_exp) in enumerate(samples):
		# for i_particle, (min_power, path_loss_exp) in enumerate(zip(min_power_seq, path_loss_exponent_seq)):

			# the "inner" PF (for approximating the likelihood) is initialized
			self._pf_state.initialize()

			# import code
			# code.interact(local=dict(globals(), **locals()))

			# the parameters of the sensors *within* the PF are set accordingly
			for s in self._pf_state.sensors:

				s.set_parameters(minimum_amount_of_power=min_power, path_loss_exponent=path_loss_exp)

			# import code
			# code.interact(local=dict(globals(), **locals()))

			# the collection of observations is processed
			for i_time, obs in enumerate(self._observations):

				print(
					colorama.Fore.LIGHTWHITE_EX +
					'---------- i frame = {}, i time = {}'.format(self._i_current_frame, i_time) +
					colorama.Style.RESET_ALL)

				print(colorama.Fore.CYAN + 'position:\n' + colorama.Style.RESET_ALL, target_position[:, i_time:i_time + 1])
				print(colorama.Fore.YELLOW + 'velocity:\n' + colorama.Style.RESET_ALL, target_velocity[:, i_time:i_time + 1])

				# ...a step is taken
				self._pf_state.step(obs)

				# likelihoods[i_particle] *= self._pf_state.last_unnormalized_likelihoods.mean()

				# the loglikelihoods computed by the "inner" bootstrap filter
				loglikes = self._pf_state.last_unnormalized_loglikelihoods

				# logarithm of the average
				loglikelihoods[i_particle] += manu.smc.util.log_sum_from_individual_logs(loglikes) - np.log(len(loglikes))

				# if np.isnan(likelihoods[i_particle]):
				#
				# 	import code
				# 	code.interact(local=dict(globals(), **locals()))

				# import code
				# code.interact(local=dict(globals(), **locals()))

				current_estimated_pos = state.to_position(self._estimator.estimate())

				# the position given by this estimator at the current time instant is written to the HDF5 file
				estimated_pos[:, i_time:i_time + 1, 0] = current_estimated_pos

				self._estimated_pos[:, i_time:i_time + 1, self._i_current_frame] = current_estimated_pos

				print('position estimated\n', current_estimated_pos)

			# import code
			# code.interact(local=dict(globals(), **locals()))

		# print(likelihoods)
		print(loglikelihoods)

		weights = manu.smc.util.normalize_from_logs(loglikelihoods)

		mean = weights @ samples
		covar = np.cov(samples.T, ddof=0, aweights=weights)

		# mean_min_power = min_power_seq @ weights
		# mean_path_loss_exp = path_loss_exponent_seq @ weights

		import code
		code.interact(local=dict(globals(), **locals()))

		# in order to make sure the HDF5 files is valid...
		self._f.flush()
