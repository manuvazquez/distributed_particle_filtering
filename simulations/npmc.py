import sys
import os
import types

import numpy as np
import colorama
import h5py

import smc.particle_filter.particle_filter
import smc.particle_filter.centralized as centralized
import smc.estimator
import simulations.base

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util


def normal_parameters_from_lognormal(mean, var):

	var = 1 + var / mean ** 2
	mean = np.log(mean / np.sqrt(var))

	return mean, var


class PopulationMonteCarlo(smc.particle_filter.particle_filter.ParticleFilter):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, prng, name=None):

		super().__init__(n_particles, resampling_algorithm, resampling_criterion, name=name)

		# self._pf = copy.deepcopy(pf)
		self._pf = pf
		self._prior_mean = prior_mean
		self._prior_covar = prior_covar
		self._prng = prng

		# these are "global" attributes
		self._samples = None
		self._loglikelihoods = None
		self._mean = None
		self._covar = None
		self._weights = None

	@property
	def weights(self):

		return self._weights

	def initialize(self):

		# the initial mean and covariance are given by the prior
		self._mean = self._prior_mean
		self._covar = self._prior_covar

	def step(self, observations):

		# 1st component (column) is "minimum amount of power" and 2nd is "path loss exponent"
		self._samples = self._prng.multivariate_normal(self._mean, self._covar, size=self._n_particles)

		self._loglikelihoods = np.zeros(self._n_particles)

		for i_particle, (tx_power, min_power, path_loss_exp) in enumerate(self._samples):

			# the "inner" PF (for approximating the likelihood) is initialized
			self._pf.initialize()

			# the parameters of the sensors *within* the PF are set accordingly
			for s in self._pf.sensors:

				s.set_parameters(
					tx_power=np.exp(tx_power), minimum_amount_of_power=np.exp(min_power), path_loss_exponent=path_loss_exp)

			# the collection of observations is processed
			for i_time, obs in enumerate(observations):

				# ...a step is taken
				self._pf.step(obs)

				# the loglikelihoods computed by the "inner" bootstrap filter
				loglikes = self._pf.last_unnormalized_loglikelihoods
				
				# logarithm of the average
				self._loglikelihoods[i_particle] += manu.smc.util.log_sum_from_individual_logs(loglikes) - np.log(len(loglikes))

		self._weights = manu.smc.util.normalize_from_logs(self._loglikelihoods)

		self.update_proposal()

		adjusted_mean = self._mean.copy()
		adjusted_mean[:2] = np.exp(adjusted_mean[:2])

		print('mean:\n', self._mean)
		print('covar:\n', self._covar)
		print('adjusted mean:\n', colorama.Fore.LIGHTWHITE_EX + '{}'.format(adjusted_mean) + colorama.Style.RESET_ALL)

	def update_proposal(self):

		self._mean = self._weights @ self._samples
		self._covar = np.cov(self._samples.T, ddof=0, aweights=self._weights)


class NonLinearPopulationMonteCarlo(PopulationMonteCarlo):

	def __init__(
			self, n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, M_T, prng, name=None):

		super().__init__(
			n_particles, resampling_algorithm, resampling_criterion, pf, prior_mean, prior_covar, prng, name=name)

		self._M_T = M_T

		self._unclipped_weights = None

	@property
	def weights(self):

		return self._unclipped_weights

	def update_proposal(self):

		# this is saved because it's returned by the above property
		self._unclipped_weights = self._weights.copy()

		# indices of the samples whose weight is to be clipped
		i_clipped = np.argpartition(self._loglikelihoods, -self._M_T)[-self._M_T:]

		# minimum (unnormalized) weight among those to be clipped
		clipping_threshold = self._loglikelihoods[i_clipped[0]]

		self._loglikelihoods[i_clipped] = clipping_threshold

		self._weights = manu.smc.util.normalize_from_logs(self._loglikelihoods)

		self._mean = self._weights @ self._samples
		self._covar = np.cov(self._samples.T, ddof=0, aweights=self._weights)


class NonLinearPopulationMonteCarloCovarOnly(NonLinearPopulationMonteCarlo):

	def update_proposal(self):

		super().update_proposal()

		# mean is recomputed using the unclipped weights
		self._mean = self._unclipped_weights @ self._samples


# ======================================================


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
		n_particles = self._simulation_parameters["number of particles"]
		self._n_iter_pmc = self._simulation_parameters["number of Monte Carlo iterations"]

		self._n_trials = self._simulation_parameters["number of trials"]

		n_clipped_particles_from_overall = eval(
			self._simulation_parameters["number of clipped particles from overall number"])

		# the above parameter should be a function
		assert isinstance(n_clipped_particles_from_overall, types.FunctionType)

		# the number of particles to be clipped is obtained using this function
		M_Ts = [n_clipped_particles_from_overall(M) for M in n_particles]

		# the pseudo random numbers generator this class will be using
		prng = self._PRNGs['Sensors and Monte Carlo pseudo random numbers generator']

		# the "inner" PF is built
		inner_pf = centralized.TargetTrackingParticleFilter(
			self._n_particles_likelihood, self._resampling_algorithm, self._resampling_criterion,
			self._prior, self._transition_kernel, self._sensors
		)

		# the *minimum amount of power* is assumed to be log-normal" (it must be positive)...
		mean_log_min_power = self._simulation_parameters["prior"]["minimum amount of power"]["mean"]
		var_log_min_power = self._simulation_parameters["prior"]["minimum amount of power"]["variance"]

		# ...and the parameters of the corresponding normal are
		mean_min_power, var_min_power = normal_parameters_from_lognormal(mean_log_min_power, var_log_min_power)

		# the same for the *transmitter power*
		mean_log_tx_power = self._simulation_parameters["prior"]["transmitter power"]["mean"]
		var_log_tx_power = self._simulation_parameters["prior"]["transmitter power"]["variance"]

		mean_tx_power, var_tx_power = normal_parameters_from_lognormal(mean_log_tx_power, var_log_tx_power)

		prior_mean = np.array([
			mean_tx_power,
			mean_min_power,
			self._simulation_parameters["prior"]["path loss exponent"]["mean"]])

		prior_covar = np.diag([
			var_tx_power,
			var_min_power,
			self._simulation_parameters["prior"]["path loss exponent"]["variance"]])

		pmc = [PopulationMonteCarlo(
			M, resampling_algorithm, resampling_criterion, inner_pf, prior_mean, prior_covar, prng, name='PMC')
		for M in n_particles]

		nonlinear_pmc = [NonLinearPopulationMonteCarlo(
			M, resampling_algorithm, resampling_criterion, inner_pf, prior_mean, prior_covar, M_T, prng, name='NPMC')
		for M, M_T in zip(n_particles, M_Ts)]

		nonlinear_pmc_only_covar = [NonLinearPopulationMonteCarloCovarOnly(
			M, resampling_algorithm, resampling_criterion, inner_pf, prior_mean, prior_covar, M_T, prng, name='NPMC (covar)')
		for M, M_T in zip(n_particles, M_Ts)]

		self._algorithms = [pmc, nonlinear_pmc, nonlinear_pmc_only_covar]

		# [<component>,<iteration>,<algorithm>,<particles>,<trial>,<frame>]
		self._estimated_parameters = np.empty((
			len(prior_mean), self._n_iter_pmc, len(self._algorithms), len(n_particles), self._n_trials,
			parameters["number of frames"]))

		# [<iteration>,<algorithm>,<particles>,<trial>,<frame>]
		self._max_weight = np.empty((
			self._n_iter_pmc, len(self._algorithms), len(n_particles), self._n_trials,
			parameters["number of frames"]))

		# [<iteration>,<algorithm>,<particles>,<trial>,<frame>]
		self._M_eff = np.empty((
			self._n_iter_pmc, len(self._algorithms), len(n_particles), self._n_trials,
			parameters["number of frames"]))

		# ----- HDF5

		# the names of the algorithms are also stored
		param_names = self._f.create_dataset(
			self._h5py_prefix + 'parameters', shape=(3,), dtype=h5py.special_dtype(vlen=str))
		param_names[0] = 'transmitter_power'
		param_names[1] = 'minimum_amount_of_power'
		param_names[2] = 'path_loss_exponent'

		self._f.create_dataset(self._h5py_prefix + 'prior mean', data=prior_mean)
		self._f.create_dataset(self._h5py_prefix + 'prior covariance', data=prior_covar)

		# the positions of the sensors
		self._f.create_dataset(
			self._h5py_prefix + 'sensors/positions', shape=self._sensors_positions.shape, data=self._sensors_positions)

	def save_data(self, target_position):

		# let the *grandparent* class do its thing...
		simulations.base.Simulation.save_data(self, target_position)

		self._f.create_dataset(self._h5py_prefix + 'estimated parameters', data=self._estimated_parameters)
		self._f.create_dataset(self._h5py_prefix + 'maximum weight', data=self._max_weight)
		self._f.create_dataset(self._h5py_prefix + 'effective sample size', data=self._M_eff)

		# if a reference to an HDF5 was not received, that means the file was created by this object,
		# and hence it is responsible of closing it...
		if not self._h5py_file:
			# ...in order to make sure the HDF5 file is valid...
			self._f.close()

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		for i_trial in range(self._n_trials):

			for alg in self._algorithms:

				for alg_particles in alg:

					alg_particles.initialize()

			for i_iter in range(self._n_iter_pmc):

				print(
					colorama.Fore.LIGHTWHITE_EX + 'frame {}'.format(self._i_current_frame) + colorama.Style.RESET_ALL +
					' | ' +
					colorama.Fore.LIGHTGREEN_EX + 'trial {}'.format(i_trial) + colorama.Style.RESET_ALL +
					' | ' +
					colorama.Fore.LIGHTBLUE_EX + 'PMC iteration {}'.format(i_iter) + colorama.Style.RESET_ALL)

				for i_alg, alg in enumerate(self._algorithms):

					print(colorama.Fore.LIGHTMAGENTA_EX + alg[0].name + colorama.Style.RESET_ALL)

					for i_particles, alg_particles in enumerate(alg):

						print(
							colorama.Fore.LIGHTCYAN_EX + 'n particles {}'.format(alg_particles.n_particles) +
							colorama.Style.RESET_ALL)

						alg_particles.step(self._observations)

						self._estimated_parameters[:, i_iter, i_alg, i_particles, i_trial, self._i_current_frame] =\
							alg_particles._mean

						self._max_weight[i_iter, i_alg, i_particles, i_trial, self._i_current_frame] =\
							alg_particles.weights.max()

						self._M_eff[i_iter, i_alg, i_particles, i_trial, self._i_current_frame] =\
							1. / np.sum(alg_particles.weights ** 2)

					print('=========')

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# in order to make sure the HDF5 files is valid...
		self._f.flush()
