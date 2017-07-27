import sys
import os
import types
import abc

import numpy as np
import colorama

import mc.util
import mc.pmc
import mc.mh
import mc.amis
import mc.distributions
import smc.particle_filter.centralized as centralized
import simulations.base

import manu.util


class AbstractNPMC(simulations.base.SimpleSimulation, metaclass=abc.ABCMeta):

	@property
	def monte_carlo_algorithm_name(self):

		return "Population Monte Carlo"

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='', n_processing_elements=None,
			n_sensors=None):

		# let the super class do its thing...
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements, n_sensors)

		self._n_trials = self._simulation_parameters["number of trials"]

		self._n_iter_pmc = parameters[self.monte_carlo_algorithm_name]["number of iterations"]
		n_clipped_particles_from_overall = eval(
			parameters[self.monte_carlo_algorithm_name]["Nonlinear"]["number of clipped particles from overall number"])

		# the above parameter should be a function
		assert isinstance(n_clipped_particles_from_overall, types.FunctionType)

		self._n_particles = self._simulation_parameters["number of particles"]

		# number of particles for the filter used to compute the weights (likelihoods)
		self._n_particles_likelihood = self._simulation_parameters["number of particles for approximating the likelihood"]

		# the number of particles to be clipped is obtained using this function
		self._M_Ts = [n_clipped_particles_from_overall(M) for M in self._n_particles]

		# the pseudo random numbers generator this class will be using
		self._prng = self._PRNGs['Sensors and Monte Carlo pseudo random numbers generator']

		# "True" if Monte Carlo samples are to be saved (might take up a lot of space) by the children classes
		self._save_monte_carlo_samples =\
			("save MC samples" in self._simulation_parameters) and self._simulation_parameters["save MC samples"]

		# ------------------------- prior

		# the *minimum amount of power* is assumed to be log-normal" (it must be positive)...
		mean_log_min_power = self._simulation_parameters["prior"]["minimum amount of power"]["mean"]
		var_log_min_power = self._simulation_parameters["prior"]["minimum amount of power"]["variance"]

		# ...and the parameters of the corresponding normal are
		mean_min_power, var_min_power = mc.util.normal_parameters_from_lognormal(mean_log_min_power, var_log_min_power)

		# ---

		# the same for the *transmitter power*
		mean_log_tx_power = self._simulation_parameters["prior"]["transmitter power"]["mean"]
		var_log_tx_power = self._simulation_parameters["prior"]["transmitter power"]["variance"]

		mean_tx_power, var_tx_power = mc.util.normal_parameters_from_lognormal(mean_log_tx_power, var_log_tx_power)

		# ---

		# the same for the *transmitter power*
		mean_log_path_loss_exponent = self._simulation_parameters["prior"]["path loss exponent"]["mean"]
		var_log_path_loss_exponent = self._simulation_parameters["prior"]["path loss exponent"]["variance"]

		mean_path_loss_exponent, var_path_loss_exponent = mc.util.normal_parameters_from_lognormal(
			mean_log_path_loss_exponent, var_log_path_loss_exponent)

		# ---

		self._prior_mean = np.array([mean_tx_power, mean_min_power, mean_path_loss_exponent])

		self._prior_covar = np.diag([var_tx_power, var_min_power, var_path_loss_exponent])

		# ----- HDF5

		# the names of the parameters are stored
		manu.util.write_strings_list_into_hdf5(
			self._f, self._h5py_prefix + 'parameters',
			['transmitter_power', 'minimum_amount_of_power', 'path_loss_exponent'])

		self._f.create_dataset(self._h5py_prefix + 'prior mean', data=self._prior_mean)
		self._f.create_dataset(self._h5py_prefix + 'prior covariance', data=self._prior_covar)

		# the positions of the sensors
		self._f.create_dataset(
			self._h5py_prefix + 'sensors/positions', shape=self._sensors_positions.shape, data=self._sensors_positions)

	def initialize_pmc_algorithms(self):

		for alg in self._algorithms:

			for alg_particles in alg:

				alg_particles.initialize()

	def run_pmc_algorithms(self, i_trial, i_iter):

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

				self._estimated_parameters[:, i_iter, i_alg, i_particles, i_trial, self._i_current_frame] = \
					alg_particles._mean

				self._max_weight[i_iter, i_alg, i_particles, i_trial, self._i_current_frame] = \
					alg_particles.weights.max()

				self._M_eff[i_iter, i_alg, i_particles, i_trial, self._i_current_frame] = \
					1. / np.sum(alg_particles.weights ** 2)

				if self._save_monte_carlo_samples:

					# [<sample>,<component>,<trial>,<frame>]
					self._monte_carlo_samples[i_alg][i_particles][i_iter][
						..., i_trial, self._i_current_frame] = alg_particles.samples

					# [<sample>,<algorithm>,<trial>,<frame>]
					self._monte_carlo_weights[i_alg][i_particles][i_iter][
						..., i_trial, self._i_current_frame] = alg_particles.weights

			print('=========')


class NPMC(AbstractNPMC):

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='',
			n_processing_elements=None, n_sensors=None):

		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements,
			n_sensors)

		burn_in_period_metropolis_hastings = parameters["Metropolis-Hastings"]["burn-in period"]
		kernel_prior_covar_ratio_metropolis_hastings = parameters["Metropolis-Hastings"]["ratio kernel-prior covariances"]

		# the "inner" PF is built
		self._inner_pf = centralized.TargetTrackingParticleFilter(
			self._n_particles_likelihood, self._resampling_algorithm, self._resampling_criterion, self._prior,
			self._transition_kernel, self._sensors
		)

		n_samples_metropolis_hastings = self._n_iter_pmc * self._n_particles[-1]

		prior = mc.distributions.Gaussian(self._prng, {'mean': self._prior_mean, 'cov': self._prior_covar})
		proposal = mc.distributions.Gaussian(self._prng, {
			'cov': self._prior_covar*kernel_prior_covar_ratio_metropolis_hastings})

		# Metropolis-Hastings algorithm is run considering the largest number of samples and iterations
		self.metropolis_hastings = mc.mh.MetropolisHastings(
			n_samples_metropolis_hastings, self._inner_pf, prior, self._prng, burn_in_period_metropolis_hastings,
			proposal, name='MetropolisHastings')

		if self._simulation_parameters["only run MCMC"]:

			self._algorithms = []

		else:

			self._algorithms = self.create_smc_algoritms()

		# ------------------------- accumulators

		# [<#particles>,<trial>,<frame>]
		common_parameters = (len(self._n_particles), self._n_trials, parameters["number of frames"])

		# [<component>,<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._estimated_parameters = np.empty((
			len(self._prior_mean), self._n_iter_pmc, len(self._algorithms) + 1, *common_parameters))

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._max_weight = np.empty((self._n_iter_pmc, len(self._algorithms), *common_parameters))

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._M_eff = np.empty((self._n_iter_pmc, len(self._algorithms), *common_parameters))

		# [<component>, <sample>, <trial>, <frame>]
		self._markov_chains = np.empty((
			len(self._prior_mean), n_samples_metropolis_hastings, *common_parameters[1:]))

		# it must be done *in this class* because earlier the number of algorithms is still unknown
		if self._save_monte_carlo_samples:

			self._monte_carlo_samples = [([None] * len(self._n_particles)) for _ in enumerate(self._algorithms)]
			self._monte_carlo_weights = [([None] * len(self._n_particles)) for _ in enumerate(self._algorithms)]

			for i_alg, alg in enumerate(self._algorithms):

				for i_n_part, alg_n_part in enumerate(alg):

					self._monte_carlo_samples[i_alg][i_n_part] = [
						np.empty((
							# [<sample>,<component>,<trial>,<frame>]
							alg_n_part.n_particles_at_iteration(iter_pmc), len(self._prior_mean), self._n_trials,
							self._parameters["number of frames"])) for iter_pmc in range(self._n_iter_pmc)]

					self._monte_carlo_weights[i_alg][i_n_part] = [
						np.empty((
							# [<sample>,<trial>,<frame>]
							alg_n_part.n_particles_at_iteration(iter_pmc), self._n_trials,
							self._parameters["number of frames"])) for iter_pmc in range(self._n_iter_pmc)]

		# ----- HDF5

		# and so are those of the algorithms
		manu.util.write_strings_list_into_hdf5(
			self._f, self._h5py_prefix + 'algorithms/names',
			[alg[0].name for alg in self._algorithms] + [self.metropolis_hastings.name])

	def create_smc_algoritms(self):

		pmc = [
			mc.pmc.PopulationMonteCarlo(
				M, self._resampling_algorithm, self._resampling_criterion, self._inner_pf, self._prior_mean,
				self._prior_covar, self._prng, name='PMC')
			for M in self._n_particles]

		nonlinear_pmc = [
			mc.pmc.NonLinearPopulationMonteCarlo(
				M, self._resampling_algorithm, self._resampling_criterion, self._inner_pf, self._prior_mean,
				self._prior_covar, M_T, self._prng, name='NPMC')
			for M, M_T in zip(self._n_particles, self._M_Ts)]

		return [pmc, nonlinear_pmc]

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		# for every Monte Carlo trial
		for i_trial in range(self._n_trials):

			# the implementation of Metropolis-Hastings does not require initialization (it's automatically performed
			# by run at the end)
			self.metropolis_hastings.run(self._observations)

			self._markov_chains[:, :, i_trial, self._i_current_frame] = self.metropolis_hastings.chain

			self.initialize_pmc_algorithms()

			for i_iter in range(self._n_iter_pmc):

				self.run_pmc_algorithms(i_trial, i_iter)

				print(colorama.Fore.LIGHTMAGENTA_EX + self.metropolis_hastings.name + colorama.Style.RESET_ALL)

				for i_particles, n_particles in enumerate(self._n_particles):

					# the estimated parameters are saved in the slot for the *last* algorithm
					self._estimated_parameters[:, i_iter, -1, i_particles, i_trial, self._i_current_frame] =\
						self.metropolis_hastings.compute_half_sample_mean(n_particles*(i_iter +1 ))

					print(self._estimated_parameters[:, i_iter, -1, i_particles, i_trial, self._i_current_frame])

				print('=========')

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# in order to make sure the HDF5 files is valid...
		self._f.flush()

	def save_data(self, target_position):

		# let the *grandparent* class do its thing...
		simulations.base.Simulation.save_data(self, target_position)

		self._f.create_dataset(self._h5py_prefix + 'estimated parameters', data=self._estimated_parameters)
		self._f.create_dataset(self._h5py_prefix + 'maximum weight', data=self._max_weight)
		self._f.create_dataset(self._h5py_prefix + 'effective sample size', data=self._M_eff)

		# FIXME: the MCMC chains are stored in memory anyway (in order to avoid an "if" in "process_frame" above)
		if self._simulation_parameters["save MCMC chains"]:

			self._f.create_dataset(self._h5py_prefix + 'Markov chains', data=self._markov_chains)

		if self._save_monte_carlo_samples:

			for i_alg, alg in enumerate(self._algorithms):

				for i_n_part, alg_n_part in enumerate(alg):

					for iter_pmc, (samples, weights) in enumerate(
							zip(self._monte_carlo_samples[i_alg][i_n_part], self._monte_carlo_weights[i_alg][i_n_part])):

						dataset = self._f.create_dataset(
							self._h5py_prefix +
							'Monte Carlo samples/algorithms/{}/number of particles/{}/iteration/{}/samples'.format(
								alg_n_part.name, alg_n_part.n_particles, iter_pmc), data=samples)
						dataset.attrs['mapping'] = '[<sample>,<component>,<trial>,<frame>]'

						dataset = self._f.create_dataset(
							self._h5py_prefix +
							'Monte Carlo samples/algorithms/{}/number of particles/{}/iteration/{}/weights'.format(
								alg_n_part.name, alg_n_part.n_particles, iter_pmc), data=weights)
						dataset.attrs['mapping'] = '[<sample>,<trial>,<frame>]'

		# if a reference to an HDF5 was not received, that means the file was created by this object,
		# and hence it is responsible of closing it...
		if not self._h5py_file:
			# ...in order to make sure the HDF5 file is valid...
			self._f.close()


class NPMCvsInnerFilterNumberOfParticles(AbstractNPMC):

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='',
			n_processing_elements=None, n_sensors=None):

		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements,
			n_sensors)

		# the "inner" PFs are built
		inner_pfs = [centralized.TargetTrackingParticleFilter(
			n, self._resampling_algorithm, self._resampling_criterion, self._prior,
			self._transition_kernel, self._sensors
		) for n in self._n_particles_likelihood]

		# the list of algorithms to be run
		self._algorithms = []

		for n_particles, pf in zip(self._n_particles_likelihood, inner_pfs):

			# a NPMC algorithm embedding a PF with the given number of particles is built...
			nonlinear_pmc = [
				mc.pmc.NonLinearPopulationMonteCarlo(
					M, resampling_algorithm, resampling_criterion, pf, self._prior_mean, self._prior_covar, M_T,
					self._prng, name='NPMC (N = {})'.format(n_particles))
				for M, M_T in zip(self._n_particles, self._M_Ts)]

			# ...and added to the list
			self._algorithms.append(nonlinear_pmc)

		# ------------------------- accumulators

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		common_parameters = (self._n_iter_pmc, len(self._algorithms), len(self._n_particles), self._n_trials,
			parameters["number of frames"])

		# [<component>,<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._estimated_parameters = np.empty((len(self._prior_mean), *common_parameters))

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._max_weight = np.empty(common_parameters)

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._M_eff = np.empty(common_parameters)

		# ----- HDF5

		# the names of the algorithms are stored
		manu.util.write_strings_list_into_hdf5(
			self._f, self._h5py_prefix + 'algorithms/names',
			[alg[0].name for alg in self._algorithms])

	def save_data(self, target_position):

		# let the *grandparent* class do its thing...
		simulations.base.Simulation.save_data(self, target_position)

		self._f.create_dataset(self._h5py_prefix + 'estimated parameters', data=self._estimated_parameters)
		self._f.create_dataset(self._h5py_prefix + 'maximum weight', data=self._max_weight)
		self._f.create_dataset(self._h5py_prefix + 'effective sample size', data=self._M_eff)

		# if a reference to an HDF5 was not received, that means the file was created by this object,
		# and hence the latter is responsible of closing it...
		if not self._h5py_file:
			# ...in order to make sure the HDF5 file is valid...
			self._f.close()

	def process_frame(self, target_position, target_velocity):

		# let the grandparent class do its thing...
		AbstractNPMC.process_frame(self, target_position, target_velocity)

		# for every Monte Carlo trial
		for i_trial in range(self._n_trials):

			self.initialize_pmc_algorithms()

			for i_iter in range(self._n_iter_pmc):

				self.run_pmc_algorithms(i_trial, i_iter)

		# in order to make sure the HDF5 files is valid...
		self._f.flush()


class AMIS(AbstractNPMC):

	@property
	def monte_carlo_algorithm_name(self):

		return "Adaptive Multiple Importance Sampling"

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='',
			n_processing_elements=None, n_sensors=None):

		# let the parent class do its thing
		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements,
			n_sensors)

		# the "inner" PF is built
		inner_pf = centralized.TargetTrackingParticleFilter(
			self._n_particles_likelihood, self._resampling_algorithm, self._resampling_criterion, self._prior,
			self._transition_kernel, self._sensors
		)

		amis = [
			mc.amis.AdaptiveMultipleImportanceSampling(
				M, parameters[self.monte_carlo_algorithm_name]["number of iterations"], resampling_algorithm,
				resampling_criterion, inner_pf, self._prior_mean, self._prior_covar, self._prng, name='AMIS')
			for M in self._n_particles]

		nonlinear_amis = [
			mc.amis.NonLinearAdaptiveMultipleImportanceSampling(
				M, parameters[self.monte_carlo_algorithm_name]["number of iterations"], resampling_algorithm,
				resampling_criterion, inner_pf, self._prior_mean, self._prior_covar, M_T, self._prng, name='NAMIS')
			for M, M_T in zip(self._n_particles, self._M_Ts)]

		self._algorithms = [amis, nonlinear_amis]

		# ------------------------- accumulators

		# [<#particles>,<trial>,<frame>]
		common_parameters = (self._n_iter_pmc, len(self._algorithms), len(self._n_particles), self._n_trials, parameters["number of frames"])

		# [<component>,<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._estimated_parameters = np.empty((len(self._prior_mean), *common_parameters))

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._max_weight = np.empty(common_parameters)

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._M_eff = np.empty(common_parameters)

		# ----- HDF5

		# the names of the algorithms are stored
		manu.util.write_strings_list_into_hdf5(
			self._f, self._h5py_prefix + 'algorithms/names',
			[alg[0].name for alg in self._algorithms])

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		# for every Monte Carlo trial
		for i_trial in range(self._n_trials):

			self.initialize_pmc_algorithms()

			for i_iter in range(self._n_iter_pmc):

				self.run_pmc_algorithms(i_trial, i_iter)

				print('=========')

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# in order to make sure the HDF5 files is valid...
		self._f.flush()

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


class NPMCvAMIS(NPMC):

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='',
			n_processing_elements=None, n_sensors=None):

		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements, n_sensors)

		assert self._parameters["Adaptive Multiple Importance Sampling"]["number of iterations"] ==\
		       self._parameters["Population Monte Carlo"]["number of iterations"]

		assert self._parameters["Adaptive Multiple Importance Sampling"]["Nonlinear"]["number of clipped particles from overall number"] ==\
		       self._parameters["Population Monte Carlo"]["Nonlinear"]["number of clipped particles from overall number"]

	def create_smc_algoritms(self):

		# for the sake of convenience
		n_iterations = self._parameters["Adaptive Multiple Importance Sampling"]["number of iterations"]

		amis = [
			mc.amis.AdaptiveMultipleImportanceSampling(
				M, n_iterations, self._resampling_algorithm, self._resampling_criterion, self._inner_pf,
				self._prior_mean, self._prior_covar, self._prng, name='AMIS')
			for M in self._n_particles]

		nonlinear_amis = [
			mc.amis.NonLinearAdaptiveMultipleImportanceSampling(
				M, n_iterations, self._resampling_algorithm, self._resampling_criterion, self._inner_pf,
				self._prior_mean, self._prior_covar, M_T, self._prng, name='NAMIS')
			for M, M_T in zip(self._n_particles, self._M_Ts)]

		# the function to get M_T from the number of particles
		n_clipped_particles_from_overall = eval(
			self._parameters["Adaptive Multiple Importance Sampling"]["Nonlinear"]["number of clipped particles from overall number"])

		# a list of lists in which every list contains the M_Ts for a different number of particles (M)
		m_ts_for_every_iteration = [
			[n_clipped_particles_from_overall(M*(i+1)) for i in range(n_iterations)] for M in self._n_particles]

		vaying_clipped_number_nonlinear_amis = [
			mc.amis.VaryingClippedNumberNonLinearAdaptiveMultipleImportanceSampling(
				M, n_iterations, self._resampling_algorithm, self._resampling_criterion, self._inner_pf,
				self._prior_mean, self._prior_covar, M_T, self._prng, name='NAMIS_varying_Mt')
			for M, M_T in zip(self._n_particles, m_ts_for_every_iteration)]

		return super().create_smc_algoritms() + [amis, nonlinear_amis, vaying_clipped_number_nonlinear_amis]


class EffectiveSampleSizeBeforeAndAfterClipping(NPMC):

	def __init__(
			self, parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel,
			output_file_basename, pseudo_random_numbers_generators, h5py_file=None, h5py_prefix='',
			n_processing_elements=None, n_sensors=None):

		super().__init__(
			parameters, room, resampling_algorithm, resampling_criterion, prior, transition_kernel, output_file_basename,
			pseudo_random_numbers_generators, h5py_file, h5py_prefix, n_processing_elements, n_sensors)

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		common_parameters = (self._n_iter_pmc, len(self._algorithms)+1, len(self._n_particles), self._n_trials,
			parameters["number of frames"])

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._max_weight = np.empty(common_parameters)

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._M_eff = np.empty(common_parameters)

		# Metropolis-Hastings is not even run
		self._simulation_parameters["save MCMC chains"] = False

	def create_smc_algoritms(self):

		nonlinear_pmc = [
			mc.pmc.NonLinearPopulationMonteCarlo(
				M, self._resampling_algorithm, self._resampling_criterion, self._inner_pf, self._prior_mean,
				self._prior_covar, M_T, self._prng, name='NPMC')
			for M, M_T in zip(self._n_particles, self._M_Ts)]

		return [nonlinear_pmc]

	def run_pmc_algorithms(self, i_trial, i_iter):

		super().run_pmc_algorithms(i_trial, i_iter)

		# the maximum weight and ESS are saved in the "last slice" of the corresponding arrays
		for i_particles, alg_particles in enumerate(self._algorithms[-1]):

			self._max_weight[i_iter, -1, i_particles, i_trial, self._i_current_frame] = \
				alg_particles.clipped_weights.max()

			self._M_eff[i_iter, -1, i_particles, i_trial, self._i_current_frame] = \
				1. / np.sum(alg_particles.clipped_weights ** 2)

	def process_frame(self, target_position, target_velocity):

		# let the super-super class do its thing...
		AbstractNPMC.process_frame(self, target_position, target_velocity)

		# for every Monte Carlo trial
		for i_trial in range(self._n_trials):

			self.initialize_pmc_algorithms()

			for i_iter in range(self._n_iter_pmc):

				self.run_pmc_algorithms(i_trial, i_iter)

				print('=========')

		# in order to make sure the HDF5 files is valid...
		self._f.flush()