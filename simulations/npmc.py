import sys
import os
import types

import numpy as np
import scipy.stats
import colorama

import smc.particle_filter.particle_filter
import smc.particle_filter.centralized as centralized
import smc.estimator
import simulations.base

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util
import manu.util


def normal_parameters_from_lognormal(mean, var):

	var = 1 + var / mean ** 2
	mean = np.log(mean / np.sqrt(var))

	return mean, var


def loglikelihood(pf, observations, log_tx_power, log_min_power, path_loss_exp):

	# the "inner" PF (for approximating the likelihood) is initialized
	pf.initialize()

	# the parameters of the sensors *within* the PF are set accordingly
	for s in pf.sensors:

		s.set_parameters(
			tx_power=np.exp(log_tx_power), minimum_amount_of_power=np.exp(log_min_power),
			path_loss_exponent=path_loss_exp)

	res = 0.

	# the collection of observations is processed
	for obs in observations:

		# ...a step is taken
		pf.step(obs)

		# the loglikelihoods computed by the "inner" bootstrap filter
		loglikes = pf.last_unnormalized_loglikelihoods

		# logarithm of the average
		res += manu.smc.util.log_sum_from_individual_logs(loglikes) - np.log(len(loglikes))

	return res


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

		# samples are drawn from the mean and covariance
		self._samples = self._prng.multivariate_normal(self._mean, self._covar, size=self._n_particles)

		self._loglikelihoods = np.zeros(self._n_particles)

		for i_sample, (tx_power, min_power, path_loss_exp) in enumerate(self._samples):

			self._loglikelihoods[i_sample] = loglikelihood(self._pf, observations, tx_power, min_power, path_loss_exp)

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


class MetropolisHastings:

	def __init__(self, n_samples, pf, prior_mean, prior_covar, prng, burn_in_period, covar_ratio, name=None):

		# otherwise, the number of samples is not comparable to that of the IS algorithms
		assert burn_in_period == 0

		self.name = name

		self._n_samples = n_samples
		self._pf = pf
		self._prior_mean = prior_mean
		self._prior_covar = prior_covar
		self._prng = prng

		self._burn_in_period = burn_in_period

		# number of samples + number of samples for burn-in period + initial sample
		self._chain = np.empty((len(prior_mean), n_samples + burn_in_period))

		self._kernel_covar = prior_covar*covar_ratio

		self._i_sample = 0

		self._last_posterior = None

	def run(self, observations):

		# initialization
		self._chain[:, self._i_sample] = self._prng.multivariate_normal(self._prior_mean, self._prior_covar)

		# the logarithm of the prior for the sample just generated
		log_prior = np.log(scipy.stats.multivariate_normal.pdf(
			x=self._chain[:, self._i_sample], mean=self._prior_mean, cov=self._prior_covar))

		# the posterior (up to a proportionality constant) of the last sample
		self._last_posterior = loglikelihood(self._pf, observations, *self._chain[:, self._i_sample]) + log_prior

		# another sample is to be drawn
		self._i_sample += 1

		self.extend_chain(observations, self._burn_in_period + self._n_samples - 1)

		# reset (so that run can be called again)
		self._i_sample = 0

	@property
	def chain(self):

		return self._chain

	def compute_mean(self, n_samples):

		return self._chain[:, self._burn_in_period: self._burn_in_period + n_samples].mean(axis=1)

	def compute_half_sample_mean(self, n_samples):

		return self._chain[:, n_samples//2:  n_samples].mean(axis=1)

	def extend_chain(self, observations, n):

		# for every new sample to be added to the Markov chain
		for i in range(n):

			# a candidate sample is generated...
			candidate = self._prng.multivariate_normal(self._chain[:, self._i_sample-1], self._kernel_covar)

			# the log of the prior evaluated at the candidate
			log_prior = np.log(scipy.stats.multivariate_normal.pdf(
				x=candidate, mean=self._prior_mean, cov=self._prior_covar))

			# ...and its posterior (up to a proportionality constant) computed
			candidate_posterior = loglikelihood(self._pf, observations, *candidate) + log_prior

			# if the likelihood of the candidate is larger...
			if candidate_posterior > self._last_posterior:

				# the sample is accepted
				self._chain[:, self._i_sample] = candidate

				# the new posterior is kept
				self._last_posterior = candidate_posterior

			else:

				# the threshold does not depend on the proposal due the symmetry thereof
				threshold = np.exp(candidate_posterior - self._last_posterior)

				# if the threshold is essentially or the random sample is larger
				if np.isclose(threshold, 0.) or (self._prng.random_sample() > threshold):

					# the new sample is equal to the previous
					self._chain[:, self._i_sample] = self._chain[:, self._i_sample-1]

				else:

					# the sample is accepted
					self._chain[:, self._i_sample] = candidate

					# the new posterior is kept
					self._last_posterior = candidate_posterior

			self._i_sample += 1

		# import code
		# code.interact(local=dict(globals(), **locals()))


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
		self._n_particles = self._simulation_parameters["number of particles"]

		self._n_trials = self._simulation_parameters["number of trials"]

		self._n_iter_pmc = parameters["Population Monte Carlo"]["number of iterations"]
		n_clipped_particles_from_overall = eval(
			parameters["Population Monte Carlo"]["Nonlinear"]["number of clipped particles from overall number"])

		burn_in_period_metropolis_hastings = parameters["Metropolis-Hastings"]["burn-in period"]
		kernel_prior_covar_ratio_metropolis_hastings = parameters["Metropolis-Hastings"]["ratio kernel-prior covariances"]

		# the above parameter should be a function
		assert isinstance(n_clipped_particles_from_overall, types.FunctionType)

		# the number of particles to be clipped is obtained using this function
		M_Ts = [n_clipped_particles_from_overall(M) for M in self._n_particles]

		# the pseudo random numbers generator this class will be using
		prng = self._PRNGs['Sensors and Monte Carlo pseudo random numbers generator']

		# the "inner" PF is built
		inner_pf = centralized.TargetTrackingParticleFilter(
			self._n_particles_likelihood, self._resampling_algorithm, self._resampling_criterion,
			self._prior, self._transition_kernel, self._sensors
		)

		# ------------------------- prior

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

		n_samples_metropolis_hastings = self._n_iter_pmc*self._n_particles[-1]

		# ------------------------- algorithms

		pmc = [
			PopulationMonteCarlo(
				M, resampling_algorithm, resampling_criterion, inner_pf, prior_mean, prior_covar, prng, name='PMC')
			for M in self._n_particles]

		nonlinear_pmc = [
			NonLinearPopulationMonteCarlo(
				M, resampling_algorithm, resampling_criterion, inner_pf, prior_mean, prior_covar, M_T, prng, name='NPMC')
			for M, M_T in zip(self._n_particles, M_Ts)]

		# nonlinear_pmc_only_covar = [
		# 	NonLinearPopulationMonteCarloCovarOnly(
		# 		M, resampling_algorithm, resampling_criterion, inner_pf, prior_mean, prior_covar, M_T, prng,
		# 		name='NPMC_covar')
		# 	for M, M_T in zip(self._n_particles, M_Ts)]

		# Metropolis-Hastings algorithm is run considering the larger number of samples and iterations
		self.metropolis_hastings = MetropolisHastings(
			n_samples_metropolis_hastings, inner_pf, prior_mean, prior_covar, prng, burn_in_period_metropolis_hastings,
			kernel_prior_covar_ratio_metropolis_hastings, name='MetropolisHastings')

		# self._algorithms = [pmc, nonlinear_pmc, nonlinear_pmc_only_covar]
		self._algorithms = [pmc, nonlinear_pmc]

		# ------------------------- accumulators

		# [<component>,<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._estimated_parameters = np.empty((
			len(prior_mean), self._n_iter_pmc, len(self._algorithms) + 1, len(self._n_particles), self._n_trials,
			parameters["number of frames"]))

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._max_weight = np.empty((
			self._n_iter_pmc, len(self._algorithms), len(self._n_particles), self._n_trials,
			parameters["number of frames"]))

		# [<iteration>,<algorithm>,<#particles>,<trial>,<frame>]
		self._M_eff = np.empty((
			self._n_iter_pmc, len(self._algorithms), len(self._n_particles), self._n_trials,
			parameters["number of frames"]))

		# [<component>, <sample>, <trial>, <frame>]
		self._markov_chains = np.empty((
			len(prior_mean), n_samples_metropolis_hastings, self._n_trials, parameters["number of frames"]))

		# ----- HDF5

		# the names of the parameters are stored
		manu.util.write_strings_list_into_hdf5(
			self._f, self._h5py_prefix + 'parameters',
			['transmitter_power', 'minimum_amount_of_power', 'path_loss_exponent'])

		# and so are those of the algorithms
		manu.util.write_strings_list_into_hdf5(
			self._f, self._h5py_prefix + 'algorithms/names',
			[alg[0].name for alg in self._algorithms] + [self.metropolis_hastings.name])

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

		self._f.create_dataset(self._h5py_prefix + 'Markov chains', data=self._markov_chains)

		# if a reference to an HDF5 was not received, that means the file was created by this object,
		# and hence it is responsible of closing it...
		if not self._h5py_file:
			# ...in order to make sure the HDF5 file is valid...
			self._f.close()

	def process_frame(self, target_position, target_velocity):

		# let the super class do its thing...
		super().process_frame(target_position, target_velocity)

		# for every Monte Carlo trial
		for i_trial in range(self._n_trials):

			self.metropolis_hastings.run(self._observations)

			self._markov_chains[:, :, i_trial, self._i_current_frame] = self.metropolis_hastings.chain

			# algorithms are initialized
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

				print(colorama.Fore.LIGHTMAGENTA_EX + self.metropolis_hastings.name + colorama.Style.RESET_ALL)

				for n_particles in self._n_particles:

					self._estimated_parameters[:, i_iter, i_alg + 1, i_particles, i_trial, self._i_current_frame] =\
						self.metropolis_hastings.compute_half_sample_mean(n_particles*(i_iter +1 ))

					print(self._estimated_parameters[:, i_iter, i_alg + 1, i_particles, i_trial, self._i_current_frame])

				print('=========')

		# import code
		# code.interact(local=dict(globals(), **locals()))

		# in order to make sure the HDF5 files is valid...
		self._f.flush()
