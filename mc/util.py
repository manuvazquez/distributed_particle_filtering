import sys
import os
import numpy as np

sys.path.append(os.path.join(os.environ['HOME'], 'python'))
import manu.smc.util


def normal_parameters_from_lognormal(mean, var):

	aux = 1 + var / mean ** 2
	var = np.log(aux)
	mean = np.log(mean) - np.log(np.sqrt(aux))

	return mean, var


def loglikelihood(pf, observations, log_tx_power, log_min_power, path_loss_exp):

	# the "inner" PF (for approximating the likelihood) is initialized
	pf.initialize()

	# the parameters of the sensors *within* the PF are set accordingly
	for s in pf.sensors:

		s.set_parameters(
			tx_power=np.exp(log_tx_power), minimum_amount_of_power=np.exp(log_min_power),
			path_loss_exponent=path_loss_exp)

	pf.reset_sensors_array()

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