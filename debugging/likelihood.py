import numpy as np
import mc.util

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# to be run from, at least, mc.pmc.PopulationMonteCarlo.step
def explore(pf, observations):

	true_tx_power = 0.8
	true_min_power = 1e-5
	true_path_loss_exp = 3

	# log_tx_power, log_min_power, log_path_loss_exp = [np.log(tx_power)], [np.log(min_power)], [np.log(path_loss_exp)]
	#
	# for parameters_tuple in zip(log_tx_power, log_min_power, log_path_loss_exp):
	#
	# 	l = mc.util.loglikelihood(pf, observations, *parameters_tuple)
	# 	print(l)

	tx_power_array = np.linspace(true_tx_power/2., true_tx_power*2., num=20)
	min_power_array = np.linspace(true_min_power / 2., true_min_power * 2., num=20)
	path_loss_exp_array = np.linspace(true_path_loss_exp / 2., true_path_loss_exp * 2., num=20)

	log_tx_power_array, log_min_power_array, log_path_loss_exp_array =\
		np.log(tx_power_array), np.log(min_power_array), np.log(path_loss_exp_array)

	likelihood = np.empty((len(tx_power_array), len(min_power_array), len(path_loss_exp_array)))

	# mesh = np.meashgrid(np.log(tx_power_array), np.log(min_power_array), np.log(path_loss_exp_array))

	for i_log_tx_power, log_tx_power in enumerate(log_tx_power_array):

		print('log_tx_power = {}'.format(log_tx_power))

		for i_log_min_power, log_min_power in enumerate(log_min_power_array):

			for i_log_path_loss_exp, log_path_loss_exp in enumerate(log_path_loss_exp_array):

				likelihood[i_log_tx_power, i_log_min_power, i_log_path_loss_exp] =\
					mc.util.loglikelihood(pf, observations, log_tx_power, log_min_power, log_path_loss_exp)

				# print(likelihood[i_log_tx_power, i_log_min_power, i_log_path_loss_exp])

	print(likelihood)

	# this is a tuple
	i_max = np.unravel_index(likelihood.argmax(), likelihood.shape)

	print('maximum likelihood = {} at {}'.format(likelihood[i_max], i_max))
	print('transmiter power = {}, min power = {}, path loss exponent = {}'.format(
		tx_power_array[i_max[0]], min_power_array[i_max[1]], path_loss_exp_array[i_max[2]]))

	likelihood_at_true = mc.util.loglikelihood(
		pf, observations, np.log(true_tx_power), np.log(true_min_power), np.log(true_path_loss_exp))

	print('likelihood at true parameters: {}'.format(likelihood_at_true))

	i_closest = (
		np.abs(tx_power_array - true_tx_power).argmin(),
		np.abs(min_power_array - true_min_power).argmin(),
		np.abs(path_loss_exp_array - true_path_loss_exp).argmin())

	print('closest point to ground truth is transmiter power = {}, min power = {}, path loss exponent = {}, likelihood {}'.format(
		tx_power_array[i_closest[0]], min_power_array[i_closest[1]], path_loss_exp_array[i_closest[2]], likelihood[i_closest]))

	return tx_power_array, min_power_array, path_loss_exp_array, likelihood


def plot3d(x, y, Z):

	fig = plt.figure()
	ax = fig.gca(projection='3d')

	X, Y = np.meshgrid(x, y)
	ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# ax.bar([np.log(0.8)], -5000, [np.log(1e-5)])
	ax.scatter([0.8], [1e-5], [1], s=20)
	ax.scatter([1.22], [1.28e-5], [1], s=20)

	plt.show()

	return ax


def plot(tx_power_array, min_power_array, path_loss_exp_array, likelihood):

	plot3d(tx_power_array, min_power_array, likelihood[:, :, 7])