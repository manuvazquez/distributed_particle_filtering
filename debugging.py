import numpy as np
import mc.util

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Likelihood:

	def __init__(self, true_tx_power=0.8, true_min_power=1e-5, true_path_loss_exp=3, n=10) -> None:

		self.true_tx_power = true_tx_power
		self.true_min_power = true_min_power
		self.true_path_loss_exp = true_path_loss_exp

		self.tx_power_array = np.linspace(true_tx_power/2., true_tx_power*2., num=n)
		self.min_power_array = np.linspace(true_min_power / 2., true_min_power * 2., num=n)
		self.path_loss_exp_array = np.linspace(true_path_loss_exp / 2., true_path_loss_exp * 2., num=n)

		self.i_max = None
		self.i_closest = None
		
		self.likelihood = None

		# non-blocking plots
		plt.ion()

		self.tx_power_grid_1st, self.min_power_grid_2nd = np.meshgrid(self.tx_power_array, self.min_power_array)
		_, self.path_loss_exp_grid_2nd = np.meshgrid(self.tx_power_array, self.path_loss_exp_array)
		self.min_power_grid_1st, _ = np.meshgrid(self.min_power_array, self.path_loss_exp_array)

		self.plots_z_value = 10000

	# to be run from, at least, mc.pmc.PopulationMonteCarlo.step
	def explore(self, pf, observations):

		log_tx_power_array, log_min_power_array, log_path_loss_exp_array =\
			np.log(self.tx_power_array), np.log(self.min_power_array), np.log(self.path_loss_exp_array)

		self.likelihood = np.empty((len(self.tx_power_array), len(self.min_power_array), len(self.path_loss_exp_array)))

		for i_log_tx_power, log_tx_power in enumerate(log_tx_power_array):

			print('log_tx_power = {}'.format(log_tx_power))

			for i_log_min_power, log_min_power in enumerate(log_min_power_array):

				for i_log_path_loss_exp, log_path_loss_exp in enumerate(log_path_loss_exp_array):

					self.likelihood[i_log_tx_power, i_log_min_power, i_log_path_loss_exp] =\
						mc.util.loglikelihood(pf, observations, log_tx_power, log_min_power, log_path_loss_exp)

		print(self.likelihood)

		# this is a tuple
		self.i_max = np.unravel_index(self.likelihood.argmax(), self.likelihood.shape)

		print('maximum likelihood = {} at transmiter power = {}, min power = {}, path loss exponent = {} ({})'.format(
			self.likelihood[self.i_max], self.tx_power_array[self.i_max[0]], self.min_power_array[self.i_max[1]],
			self.path_loss_exp_array[self.i_max[2]], self.i_max))

		likelihood_at_true = mc.util.loglikelihood(
			pf, observations, np.log(self.true_tx_power), np.log(self.true_min_power), np.log(self.true_path_loss_exp))

		print('likelihood at true parameters: {}'.format(likelihood_at_true))

		self.i_closest = (
			np.abs(self.tx_power_array - self.true_tx_power).argmin(),
			np.abs(self.min_power_array - self.true_min_power).argmin(),
			np.abs(self.path_loss_exp_array - self.true_path_loss_exp).argmin())

		print('closest point to ground truth is transmiter power = {}, min power = {}, path loss exponent = {}, likelihood {} ({})'.format(
			self.tx_power_array[self.i_closest[0]], self.min_power_array[self.i_closest[1]],
			self.path_loss_exp_array[self.i_closest[2]], self.likelihood[self.i_closest], self.i_closest))

	def plot3d(self, x, y, Z):

		fig = plt.figure()
		ax = fig.gca(projection='3d')

		X, Y = np.meshgrid(x, y)
		ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

		plt.show()

		return ax

	def plot(self, i_fixed_dimension, z_value=None):

		ax = self.plot3d(self.tx_power_array, self.min_power_array, self.likelihood[:, :, i_fixed_dimension].T)

		ax.scatter([self.true_tx_power], [self.true_min_power], [z_value if z_value else self.plots_z_value], s=20)

		plt.show()

	def fixed_min_power(self, i_fixed_dimension, z_value=None):

		ax = self.plot3d(self.tx_power_array, self.path_loss_exp_array, self.likelihood[:, i_fixed_dimension, :].T)

		ax.scatter([self.true_tx_power], [self.true_path_loss_exp], [z_value if z_value else self.plots_z_value], s=20)

		plt.show()

	def fixed_tx_power(self, i_fixed_dimension, z_value=None):

		ax = self.plot3d(self.min_power_array, self.path_loss_exp_array, self.likelihood[i_fixed_dimension, :, :].T)

		ax.scatter([self.true_min_power], [self.true_path_loss_exp], [z_value if z_value else self.plots_z_value], s=20)

		plt.show()