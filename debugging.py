import numpy as np
import mc.util

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Likelihood:

	def __init__(self, true_tx_power=0.8, true_min_power=1e-5, true_path_loss_exp=3) -> None:

		self.true_tx_power = true_tx_power
		self.true_min_power = true_min_power
		self.true_path_loss_exp = true_path_loss_exp

		self.i_max = None
		self.i_closest = None

		self.tx_power_array = None
		self.min_power_array = None
		self.path_loss_exp_array = None
		
		self.likelihood = None

		plt.ion()

		# a,b = np.meshgrid(self.tx_power_array, self.min_power_array)
		# a,b = np.meshgrid(self.tx_power_array, self.path_loss_exp_array)
		# a,b = np.meshgrid(self.tx_power_array, self.path_loss_exp_array)

	# to be run from, at least, mc.pmc.PopulationMonteCarlo.step
	def explore(self, pf, observations):

		self.tx_power_array = np.linspace(self.true_tx_power/2., self.true_tx_power*2., num=10)
		self.min_power_array = np.linspace(self.true_min_power / 2., self.true_min_power * 2., num=10)
		self.path_loss_exp_array = np.linspace(self.true_path_loss_exp / 2., self.true_path_loss_exp * 2., num=10)

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

		ax.scatter([self.true_tx_power], [self.true_min_power], [self.true_path_loss_exp], s=20)

		plt.show()

		return ax

	def plot(self, i_fixed_dimension):

		self.plot3d(self.tx_power_array, self.min_power_array, self.likelihood[:, :, i_fixed_dimension].T)

	def fixed_min_power(self, i_fixed_dimension):

		self.plot3d(self.tx_power_array, self.path_loss_exp_array, self.likelihood[:, i_fixed_dimension, :].T)

	def fixed_tx_power(self, i_fixed_dimension):

		self.plot3d(self.min_power_array, self.path_loss_exp_array, self.likelihood[i_fixed_dimension, :, :].T)