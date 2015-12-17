#! /usr/bin/env python3

import numpy as np
import json
import time
import timeit
import datetime
import sys
import os
import copy
import signal
import socket
import pickle
import argparse


# keys used to identify the different pseudo random numbers generators
# (they must coincide with those in the parameters file...)
PRNGsKeys = ['Sensors and Monte Carlo pseudo random numbers generator',
             'Trajectory pseudo random numbers generator', 'topology pseudo random numbers generator']

# in order to clock the execution time...
start_time = timeit.default_timer()

# -------------------------------------------- arguments parser --------------------------------------------------------

parser = argparse.ArgumentParser(description='Distributed Resampling with Non-proportional Allocation.')
parser.add_argument('-r', '--reproduce', type=argparse.FileType('r'), dest='reproduce_filename',
                    help='repeat the simulation given by the parameters file')

# a different number of frames when re-running a simulation
parser.add_argument('-n', '--new-number-of-frames', dest='new_n_frames', type=int)

# the frame in which the re-running will start
parser.add_argument('-i', '--index-of-first-frame', dest='i_first_frame', default=0, type=int)

command_arguments = parser.parse_args(sys.argv[1:])

# -----

with open('parameters.json') as jsonData:

	# the parameters file is read to memory
	parameters = json.load(jsonData)

# if a new number of frames was passed...
if command_arguments.new_n_frames:

	# ...it is used
	parameters['number of frames'] = command_arguments.new_n_frames

# number of time instants
n_time_instants = parameters["number of time instants"]

# room dimensions
settings_room = parameters["room"]

# state transition kernel parameters
settings_state_transition = parameters["state transition"]

# for the particle filters: type of simulation and the corresponding parameters
settings_simulation = parameters['simulations']

# we use the "agg" backend if the DISPLAY variable is not present
# (the program is running without a display server) or the parameters file says so
use_matplotlib_agg_backend = ('DISPLAY' not in os.environ) or (
	not parameters["painter"]["use display server if available?"])

if use_matplotlib_agg_backend:
	import matplotlib
	matplotlib.use('agg')

# some of this modules also import matplotlib.pyplot, and since this should be done AFTER calling "matplotlib.use",
# they have been imported here and not at the very beginning
import target
import state
from smc import resampling
import simulation

# the name of the machine running the program (supposedly, using the socket module gives rise to portable code)
hostname = socket.gethostname()

# date and time
date = time.strftime("%a_%Y-%m-%d_%H:%M:%S")

# output data file
output_file = hostname + '_' + date + '_' + str(os.getpid())

# how numpy arrays are printed on screen is specified here
np.set_printoptions(precision=3, linewidth=100)

# NOTE: most functions access global variables, though they don't modify them (except one of the handlers)


def save_parameters():

	# in a separate file with the same name as the data file but different extension...
	parameters_file = 'res_{}.parameters'.format(output_file)

	with open(parameters_file, mode='wb') as f:

		#  ...parameters and pseudo random numbers generators are pickled
		pickle.dump((parameters, frozen_pseudo_random_numbers_generators), f)

	print('parameters saved in "{}"'.format(parameters_file))

# ---------------------------------------------

# we'd rather have the coordinates of the corners stored as numpy arrays...
settings_room["bottom left corner"] = np.array(settings_room["bottom left corner"])
settings_room["top right corner"] = np.array(settings_room["top right corner"])

# ---------------------------------------------- random numbers --------------------------------------------------------

# a dictionary that will be filled with several "RandomState" (pseudo random number generator) objects
PRNGs = {}

# if this is a re-run of a previous simulation...
if command_arguments.reproduce_filename:

	assert os.path.splitext(command_arguments.reproduce_filename.name)[1]==".hdf5"

	saved_PRNGs = simulation.SimpleSimulation.pseudo_random_numbers_generators_from_file(
			command_arguments.reproduce_filename.name)

	# every pseudo random numbers generator...
	for k in PRNGsKeys:

		# ...is restored
		PRNGs[k] = saved_PRNGs[k]

# if this is NOT a rerun of a previous simulation...
else:

	for question_within_parameters_file, key in zip([
		"load sensors and Monte Carlo pseudo random numbers generator?",
		"load trajectory pseudo random numbers generator?",
		"load topology pseudo random numbers generator?"],
			PRNGsKeys):

		# if loading the corresponding previous pseudo random numbers generator is requested...
		if parameters[question_within_parameters_file]:

			print('loading "{}"...'.format(key))

			with open(parameters[key], "rb") as f:

				# the previously "pickled" RandomState object is loaded
				PRNGs[key] = pickle.load(f)

		# otherwise, if a random seed is requested...
		else:

			# a new pseudo random numbers generator object is created...
			PRNGs[key] = np.random.RandomState()

			# ...and saved
			with open(parameters[key], mode='wb') as f:

				# the above created PRNG object is saved pickled into a file
				pickle.dump(PRNGs[key], f)

# the PRNGs will change as the program runs, and we want to store them as they were in the beginning
frozen_pseudo_random_numbers_generators = copy.deepcopy(PRNGs)

# ---------------------------------------------- signals handling ------------------------------------------------------

# within the handler, once Ctrl-C is pressed once, the default behaviour is restored
original_sigint_handler = signal.getsignal(signal.SIGINT)


# the interrupt signal (ctrl-c) is handled by this function
def sigint_handler(signum, frame):

	# we may need to modify this global variable
	global ctrlCpressed

	if not ctrlCpressed:

		print('\nCtrl-C pressed...one more to exit the program right now...')
		ctrlCpressed = True

		# the default behaviour is restored
		signal.signal(signal.SIGINT, original_sigint_handler)


def sigusr1_handler(signum, frame):
	# plots and data are saved...
	sim.save_data(target_position)

	# ...and the parameters as well
	save_parameters()

# Ctrl-C has not been pressed yet...well, if it has, then the program has not even reached here
ctrlCpressed = False

# handler for SIGINT is installed
signal.signal(signal.SIGINT, sigint_handler)

## handler for SIGUSR1 is installed
# signal.signal(signal.SIGUSR1, sigusr1_handler)

# ----------------------------------------------- dynamic model --------------------------------------------------------

# a object that represents the prior distribution is instantiated...
prior = state.UniformBoundedPositionGaussianVelocityPrior(
	settings_room["bottom left corner"], settings_room["top right corner"],
	velocityVariance=parameters["prior distribution"]["velocity variance"],
	PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a different one for the transition kernel belonging to class...
settings_transition_kernel = settings_state_transition[settings_state_transition['type']]

# ...is instantiated here
transitionKernel = getattr(state, settings_transition_kernel['implementing class'])(
	settings_room["bottom left corner"], settings_room["top right corner"],
	velocityVariance=settings_state_transition["velocity variance"],
	noiseVariance=settings_state_transition["position variance"], stepDuration=settings_state_transition['time step size'],
	PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"], **settings_transition_kernel['parameters'])

# ------------------------------------------------ SMC stuff -----------------------------------------------------------

# a resampling algorithm...
resampling_algorithm = resampling.MultinomialResamplingAlgorithm(
	PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a resampling criterion are needed for the particle filters
# resampling_criterion = resampling.EffectiveSampleSizeBasedResamplingCriterion(parameters["SMC"]["resampling ratio"])
resampling_criterion = resampling.AlwaysResamplingCriterion()

# -------------------------------------------------- other stuff  ------------------------------------------------------

# there will be as many trajectories as frames
target_position = np.empty((2, n_time_instants, parameters["number of frames"]))

# the object representing the target
mobile = target.Target(prior, transitionKernel, pseudo_random_numbers_generator=PRNGs["Trajectory pseudo random numbers generator"])

# the class of the "simulation" object to be created...
simulation_class = getattr(simulation, settings_simulation[parameters['simulation type']]['implementing class'])

# ...is used to instantiate the latter
sim = simulation_class(parameters, resampling_algorithm, resampling_criterion, prior, transitionKernel, output_file, PRNGs)

# if this is a re-run of a previous simulation
if command_arguments.reproduce_filename:

	# the pseudo-random numbers generators for the requested frame (default is 0) are extracted from the data file...
	saved_pseudo_random_numbers_generators = simulation.SimpleSimulation.pseudo_random_numbers_generators_from_file(
		command_arguments.reproduce_filename.name, command_arguments.i_first_frame)

	# ...and used to set the state of existing ones
	for k in PRNGs:

		PRNGs[k].set_state(saved_pseudo_random_numbers_generators[k].get_state())

# the pseudo-random numbers generators are saved as they were at the beginning (unused)
sim.save_initial_pseudo_random_numbers_generators(frozen_pseudo_random_numbers_generators)

# ----------------------------------------------- PF estimation  -------------------------------------------------------

for iFrame in range(parameters["number of frames"]):

	if ctrlCpressed:

		break

	# a copy of the PRNGs is done before any of them is used in the loop
	copy_pseudo_random_numbers_generators = copy.deepcopy(PRNGs)

	# a trajectory is simulated
	target_position[:, :, iFrame], targetVelocity = mobile.simulate_trajectory(n_time_instants)

	# ...processed by the corresponding simulation
	sim.process_frame(target_position[:, :, iFrame], targetVelocity)

	sim.save_this_frame_pseudo_random_numbers_generators(copy_pseudo_random_numbers_generators)

# data is saved...
sim.save_data(target_position)

# ...and also the parameters (in a different file)
save_parameters()

# ------------------------------------------------ benchmarking  -------------------------------------------------------

# for benchmarking purposes, it is assumed that the execution ends here
end_time = timeit.default_timer()

# the elapsed time in seconds
elapsed_time = end_time - start_time

# a "timedelta" object is used to conveniently format the seconds
print('Execution time: {}'.format(datetime.timedelta(seconds=elapsed_time)))

# ----------------------------------------------------------------------------------------------------------------------

# if using the agg backend (no pictures shown), there is no point in bringing up the interactive prompt before exiting
if not use_matplotlib_agg_backend:

	import code
	code.interact(local=dict(globals(), **locals()))
