#! /usr/bin/env python3

import numpy as np
import json
import timeit
import datetime
import sys
import os
import copy
import signal
import pickle
import argparse

import colorama

import region
import target
import state

import simulations.base
import simulations.drna
import simulations.mposterior
import simulations.npmc

# # in order to turn every warning into an exception
# np.seterr(all='raise')
# np.seterr(over='raise')

simulation_modules = [simulations.drna, simulations.mposterior, simulations.npmc]

# sys.path.append(os.path.join(os.environ['HOME'], 'python'))

import manu.util
import manu.smc.resampling

# keys used to identify the different pseudo random numbers generators
# (they must coincide with those in the parameters file...)
PRNGsKeys = [
	'Sensors and Monte Carlo pseudo random numbers generator',
	'Trajectory pseudo random numbers generator',
	'topology pseudo random numbers generator']

# in order to clock the execution time...
start_time = timeit.default_timer()

# -------------------------------------------- arguments parser --------------------------------------------------------

parser = argparse.ArgumentParser(description='Distributed Resampling with Non-proportional Allocation.')
parser.add_argument(
	'-r', '--reproduce', type=argparse.FileType('r'), dest='reproduce_filename',
	help='repeat the simulation given by the parameters file')

# a different number of frames when re-running a simulation
parser.add_argument(
	'-n', '--new-number-of-frames', dest='new_n_frames', type=int,
	help='set the number of frames regardless of the value in the parameters file (to be used along "-r")')

# the frame in which the re-running will start
parser.add_argument(
	'-i', '--index-of-first-frame', dest='i_first_frame', default=0, type=int,
	help='the index of the first frame to be (re) simulated (to be used along "-r")')

# the seed for a pseudo-random numbers generator that is, in turn, used to initialize those of the *actual* ones
parser.add_argument('-s', '--seed', type=int, help='a random seed to initialize the different PRNGs')

parser.add_argument(
	'-p', '--parameters-path', dest='parameters_file', type=argparse.FileType('r'), default='parameters.json',
	help='parameters file')

parser.add_argument(
	'-o', '--output-path', dest='output_path', default=os.getcwd(), help='path for output files')

command_arguments = parser.parse_args(sys.argv[1:])

# -----

with open(command_arguments.parameters_file.name) as json_data:

	# the parameters file is read to memory
	parameters = json.load(json_data)

# if a new number of frames was passed...
if command_arguments.new_n_frames:

	# ...it is used
	parameters['number of frames'] = command_arguments.new_n_frames

# number of time instants
n_time_instants = parameters["number of time instants"]

# room dimensions
settings_room = parameters["room"]

# state prior parameters
settings_prior_distribution = parameters["prior distribution"]

# state transition kernel parameters
settings_state_transition = parameters["state transition"]

# for the particle filters: type of simulation and the corresponding parameters
settings_simulation = parameters['simulations']

output_file_basename = os.path.join(command_arguments.output_path, 'res_' + manu.util.filename_from_host_and_date())

# how numpy arrays are printed on screen is specified here
np.set_printoptions(precision=3, linewidth=300)

# NOTE: most functions access global variables, though they don't modify them (except one of the handlers)


def save_parameters():

	# in a separate file with the same name as the data file but different extension...
	parameters_file = os.path.join(command_arguments.output_path, output_file_basename + '.parameters')

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

	assert os.path.splitext(command_arguments.reproduce_filename.name)[1] == ".hdf5"

	saved_PRNGs = simulations.base.SimpleSimulation.pseudo_random_numbers_generators_from_file(
			command_arguments.reproduce_filename.name)

	# every pseudo random numbers generator...
	for k in PRNGsKeys:

		# ...is restored
		PRNGs[k] = saved_PRNGs[k]

# if this is NOT a rerun of a previous simulation...
else:

	# if a random seed has been passed...
	if command_arguments.seed:

		# a "meta" PRNG to generate the seed of every required PRNG is built
		meta_prng = np.random.RandomState(command_arguments.seed)

		print(colorama.Fore.LIGHTWHITE_EX + 'a random seed has been received...' + colorama.Style.RESET_ALL)

		# the seed of a PRNG in python must be a 32 bits usigned integer
		max_seed = 2**32

	for key in PRNGsKeys:

		question_within_parameters_file = 'load ' + key[0][0].lower() + key[1:] + '?'

		# if the corresponding key is in the parameters file and is set to true...
		if (question_within_parameters_file in parameters) and parameters[question_within_parameters_file]:

			print('loading "{}"...'.format(key))

			with open(parameters[key], "rb") as f:

				# the previously "pickled" RandomState object is loaded
				PRNGs[key] = pickle.load(f)

		# otherwise, a random seed is needed...
		else:

			# if a file name is specified in the parameters file (not likely if the question is not present...)
			if key in parameters:

				filename = parameters[key]

			else:

				# the first word in the key followed by the usual prefix
				filename = key.split()[0] + '.RandomState'

			# the seed is a (pseudo)random number obtained from the meta-PRNG initialized above or None,
			# the latter meaning "pick a random seed" in the "RandomState" constructor below
			seed = meta_prng.randint(max_seed) if command_arguments.seed else None

			# a new pseudo random numbers generator object is created...
			PRNGs[key] = np.random.RandomState(seed)

			# ...and saved
			with open(filename, mode='wb') as f:

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

# an object representing the bounded region
room = region.Rectangle(settings_room["bottom left corner"], settings_room["top right corner"])

# the settings for the prior distribution are retrieved...
settings_state_prior = settings_prior_distribution[settings_prior_distribution['type']]

# and the corresponding object is instantiated...
prior = getattr(state, settings_state_prior['implementing class'])(
	room, **settings_state_prior['parameters'], PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# the settings for the state transition kernel...
settings_transition_kernel = settings_state_transition[settings_state_transition['type']]

# ...are used to instantiate the requested object
transitionKernel = getattr(state, settings_transition_kernel['implementing class'])(
	room,
	velocity_variance=settings_state_transition["velocity variance"],
	noise_variance=settings_state_transition["position variance"], step_duration=settings_state_transition['time step size'],
	PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"], **settings_transition_kernel['parameters'])

# ------------------------------------------------ SMC stuff -----------------------------------------------------------

# a resampling algorithm...
resampling_algorithm = manu.smc.resampling.MultinomialResamplingAlgorithm(
	PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a resampling criterion are needed for the particle filters
# resampling_criterion = manu.smc.resampling.EffectiveSampleSizeBasedResamplingCriterion(parameters["SMC"]["resampling ratio"])
resampling_criterion = manu.smc.resampling.AlwaysResamplingCriterion()

# -------------------------------------------------- other stuff  ------------------------------------------------------

# there will be as many trajectories as frames
target_position = np.empty((2, n_time_instants, parameters["number of frames"]))

# the object representing the target
mobile = target.Target(prior, transitionKernel, pseudo_random_numbers_generator=PRNGs["Trajectory pseudo random numbers generator"])

# the class of the "simulation" object to be created...
simulation_class = None

for m in simulation_modules:

	try:

		simulation_class = getattr(m, settings_simulation[parameters['simulation type']]['implementing class'])

	except AttributeError:

		pass

if not simulation_class:

	raise Exception("don't know about that simulation")

# ...is used to instantiate the latter
sim = simulation_class(
	parameters, room, resampling_algorithm, resampling_criterion, prior, transitionKernel, output_file_basename, PRNGs)

# if this is a re-run of a previous simulation
if command_arguments.reproduce_filename:

	# the pseudo-random numbers generators for the requested frame (default is 0) are extracted from the data file...
	saved_pseudo_random_numbers_generators = simulations.base.SimpleSimulation.pseudo_random_numbers_generators_from_file(
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