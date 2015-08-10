#! /usr/bin/env python3

import numpy as np
import json
import time
import sys
import os
import copy
import signal
import socket
import pickle
import argparse

# def info(type, value, tb):
#    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
#       # we are in interactive mode or we don't have a tty-like device, so we call the default hook
#       sys.__excepthook__(type, value, tb)
#    else:
#       import traceback, pdb
#       # we are NOT in interactive mode, print the exception...
#       traceback.print_exception(type, value, tb)
#       print
#       # ...then start the debugger in post-mortem mode.
#       pdb.pm()
#
# sys.excepthook = info

# # so that numpy takes a specified measure when an "error" occurs
# np.seterr(divide='raise')
# np.seterr(all='raise')

# keys used to identify the different pseudo random numbers generators
# (they must coincide with those in the parameters file...)
PRNGsKeys = ['Sensors and Monte Carlo pseudo random numbers generator',
             'Trajectory pseudo random numbers generator', 'topology pseudo random numbers generator']

# in order to clock the execution time...
startTime = time.time()

# ----------------------------------------------------------------- arguments parser ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Distributed Resampling with Non-proportional Allocation.')
parser.add_argument('-r', '--reproduce', type=argparse.FileType('r'), dest='parametersToBeReproducedFilename',
                    help='repeat the simulation given by the parameters file')
commandArguments = parser.parse_args(sys.argv[1:])

# -----

# a dictionary that will be filled with several "RandomState" (pseudo random number generator) objects
PRNGs = {}

# if this is a re-run of a previous simulation...
if commandArguments.parametersToBeReproducedFilename:

	# we open the file passed...
	with open(commandArguments.parametersToBeReproducedFilename.name, "rb") as f:

		# ...to extract the parameters and random state from the previous simulation
		parameters,randomStates = pickle.load(f)

	# every pseudo random numbers generator...
	for key in PRNGsKeys:
		# ...is restored via the key
		PRNGs[key] = randomStates[key]

# otherwise, we just read the parameters in the usual way
else:

	with open('parameters.json') as jsonData:

		# the parameters file is read to memory
		parameters = json.load(jsonData)

# number of time instants
nTimeInstants = parameters["number of time instants"]

# room dimensions
roomSettings = parameters["room"]

# state transition kernel parameters
stateTransitionSettings = parameters["state transition"]

# for the particle filters
# type of simulation and the corresponding parameters
simulationSettings = parameters['simulations']

# we use the "agg" backend if the DISPLAY variable is not present (the program is running without a display server) or the parameters file says so
useAgg = ('DISPLAY' not in os.environ) or (not parameters["painter"]["use display server if available?"])

if useAgg:
	import matplotlib

	matplotlib.use('agg')

# some of this modules also import matplotlib.pyplot, and since this should be done AFTER calling "matplotlib.use", they have been imported here and not at the very beginning
import target
import state
import sensor
from smc import resampling
import simulation

# the name of the machine running the program (supposedly, using the socket module gives rise to portable code)
hostname = socket.gethostname()

# output data file
outputFile = hostname + '_' + str(os.getpid())

# how numpy arrays are printed on screen is specified here
np.set_printoptions(precision=3, linewidth=100)

# ---------------------------------------------

# NOTE: most functions access global variables, though they don't modify them (except one of the handlers)

def saveParameters():
	# in a separate file with the same name as the data file but different extension...
	with open('res_' + outputFile + '.parameters', mode='wb') as f:
		#  ...parameters and pseudo random numbers generators are pickled
		pickle.dump((parameters, frozenPRNGs), f)

# ---------------------------------------------

# we'd rather have the coordinates of the corners stored as numpy arrays...
roomSettings["bottom left corner"] = np.array(roomSettings["bottom left corner"])
roomSettings["top right corner"] = np.array(roomSettings["top right corner"])

# ------------------------------------------------------------------ random numbers ----------------------------------------------------------------------

# if this is NOT a rerun of a previous simulation... (dictionary with PRNGs is empty)
if not PRNGs:

	for withinParametersFileQuestion, key in zip(["load sensors and Monte Carlo pseudo random numbers generator?",
	                                              "load trajectory pseudo random numbers generator?",
	                                              "load topology pseudo random numbers generator?"],
	                                             PRNGsKeys):

		# if loading the corresponding previous pseudo random numbers generator is requested...
		if parameters[withinParametersFileQuestion]:

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
frozenPRNGs = copy.deepcopy(PRNGs)

# ---------------------------------------------------------------- signals handling ----------------------------------------------------------------------

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
	sim.saveData(targetPosition)

	# ...and the parameters as well
	saveParameters()

# Ctrl-C has not been pressed yet...well, if it has, then the program has not even reached here
ctrlCpressed = False

# handler for SIGINT is installed
signal.signal(signal.SIGINT, sigint_handler)

## handler for SIGUSR1 is installed
# signal.signal(signal.SIGUSR1, sigusr1_handler)

# ----------------------------------------------------------------- dynamic model ------------------------------------------------------------------------

# a object that represents the prior distribution is instantiated...
prior = state.UniformBoundedPositionGaussianVelocityPrior(roomSettings["bottom left corner"],
                                                          roomSettings["top right corner"],
                                                          velocityVariance=parameters["prior distribution"][
	                                                          "velocity variance"], PRNG=PRNGs[
		"Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a different one for the transition kernel belonging to class...
transitionKernelSettings = stateTransitionSettings[stateTransitionSettings['type']]

# ...is instantiated here
transitionKernel = getattr(state, transitionKernelSettings['implementing class'])(roomSettings["bottom left corner"],
                                                                                  roomSettings["top right corner"],
                                                                                  velocityVariance=
                                                                                  stateTransitionSettings[
	                                                                                  "velocity variance"],
                                                                                  noiseVariance=stateTransitionSettings[
	                                                                                  "position variance"],
                                                                                  stepDuration=stateTransitionSettings[
	                                                                                  'time step size'], PRNG=PRNGs[
		"Sensors and Monte Carlo pseudo random numbers generator"],
                                                                                  **transitionKernelSettings[
	                                                                                  'parameters'])

# ------------------------------------------------------------------- SMC stuff --------------------------------------------------------------------------

# a resampling algorithm...
resamplingAlgorithm = resampling.MultinomialResamplingAlgorithm(
	PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a resampling criterion are needed for the particle filters
# resamplingCriterion = resampling.EffectiveSampleSizeBasedResamplingCriterion(parameters["SMC"]["resampling ratio"])
resamplingCriterion = resampling.AlwaysResamplingCriterion()

# -------------------------------------------------------------------- other stuff  -----------------------------------------------------------------------

# there will be as many trajectories as frames
targetPosition = np.empty((2, nTimeInstants, parameters["number of frames"]))

# the object representing the target
mobile = target.Target(prior, transitionKernel, PRNG=PRNGs["Trajectory pseudo random numbers generator"])

# the class of the "simulation" object to be created...
simulationClass = getattr(simulation, simulationSettings[simulationSettings['type']]['implementing class'])

# ...is used to instantiate the latter
sim = simulationClass(parameters, resamplingAlgorithm, resamplingCriterion, prior, transitionKernel, outputFile, PRNGs)

# ------------------------------------------------------------------ PF estimation  -----------------------------------------------------------------------

# NOTE: a "while loop" is here more convenient than a "for loop" because having the "iFrame" variable defined at all times after the processing has started (and finished) 
# allows to know hown many frames have actually been processed (if any)

iFrame = 0

while iFrame < parameters["number of frames"] and not ctrlCpressed:
	# a trajectory is simulated
	targetPosition[:, :, iFrame], targetVelocity = mobile.simulateTrajectory(nTimeInstants)

	# ...processed by the corresponding simulation
	sim.processFrame(targetPosition[:, :, iFrame], targetVelocity)

	iFrame += 1

# plots and data are saved...
sim.saveData(targetPosition)

# ...and the parameters too
saveParameters()

# ------------------------------------------------------------------ benchmarking  -----------------------------------------------------------------------

# for benchmarking purposes, it is assumed that the execution ends here
endTime = time.time()

# the elapsed time in seconds
elapsedTime = endTime - startTime

if elapsedTime > 60 * 60 * 24:
	print('Execution time: {} days'.format(repr(elapsedTime / (60 * 60 * 24))))
elif elapsedTime > 60 * 60:
	print('Execution time: {} hours'.format(repr(elapsedTime / (60 * 60))))
elif elapsedTime > 60:
	print('Execution time: {} minutes'.format(repr(elapsedTime / 60)))
else:
	print('Execution time: {} seconds'.format(repr(elapsedTime)))

# --------------------------------------------------------------------------------------------------------------------------------------------------------

# if using the agg backend (no pictures shown), there is no point in bringing up the interactive prompt before exiting
if not useAgg:
	import code
	code.interact(local=dict(globals(), **locals()))
