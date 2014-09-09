#! /usr/bin/env python3

import math
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
import scipy.io

# keys used to identify the different pseudo random numbers generators (they must coincide with those in the parameters file...)
PRNGsKeys = ["Sensors and Monte Carlo pseudo random numbers generator","Trajectory pseudo random numbers generator","topology pseudo random numbers generator"]

# in order to clock the execution time...
startTime = time.time()

# ----------------------------------------------------------------- arguments parser ---------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Distributed Resampling with Non-proportional Allocation.')
parser.add_argument('-r','--reproduce',type=argparse.FileType('r'),dest='parametersToBeReproducedFilename',help='repeat the simulation given by the parameters file')
commandArguments = parser.parse_args(sys.argv[1:])

# -----

# if this is a re-run of a previous simulation...
if commandArguments.parametersToBeReproducedFilename:
	
	# we open the file passed...
	with open(commandArguments.parametersToBeReproducedFilename.name,"rb") as f:
		
		# ...to extract th parameters and random state from the previous simulation
		parametersNrandomState = pickle.load(f)
	
	# parameters are restored
	parameters = parametersNrandomState[0]

# otherwise, we just read the parameters in the usual way
else:
	
	with open('parameters.json') as jsonData:
	
		# the parameters file is read to memory
		parameters = json.load(jsonData)

# number of particles per processing element (PE)
K = parameters["number of particles per PE"]

# different setups for the PEs
topologiesSettings = parameters['topologies']

# number of sensors, radius...
sensorsSettings = parameters["sensors"]

# number of time instants
nTimeInstants = parameters["number of time instants"]

# room dimensions
room = parameters["room"]

# parameters for the distribution of the initial state
priorDistribution = parameters["prior distribution"]

# state transition kernel parameters
stateTransition = parameters["state transition"]

# for the particle filters
SMCsettings = parameters["SMC"]

# parameters related to plotting
painterSettings = parameters["painter"]

# DRNA related
DRNAsettings = parameters["DRNA"]

# we use the "agg" backend if the DISPLAY variable is not present (the program is running without a display server) or the parameters file says so
useAgg = ('DISPLAY' not in os.environ) or (not painterSettings["use display server if available?"])

if useAgg:
	import matplotlib
	matplotlib.use('agg')

import matplotlib.pyplot as plt

# some of this modules also import matplotlib.pyplot, and since this should be done AFTER calling "matplotlib.use", they have been imported here and not at the very beginning
import target
import state
import sensor
import plot
from smc import particle_filter
from smc import resampling
import topology
import drnautil

# the name of the machine running the program (supposedly, using the socket module gives rise to portable code)
hostname = socket.gethostname()

# date and time
date = time.strftime("%a_%Y-%m-%d_%H:%M:%S")

# output data file
outputFile = hostname +'_' + date

# how numpy arrays are printed on screen is specified here
np.set_printoptions(precision=3,linewidth=100)

# ---------------------------------------------

# NOTE: most functions access global variables, though they don't modify them (except one of the handlers)

def saveData():
	
	if 'iFrame' not in globals() or iFrame==0:
		print('saveData: nothing to save...skipping')
		return

	# the mean of the MSE incurred by both PFs
	centralizedPF_MSE = (np.subtract(centralizedPF_pos[:,:,:iFrame,:],targetPosition[:,:,:iFrame,np.newaxis])**2).mean(axis=0).mean(axis=1)
	distributedPF_MSE = (np.subtract(distributedPF_pos[:,:,:iFrame,:],targetPosition[:,:,:iFrame,np.newaxis])**2).mean(axis=0).mean(axis=1)
	
	# ...the same for the error (euclidean distance)
	centralizedPF_error = np.sqrt((np.subtract(centralizedPF_pos[:,:,:iFrame,:],targetPosition[:,:,:iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)
	distributedPF_error = np.sqrt((np.subtract(distributedPF_pos[:,:,:iFrame,:],targetPosition[:,:,:iFrame,np.newaxis])**2).sum(axis=0)).mean(axis=1)

	# MSE vs time (only the results for the first topology are plotted)
	plot.distributedPFagainstCentralizedPF(np.arange(nTimeInstants),centralizedPF_MSE[:,0],distributedPF_MSE[:,0],
						painterSettings["file name prefix for the MSE vs time plot"] + '_' + outputFile + '_nFrames={}.eps'.format(repr(iFrame)),
						centralizedPFparameters={'label':'Centralized PF','color':painterSettings["color for the centralized PF"],'marker':painterSettings["marker for the centralized PF"]},
						distributedPFparameters={'label':'Distributed PF','color':painterSettings["color for the distributed PF"],'marker':painterSettings["marker for the distributed PF"]},
						figureId='MSE vs Time')

	# distance vs time (only the results for the first topology are plotted)
	plot.distributedPFagainstCentralizedPF(np.arange(nTimeInstants),centralizedPF_error[:,0],distributedPF_error[:,0],
						painterSettings["file name prefix for the euclidean distance vs time plot"] + '_' + outputFile + '_nFrames={}.eps'.format(repr(iFrame)),
						centralizedPFparameters={'label':'Centralized PF','color':painterSettings["color for the centralized PF"],'marker':painterSettings["marker for the centralized PF"]},
						distributedPFparameters={'label':'Distributed PF','color':painterSettings["color for the distributed PF"],'marker':painterSettings["marker for the distributed PF"]},
						figureId='Euclidean distance vs Time')

	# the aggregated weights are  normalized at ALL TIMES, for EVERY frame and EVERY topology
	normalizedAggregatedWeights = [np.rollaxis(np.divide(np.rollaxis(w[:,:,:iFrame],2,1),w[:,:,:iFrame].sum(axis=1)[:,:,np.newaxis]),2,1) for w in distributedPFaggregatedWeights]
	
	# ...the same data structured in a dictionary
	normalizedAggregatedWeightsDic = {'normalizedAggregatedWeights_{}'.format(i):array for i,array in enumerate(normalizedAggregatedWeights)}
	
	# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
	maxWeights = np.array([(w.max(axis=1)**DRNAsettings['q']).mean(axis=1) for w in normalizedAggregatedWeights])

	# evolution of the largest aggregated weight over time (only the results for the first topology are plotted)
	plot.aggregatedWeightsSupremumVsTime(maxWeights[0,:],aggregatedWeightsUpperBounds[0],
											 painterSettings["file name prefix for the aggregated weights supremum vs time plot"] + '_' + outputFile + '_nFrames={}.eps'.format(repr(iFrame)),DRNAsettings["exchange period"])

	# if requested, save the trajectory
	if painterSettings["display evolution?"]:
		if 'iTime' in globals() and iTime>0:
			painter.save(('trajectory_up_to_iTime={}_' + hostname + '_' + date + '.eps').format(repr(iTime)))

	# a dictionary encompassing all the data to be saved
	dataToBeSaved = dict(
			aggregatedWeightsUpperBounds = aggregatedWeightsUpperBounds,
			targetPosition = targetPosition[:,:,:iFrame],
			centralizedPF_pos = centralizedPF_pos[:,:,:iFrame,:],
			distributedPF_pos = distributedPF_pos[:,:,:iFrame,:],
			**normalizedAggregatedWeightsDic
		)
	
	# data is saved
	#np.savez('res_' + outputFile + '.npz',**dataToBeSaved)
	scipy.io.savemat('res_' + outputFile,dataToBeSaved)
	print('results saved in "{}"'.format('res_' + outputFile))

def saveParameters():
	
	# in a separate file with the same name as the data file but different extension...
	with open('res_' + outputFile + '.parameters',mode='wb') as f:
		
		#  ...parameters and pseudo random numbers generators are pickled
		pickle.dump((parameters,frozenPRNGs),f)

# ---------------------------------------------

# we'd rather have the coordinates of the corners stored as numpy arrays...
room["bottom left corner"] = np.array(room["bottom left corner"])
room["top right corner"] = np.array(room["top right corner"])

# it amounts to the width and height
roomDiagonalVector = room["top right corner"] - room["bottom left corner"]

# ------------------------------------------------------------------ random numbers ----------------------------------------------------------------------

# a dictionary with several "RandomState" (pseudo random number generator) objects
PRNGs = {}

# if this is a rerun of a previous simulation...
if commandArguments.parametersToBeReproducedFilename:
	
	# every pseudo random numbers generator...
	for key in PRNGsKeys:
		
		# ...is restored via the key
		PRNGs[key] = parametersNrandomState[1][key]
	
else:

	for withinParametersFileQuestion,key in zip(["load sensors and Monte Carlo pseudo random numbers generator?","load trajectory pseudo random numbers generator?","load topology pseudo random numbers generator?"],
								PRNGsKeys):
		
		# if loading the corresponding previous pseudo random numbers generator is requested...
		if parameters[withinParametersFileQuestion]:
			
			print('loading "{}"...'.format(key))
			
			with open(parameters[key],"rb") as f:
				
				# the previously "pickled" RandomState object is loaded
				PRNGs[key] = pickle.load(f)
				
		# otherwise, if a random seed is requested...
		else:
			
			# a new pseudo random numbers generator object is created...
			PRNGs[key] = np.random.RandomState()

			# ...and saved
			with open(parameters[key],mode='wb') as f:
				
				# the above created PRNG object is saved pickled into a file
				pickle.dump(PRNGs[key],f)

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
	saveData()
	
	# ...and the parameters as well
	saveParameters()

# Ctrl-C has not been pressed yet...well, if it has, then the program has not even reached here
ctrlCpressed = False

# handler for SIGINT is installed
signal.signal(signal.SIGINT, sigint_handler)

# handler for SIGUSR1 is installed
signal.signal(signal.SIGUSR1, sigusr1_handler)

# ----------------------------------------------------------- Processing Elements (PEs) ------------------------------------------------------------------

topologies = [getattr(topology,t['class'])(t['number of PEs'],K,DRNAsettings["exchanged particles maximum percentage"],t['parameters'],PRNG=PRNGs["topology pseudo random numbers generator"]) for t in topologiesSettings]

# we compute the upper bound for the supremum of the aggregated weights that should guarante convergence
aggregatedWeightsUpperBounds = [drnautil.supremumUpperBound(t['number of PEs'],DRNAsettings['c'],DRNAsettings['q'],DRNAsettings['epsilon']) for t in topologiesSettings]

# ------------------------------------------------------------- sensors-related stuff --------------------------------------------------------------------

# an object for computing the positions of the sensors is created and used
sensorsPositions = sensor.EquispacedOnRectangleSensorLayer(room["bottom left corner"],room["top right corner"]).getPositions(sensorsSettings["number"])

# the actual number of sensor might not be equal to that requested
sensorsSettings["number"] = sensorsPositions.shape[1]

# we build the array of sensors
sensors = [sensor.Sensor(sensorsPositions[:,i:i+1],sensorsSettings["radius"],
						 probDetection=sensorsSettings["probability of detection within the radius"],probFalseAlarm=sensorsSettings["probability of false alarm"],
						 PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"]) for i in range(sensorsSettings["number"])]

# ----------------------------------------------------------------- dynamic model ------------------------------------------------------------------------

# a object that represents the prior distribution...
prior = state.UniformBoundedPositionGaussianVelocityPrior(room["bottom left corner"],room["top right corner"],velocityVariance=priorDistribution["velocity variance"],PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a different one for the transition kernel
transitionKernel = state.BouncingWithinRectangleTransitionKernel(room["bottom left corner"],room["top right corner"],velocityVariance=stateTransition["velocity variance"],noiseVariance=stateTransition["position variance"],PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ------------------------------------------------------------------- SMC stuff --------------------------------------------------------------------------

# a resampling algorithm...
resamplingAlgorithm = resampling.MultinomialResamplingAlgorithm(PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a resampling criterion are needed for the particle filters
#resamplingCriterion = resampling.EffectiveSampleSizeBasedResamplingCriterion(SMCsettings["resampling ratio"])
resamplingCriterion = resampling.AlwaysResamplingCriterion()

# plain non-parallelized particle filter
PFsForTopologies = [particle_filter.CentralizedTargetTrackingParticleFilter(K*t.getNumberOfPEs(),resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors) for t in topologies]

# distributed particle filter
distributedPFsForTopologies = [particle_filter.TargetTrackingParticleFilterWithDRNA(
	DRNAsettings["exchange period"],t,upperBound,K,DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors
	) for t,upperBound in zip(topologies,aggregatedWeightsUpperBounds)]

#------------------------------------------------------------- metrics initialization --------------------------------------------------------------------

# we store the aggregated weights...
distributedPFaggregatedWeights = [np.empty((nTimeInstants,t.getNumberOfPEs(),parameters["number of frames"])) for t in topologies]

# ...and the position estimates
centralizedPF_pos,distributedPF_pos = np.empty((2,nTimeInstants,parameters["number of frames"],len(topologies))),np.empty((2,nTimeInstants,parameters["number of frames"],len(topologies)))

# there will be as many trajectories as frames
targetPosition = np.empty((2,nTimeInstants,parameters["number of frames"]))

#------------------------------------------------------------------ PF estimation  -----------------------------------------------------------------------

# the object representing the target
mobile = target.Target(prior,transitionKernel,PRNG=PRNGs["Trajectory pseudo random numbers generator"])

# NOTE: a "while loop" is here more convenient than a "for loop" because having the "iFrame" variable defined at all times after the processing has started (and finished) 
# allows to know hown many frames have actually been processed (if any)

iFrame = 0

while iFrame < parameters["number of frames"] and not ctrlCpressed:
	
	# a trajectory is simulated
	targetPosition[:,:,iFrame],targetVelocity = mobile.simulateTrajectory(sensors,nTimeInstants)

	# observations for all the sensors at every time instant (each list)
	# NOTE: conversion to float is done so that the observations (either 1 or 0) are amenable to be used in later computations
	observations = [np.array([sensor.detect(state.position(s[:,np.newaxis])) for sensor in sensors],dtype=float) for s in targetPosition[:,:,iFrame].T]
	
	for iTopology,(pf,distributedPf) in enumerate(zip(PFsForTopologies,distributedPFsForTopologies)):
		
		# initialization of the particle filters
		pf.initialize()
		distributedPf.initialize()

		if painterSettings['display evolution?']:
			
			# if this is the first iteration...
			if 'painter' in locals():
				
				# ...then, the previous figure is closed
				painter.close()

			# this object will handle graphics...
			painter = plot.RectangularRoomPainter(room["bottom left corner"],room["top right corner"],sensorsPositions,sleepTime=painterSettings["sleep time between updates"])

			# ...e.g., draw the sensors
			painter.setup()

		for iTime in range(nTimeInstants):

			print('---------- iFrame = {}, iTopology = {}, iTime = {}'.format(repr(iFrame),repr(iTopology),repr(iTime)))

			print('position:\n',targetPosition[:,iTime:iTime+1,iFrame])
			print('velocity:\n',targetVelocity[:,iTime:iTime+1])
			
			# particle filters are updated
			pf.step(observations[iTime])
			distributedPf.step(observations[iTime])
			
			# the mean computed by the centralized and distributed PFs
			centralizedPF_mean,distributedPF_mean = pf.computeMean(),distributedPf.computeMean()
			
			centralizedPF_pos[:,iTime:iTime+1,iFrame,iTopology],distributedPF_pos[:,iTime:iTime+1,iFrame,iTopology] = state.position(centralizedPF_mean),state.position(distributedPF_mean)
			
			# the aggregated weights of the different PEs in the distributed PF are stored
			distributedPFaggregatedWeights[iTopology][iTime,:,iFrame] = distributedPf.getAggregatedWeights()
			
			print('centralized PF\n',centralizedPF_mean)
			print('distributed PF\n',distributedPF_mean)
			
			if painterSettings["display evolution?"]:

				# the plot is updated with the position of the target...
				painter.updateTargetPosition(targetPosition[:,iTime:iTime+1,iFrame])
				
				# ...those estimated by the PFs
				painter.updateEstimatedPosition(state.position(centralizedPF_mean),identifier='centralized',color=painterSettings["color for the centralized PF"])
				painter.updateEstimatedPosition(state.position(distributedPF_mean),identifier='distributed',color=painterSettings["color for the distributed PF"])

				if painterSettings["display particles evolution?"]:

					# ...and those of the particles...
					painter.updateParticlesPositions(state.position(pf.getState()),identifier='centralized',color=painterSettings["color for the centralized PF"])
					painter.updateParticlesPositions(state.position(distributedPf.getState()),identifier='distributed',color=painterSettings["color for the distributed PF"])
	
	iFrame += 1
	
# plots and data are saved...
saveData()

# ...and the parameters too
saveParameters()

# ------------------------------------------------------------------ benchmarking  -----------------------------------------------------------------------

# for benchmarking purposes, it is assumed that the execution ends here
endTime = time.time()

# the elapsed time in seconds
elapsedTime = endTime-startTime

if elapsedTime>60*60*24:
	print('Execution time: {} days'.format(repr(elapsedTime/(60*60*24))))
elif elapsedTime>60*60:
	print('Execution time: {} hours'.format(repr(elapsedTime/(60*60))))
elif elapsedTime>60:
	print('Execution time: {} minutes'.format(repr(elapsedTime/60)))
else:
	print('Execution time: {} seconds'.format(repr(elapsedTime)))

# --------------------------------------------------------------------------------------------------------------------------------------------------------

# if using the agg backend (no pictures shown), there is no point in bringing up the interactive prompt before exiting
if not useAgg:
	import code
	code.interact(local=dict(globals(), **locals()))