#! /usr/bin/env python3

import math
import numpy as np
import json
import time
import sys
import os
import signal
import socket
import pickle
import argparse

# keys used to identify the different pseudo random numbers generators (they must coincide with those in the parameters file...)
PRNGsKeys = ["Sensors and Monte Carlo pseudo random numbers generator","Trajectory pseudo random numbers generator","PEs network pseudo random numbers generator"]

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
PEs = parameters["PEs Network"]

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
import Target
import State
import Sensor
import Painter
from smc import ParticleFilter
from smc import Resampling
import PEsNetwork

# the name of the machine running the program (supposedly, using the socket module gives rise to portable code)
hostname = socket.gethostname()

# date and time
date = time.strftime("%a_%d_%H:%M:%S")

# output data file
outputFile = hostname +'_' + date

# how numpy arrays are printed on screen is specified here
np.set_printoptions(precision=3,linewidth=100)

# ---------------------------------------------

# NOTE: this function accesses global variable, though it doesn't modify them...
def saveData():
	
	if 'iFrame' not in globals() or iFrame==0:
		print('saveData: nothing to save...skipping')
		return

	# the mean of the MSE incurred by both PFs
	centralizedPF_MSE = (np.subtract(centralizedPF_pos[:,:,:iFrame],targetPosition[:,:,np.newaxis])**2).mean(axis=0).mean(axis=1)
	distributedPF_MSE = (np.subtract(distributedPF_pos[:,:,:iFrame],targetPosition[:,:,np.newaxis])**2).mean(axis=0).mean(axis=1)
	
	# ...the same for the error (euclidean distance)
	centralizedPF_error = np.sqrt((np.subtract(centralizedPF_pos[:,:,:iFrame],targetPosition[:,:,np.newaxis])**2).sum(axis=0)).mean(axis=1)
	distributedPF_error = np.sqrt((np.subtract(distributedPF_pos[:,:,:iFrame],targetPosition[:,:,np.newaxis])**2).sum(axis=0)).mean(axis=1)

	# MSE vs time
	Painter.plotDistributedAgainstCentralizedVsTime(centralizedPF_MSE,distributedPF_MSE,
						painterSettings["color for the centralized PF"],painterSettings["color for the distributed PF"],painterSettings["marker for the centralized PF"],painterSettings["marker for the distributed PF"],
						'MSE vs Time',painterSettings["file name prefix for the MSE vs time plot"] + '_' + outputFile + '_nFrames={}.eps'.format(repr(iFrame)))

	# distance vs time
	Painter.plotDistributedAgainstCentralizedVsTime(centralizedPF_error,distributedPF_error,
						painterSettings["color for the centralized PF"],painterSettings["color for the distributed PF"],painterSettings["marker for the centralized PF"],painterSettings["marker for the distributed PF"],
						'Euclidean distance vs Time',painterSettings["file name prefix for the euclidean distance vs time plot"] + '_' + outputFile + '_nFrames={}.eps'.format(repr(iFrame)))

	# the aggregated weights are  normalized at ALL TIMES and for EVERY frame
	normalizedAggregatedWeights = np.rollaxis(np.divide(np.rollaxis(distributedPFaggregatedWeights[:,:,:iFrame],2,1),distributedPFaggregatedWeights[:,:,:iFrame].sum(axis=1)[:,:,np.newaxis]),2,1)

	# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
	maxWeights = (normalizedAggregatedWeights.max(axis=1)**4).mean(axis=1)**(1/4)

	# evolution of the largest aggregated weight over time
	Painter.plotAggregatedWeightsSupremumVsTime(maxWeights,distributedPf.getAggregatedWeightsUpperBound(),
											 painterSettings["file name prefix for the aggregated weights supremum vs time plot"] + '_' + outputFile + '_nFrames={}.eps'.format(repr(iFrame)),DRNAsettings["exchange period"],True)

	# if requested, save the trajectory
	if painterSettings["display evolution?"]:
		if 'iTime' in globals() and iTime>0:
			painter.save(('trajectory_up_to_iTime={}_' + hostname + '_' + date + '.eps').format(repr(iTime)))

	# a dictionary encompassing all the data to be saved
	dataToBeSaved = {
			'normalizedAggregatedWeights': normalizedAggregatedWeights,
			#'distributedPFaggregatedWeights': distributedPFaggregatedWeights[:,:,:iFrame],
			'aggregatedWeightsUpperBound': distributedPf.getAggregatedWeightsUpperBound(),
			'targetInitialPosition': targetInitialPosition,
			'targetInitialVelocity': targetInitialVelocity,
			'targetPosition': targetPosition,
			'targetVelocity': targetVelocity,
			'centralizedPF_pos': centralizedPF_pos,
			'distributedPF_pos': distributedPF_pos
		}
	
	# data is saved
	np.savez('res_' + outputFile + '.npz',**dataToBeSaved)
		
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

	for withinParametersFileQuestion,key in zip(["load sensors and Monte Carlo pseudo random numbers generator?","load trajectory pseudo random numbers generator?","load PEs network pseudo random numbers generator?"],
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

# ---------------------------------------------------------------- parameters saving  --------------------------------------------------------------------

	# in a separate file with the same name as the data file but different extension...
	with open('res_' + outputFile + '.parameters',mode='wb') as f:
		
		#  ...parameters and pseudo random numbers generators are pickled
		pickle.dump((parameters,PRNGs),f)

# ---------------------------------------------------------------- signals handling ----------------------------------------------------------------------

# within the handler, once Ctrl-C is pressed once, the default behaviour is restored
original_sigint_handler = signal.getsignal(signal.SIGINT)

# the interrupt signal (ctrl-c) is handled by this function
def sigint_handler(signum, frame):
	
	print('wap')
	
	# we may need to modify this global variable
	global ctrlCpressed

	if not ctrlCpressed:
			
		print('\nCtrl-C pressed...one more to exit the program right now...')
		ctrlCpressed = True

		# the default behaviour is restored
		signal.signal(signal.SIGINT, original_sigint_handler)
	
def sigusr1_handler(signum, frame):
	
	# plots and data are saved
	saveData()

# Ctrl-C has not been pressed yet...well, if it has, then the program has not even reached here
ctrlCpressed = False

# handler for SIGINT is installed
signal.signal(signal.SIGINT, sigint_handler)

# handler for SIGUSR1 is installed
signal.signal(signal.SIGUSR1, sigusr1_handler)

# ----------------------------------------------------------- Processing Elements (PEs) ------------------------------------------------------------------

# a PEs network is created...used later to get the exchange tuples

if PEs["topology"]=="Customized":
	
	PEsNetwork = PEsNetwork.Customized(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],[[1,3],[0,2],[1,9],[0,4],[3,5],[4,6],[5,7],[6,8],[7,9],[2,8]],
									PRNGs["PEs network pseudo random numbers generator"])
	
elif PEs["topology"]=="Ring":
	
	PEsNetwork = PEsNetwork.Ring(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],PRNGs["PEs network pseudo random numbers generator"])
	
elif PEs["topology"]=="Mesh":
	
	PEsNetwork = PEsNetwork.Mesh(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],PEs["Mesh"]["neighbours"],*PEs["Mesh"]["geometry"][0],
							  PRNG=PRNGs["PEs network pseudo random numbers generator"])
	
elif PEs["topology"]=="FullyConnected":
	
	PEsNetwork = PEsNetwork.FullyConnected(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],PRNGs["PEs network pseudo random numbers generator"])
	
elif PEs["topology"]=="FullyConnectedWithRandomLinksRemoved":

	PEsNetwork = PEsNetwork.FullyConnectedWithRandomLinksRemoved(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],PEs["FullyConnectedWithRandomLinksRemoved"]["number of links to be removed"],
															  PRNGs["PEs network pseudo random numbers generator"])
	
else:
	print('PEs network topology not supported...')
	raise SystemExit(0)

# ------------------------------------------------------------- sensors-related stuff --------------------------------------------------------------------

# an object for computing the positions of the sensors is created and used
sensorsPositions = Sensor.EquispacedOnRectangleSensorLayer(room["bottom left corner"],room["top right corner"]).getPositions(sensorsSettings["number"])

# the actual number of sensor might not be equal to that requested
sensorsSettings["number"] = sensorsPositions.shape[1]

# we build the array of sensors
sensors = [Sensor.Sensor(sensorsPositions[:,i:i+1],sensorsSettings["radius"],PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"]) for i in range(sensorsSettings["number"])]

# ----------------------------------------------------------------- dynamic model ------------------------------------------------------------------------

# a object that represents the prior distribution...
prior = State.UniformBoundedPositionGaussianVelocityPrior(room["bottom left corner"],room["top right corner"],velocityVariance=priorDistribution["velocity variance"],PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a different one for the transition kernel
transitionKernel = State.BouncingWithinRectangleTransitionKernel(room["bottom left corner"],room["top right corner"],velocityVariance=stateTransition["velocity variance"],noiseVariance=stateTransition["position variance"],PRNG=PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ------------------------------------------------------------------- SMC stuff --------------------------------------------------------------------------

# a resampling algorithm...
resamplingAlgorithm = Resampling.MultinomialResamplingAlgorithm(PRNGs["Sensors and Monte Carlo pseudo random numbers generator"])

# ...and a resampling criterion are needed for the particle filters
#resamplingCriterion = Resampling.EffectiveSampleSizeBasedResamplingCriterion(SMCsettings["resampling ratio"])
resamplingCriterion = Resampling.AlwaysResamplingCriterion()

# plain non-parallelized particle filter
pf = ParticleFilter.CentralizedTargetTrackingParticleFilter(K*PEs["number of PEs"][0],resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)

# distributed particle filter
distributedPf = ParticleFilter.TargetTrackingParticleFilterWithDRNA(
	PEs["number of PEs"][0],DRNAsettings["exchange period"],PEsNetwork,DRNAsettings["c"],DRNAsettings["epsilon"],K,DRNAsettings["normalization period"],resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors
	)

#------------------------------------------------------------- trajectory simulation ---------------------------------------------------------------------

# the target is created...
target = Target.Target(prior,transitionKernel,PRNG=PRNGs["Trajectory pseudo random numbers generator"])

# initial position and velocity of the target are kept
targetInitialPosition,targetInitialVelocity = target.pos(),target.velocity()

print('initial position:\n',targetInitialPosition)
print('initial velocity:\n',targetInitialVelocity)

targetPosition,targetVelocity = np.empty((2,nTimeInstants)),np.empty((2,nTimeInstants))
observations = [None]*nTimeInstants

# the trajectory is simulated, and the corresponding observations are obtained (notice that there is no observation for initial position)
for iTime in range(nTimeInstants):
	
	# the target moves...
	target.step()
	
	# ..and its new position and velocity are stored
	targetPosition[:,iTime:iTime+1],targetVelocity[:,iTime:iTime+1] = target.pos(),target.velocity()
	
	
	# the observations (one per sensor) are computed
	observations[iTime] = np.array(
			[float(sensor.detect(target.pos())) for sensor in sensors]
			)

#------------------------------------------------------------- metrics initialization --------------------------------------------------------------------

# we store the aggregated weights...
distributedPFaggregatedWeights = np.empty((nTimeInstants,PEs["number of PEs"][0],parameters["number of frames"]))

# ...and the position estimates
centralizedPF_pos,distributedPF_pos = np.empty((2,nTimeInstants,parameters["number of frames"])),np.empty((2,nTimeInstants,parameters["number of frames"]))

#------------------------------------------------------------------ PF estimation  -----------------------------------------------------------------------

# NOTE: a "while loop" is here more convenient than a "for loop" because having the "iFrame" variable defined at all times after the processing has started (including when finishted) 
# allows to know hown many frames have actually been processed (if any)

iFrame = 0

while iFrame < parameters["number of frames"] and not ctrlCpressed:
	
	# initialization of the particle filters
	pf.initialize()
	distributedPf.initialize()

	if painterSettings['display evolution?']:
		
		# if this is the first iteration...
		if 'painter' in locals():
			
			# ...then, the previous figure is closed
			painter.close()

		# this object will handle graphics...
		painter = Painter.WithBorder(Painter.RoomPainter(sensorsPositions,sleepTime=painterSettings["sleep time between updates"]),room["bottom left corner"],room["top right corner"])

		# ...e.g., draw the sensors
		painter.setupSensors()
		
		# the initial position is painted...
		painter.updateTargetPosition(targetInitialPosition)

		# ...along with those estimated by the PFs (they should around the middle of the room...)
		painter.updateEstimatedPosition(State.position(pf.computeMean()),identifier='centralized',color=painterSettings["color for the centralized PF"])
		painter.updateEstimatedPosition(State.position(distributedPf.computeMean()),identifier='distributed',color=painterSettings["color for the distributed PF"])

		if painterSettings['display particles evolution?']:

			# particles are plotted
			painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=painterSettings["color for the centralized PF"])
			painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=painterSettings["color for the distributed PF"])

	for iTime in range(nTimeInstants):

		print('---------- iFrame = {}, iTime = {}'.format(repr(iFrame),repr(iTime)))

		print('position:\n',targetPosition[:,iTime:iTime+1])
		print('velocity:\n',targetVelocity[:,iTime:iTime+1])
		
		# particle filters are updated
		pf.step(observations[iTime])
		distributedPf.step(observations[iTime])
		
		# the mean computed by the centralized and distributed PFs
		centralizedPF_mean,distributedPF_mean = pf.computeMean(),distributedPf.computeMean()
		
		centralizedPF_pos[:,iTime:iTime+1,iFrame],distributedPF_pos[:,iTime:iTime+1,iFrame] = State.position(centralizedPF_mean),State.position(distributedPF_mean)
		
		# the aggregated weights of the different PEs in the distributed PF are stored
		distributedPFaggregatedWeights[iTime,:,iFrame] = distributedPf.getAggregatedWeights()
		
		print('centralized PF\n',centralizedPF_mean)
		print('distributed PF\n',distributedPF_mean)
		
		if painterSettings["display evolution?"]:

			# the plot is updated with the position of the target...
			painter.updateTargetPosition(targetPosition[:,iTime:iTime+1])
			
			# ...those estimated by the PFs
			painter.updateEstimatedPosition(State.position(centralizedPF_mean),identifier='centralized',color=painterSettings["color for the centralized PF"])
			painter.updateEstimatedPosition(State.position(distributedPF_mean),identifier='distributed',color=painterSettings["color for the distributed PF"])

			if painterSettings["display particles evolution?"]:

				# ...and those of the particles...
				painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=painterSettings["color for the centralized PF"])
				painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=painterSettings["color for the distributed PF"])
	
	iFrame += 1

# plots and data are saved
saveData()

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