#! /usr/bin/env python3

import math
import numpy as np
import json
import time
import sys
import os
import signal
import socket

# in order to clock the execution time...
startTime = time.time()

# the parameters file is read to memory
with open('parameters.json') as jsonData:
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

# some of this modules also import matplotlib.pyplot, and because this should be done AFTER calling "matplotlib.use", they have been imported here and not at the very beginning
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
outputFile = hostname+'_'+date+'_nFrames={}.eps'

# how numpy arrays are printed on screen is specified here
np.set_printoptions(precision=3,linewidth=100)

# ---------------------------------------------

# we'd rather have the coordinates of the corners stored as numpy arrays...
room["bottom left corner"] = np.array(room["bottom left corner"])
room["top right corner"] = np.array(room["top right corner"])

# it amounts to the width and height
roomDiagonalVector = room["top right corner"] - room["bottom left corner"]

if not parameters["ramdon seed?"]:
	np.random.seed(parameters["seed"])

# ---------------------------------------------------------------- signals handling ----------------------------------------------------------------------

# within the handler, once Ctrl-C is pressed once, the default behaviour is restored
original_sigint_handler = signal.getsignal(signal.SIGINT)

# the interrupt signal (ctrl-c) is handled by this function
def sigint_handler(signum, frame):
	
	global ctrlCpressed
	
	if not ctrlCpressed:
		
		print('Ctrl-C pressed...one more to exit the program right now...')
		ctrlCpressed = True
		
		# the default behaviour is restored
		signal.signal(signal.SIGINT, original_sigint_handler)

def sigusr1_handler(signum, frame):
	
	if 'iFrame' not in globals():
		print('nothing done yet...')
		return
	
	global iFrame
	
	# MSE vs time
	Painter.plotMSEvsTime(centralizedPF_MSE[:,:iFrame].mean(axis=1),distributedPF_MSE[:,:iFrame].mean(axis=1),
						painterSettings["color for the centralized PF"],painterSettings["color for the distributed PF"],'+','o',painterSettings["file name prefix for the MSE vs time plot"] + '_' + outputFile.format(repr(iFrame)))

	# the aggregated weights are  normalized at ALL TIMES and for EVERY frame
	normalizedAggregatedWeights = np.rollaxis(np.divide(np.rollaxis(distributedPFaggregatedWeights[:,:,:iFrame],2,1),distributedPFaggregatedWeights[:,:,:iFrame].sum(axis=1)[:,:,np.newaxis]),2,1)

	# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
	maxWeights = (normalizedAggregatedWeights.max(axis=1)**4).mean(axis=1)**(1/4)

	# evolution of the largest aggregated weight over time
	Painter.plotAggregatedWeightsSupremumVsTime(maxWeights,distributedPf.getAggregatedWeightsUpperBound(),painterSettings["file name prefix for the aggregated weights supremum vs time plot"] + '_' + outputFile.format(repr(iFrame)))

# Ctrl-C has not been pressed yet...well, if it has, then the program has not even reached here
ctrlCpressed = False

# handler for SIGINT is installed
signal.signal(signal.SIGINT, sigint_handler)

# handler for SIGUSR1 is installed
signal.signal(signal.SIGUSR1, sigusr1_handler)

# ----------------------------------------------------------- Processing Elements (PEs) ------------------------------------------------------------------

# a PEs network is created...used later to get the exchange tuples

if PEs["topology"]=="Customized":
	
	PEsNetwork = PEsNetwork.Customized(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],[[1,3],[0,2],[1,9],[0,4],[3,5],[4,6],[5,7],[6,8],[7,9],[2,8]])
	
elif PEs["topology"]=="Ring":
	
	PEsNetwork = PEsNetwork.Ring(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"])
	
elif PEs["topology"]=="Mesh":
	
	PEsNetwork = PEsNetwork.Mesh(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],PEs["Mesh"]["neighbours"],*PEs["Mesh"]["geometry"][0])
	
elif PEs["topology"]=="FullyConnected":
	
	PEsNetwork = PEsNetwork.FullyConnected(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"])
	
elif PEs["topology"]=="FullyConnectedWithRandomLinksRemoved":

	PEsNetwork = PEsNetwork.FullyConnectedWithRandomLinksRemoved(PEs["number of PEs"][0],K,DRNAsettings["exchanged particles maximum percentage"],PEs["FullyConnectedWithRandomLinksRemoved"]["number of links to be removed"])
	
else:
	print('PEs network topology not supported...')
	raise SystemExit(0)

# ------------------------------------------------------------- sensors-related stuff --------------------------------------------------------------------

# an object for computing the positions of the sensors is created and used
sensorsPositions = Sensor.EquispacedOnRectangleSensorLayer(room["bottom left corner"],room["top right corner"]).getPositions(sensorsSettings["number"])

# the actual number of sensor might not be equal to that requested
sensorsSettings["number"] = sensorsPositions.shape[1]

# we build the array of sensors
sensors = [Sensor.Sensor(sensorsPositions[:,i:i+1],sensorsSettings["radius"]) for i in range(sensorsSettings["number"])]

# ----------------------------------------------------------------- dynamic model ------------------------------------------------------------------------

# a object that represents the prior distribution...
prior = State.UniformBoundedPositionGaussianVelocityPrior(room["bottom left corner"],room["top right corner"],velocityVariance=priorDistribution["velocity variance"])

# ...and a different one for the transition kernel
transitionKernel = State.BouncingWithinRectangleTransitionKernel(room["bottom left corner"],room["top right corner"],velocityVariance=stateTransition["velocity variance"],noiseVariance=stateTransition["position variance"])

# ------------------------------------------------------------------- SMC stuff --------------------------------------------------------------------------

# a resampling algorithm...
resamplingAlgorithm = Resampling.MultinomialResamplingAlgorithm()

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

initialState = prior.sample()

# the target is created...
target = Target.Target(transitionKernel,State.position(initialState),State.velocity(initialState))

print('initial position:\n',target.pos())
print('initial velocity:\n',target.velocity())

targetPosition,targetVelocity = np.empty((2,nTimeInstants)),np.empty((2,nTimeInstants))
observations = [None]*nTimeInstants

# the trajectory is simulated, and the corresponding observations are obtained
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

# we store the computed mean square errors...
centralizedPF_MSE,distributedPF_MSE = np.empty((nTimeInstants,parameters["number of frames"])),np.empty((nTimeInstants,parameters["number of frames"]))

# ...and the aggregated weights
distributedPFaggregatedWeights = np.empty((nTimeInstants,PEs["number of PEs"][0],parameters["number of frames"]))


#------------------------------------------------------------------ PF estimation  -----------------------------------------------------------------------

iFrame = 0

while iFrame < parameters['number of frames'] and not ctrlCpressed:

	# initialization of the particle filters
	pf.initialize()
	distributedPf.initialize()

	if painterSettings['display evolution?']:
		
		# if this is the first iteration...
		if 'painter' in locals():
			
			# ...then, the previous figure is closed
			painter.close()

		# this object will handle graphics
		painter = Painter.WithBorder(Painter.RoomPainter(sensorsPositions,sleepTime=painterSettings["sleep time between updates"]),room["bottom left corner"],room["top right corner"])

		# we tell it to draw the sensors
		painter.setupSensors()
		
		# the initial position is painted...
		painter.updateTargetPosition(target.pos())

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
		
		# MSE for both the centralized and distributed particle filters is computed
		centralizedPF_MSE[iTime,iFrame],distributedPF_MSE[iTime,iFrame] = ((State.position(centralizedPF_mean)-targetPosition[:,iTime:iTime+1])**2).mean(),((State.position(distributedPF_mean)-targetPosition[:,iTime:iTime+1])**2).mean()
		
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

# if may be the case that less than parameters["number of frames"] were simulated if Ctrl-C was pressed...
centralizedPF_MSE,distributedPF_MSE = centralizedPF_MSE[:,:iFrame],distributedPF_MSE[:,:iFrame]
distributedPFaggregatedWeights = distributedPFaggregatedWeights[:,:,:iFrame]

# MSE vs time
Painter.plotMSEvsTime(centralizedPF_MSE.mean(axis=1),distributedPF_MSE.mean(axis=1),
					painterSettings["color for the centralized PF"],painterSettings["color for the distributed PF"],'+','o',painterSettings["file name prefix for the MSE vs time plot"] + '_' + outputFile.format(repr(iFrame)))

# the aggregated weights are  normalized at ALL TIMES and for EVERY frame
normalizedAggregatedWeights = np.rollaxis(np.divide(np.rollaxis(distributedPFaggregatedWeights,2,1),distributedPFaggregatedWeights.sum(axis=1)[:,:,np.newaxis]),2,1)

# ...and the maximum weight, also at ALL TIMES and for EVERY frame, is obtained
maxWeights = (normalizedAggregatedWeights.max(axis=1)**4).mean(axis=1)**(1/4)

# evolution of the largest aggregated weight over time
Painter.plotAggregatedWeightsSupremumVsTime(maxWeights,distributedPf.getAggregatedWeightsUpperBound(),
											painterSettings["file name prefix for the aggregated weights supremum vs time plot"] + '_' + outputFile.format(repr(iFrame)),DRNAsettings["exchange period"])

if painterSettings["display evolution?"]:
	painter.save()

# data is saved
np.savez('res_' + outputFile.format(repr(iFrame)),normalizedAggregatedWeights=normalizedAggregatedWeights,aggregatedWeightsUpperBound=distributedPf.getAggregatedWeightsUpperBound())

# ------------------------------------------------------------------ benchmarking  -----------------------------------------------------------------------

# for benchmarking purposes, it is assumed that the execution ends here
endTime = time.time()

# the elapsed time in seconds
elapsedTime = endTime-startTime

if elapsedTime>60:
	print('Execution time: {} minutes'.format(repr(elapsedTime/60)))
else:
	print('Execution time: {} seconds'.format(repr(elapsedTime)))

# --------------------------------------------------------------------------------------------------------------------------------------------------------

# if using the agg backend (no pictures shown), there is no point in bringing up the interactive prompt before exitingls
if not useAgg:
	import code
	code.interact(local=dict(globals(), **locals()))