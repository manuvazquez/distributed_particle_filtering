#! /usr/bin/env python3

import math
import numpy as np
import json
import time
import sys
import matplotlib.pyplot as plt

import Target
import State
import Sensor
import Painter
from smc import ParticleFilter
from smc import Resampling
import PEsNetwork

np.set_printoptions(precision=3,linewidth=100)

# the parameters file is read to memory
with open('parameters.json') as jsonData:
	parameters = json.load(jsonData)

# number of particles per processing element (PE)
K = parameters["number-of-particles-per-PE"]

# different setups for the PEs
PEs = parameters["PEs"]

# number of sensors
nSensors = parameters["number-of-sensors"]

# radius over which a sensor is able to detect the target
sensorRadius = parameters["sensor-radius"]

# number of time instants
nTimeInstants = parameters["number-of-time-instants"]

# arrays containing the coordinates that define the bounds of the room
roomBottomLeftCorner = np.array(parameters["bottom-left-corner-of-the-room"])
roomTopRightCorner = np.array(parameters["top-right-corner-of-the-room"])

# the variance of the prior density for the velocity
priorVelocityVariance = parameters["prior-velocity-variance"]

# the variance of the noise that rules the evolution of the velocity vector
stateTransitionVelocityVariance = parameters["state-transition-velocity-variance"]

# the variance of the noise that rules the evolution of the position
stateTransitionPositionVariance = parameters["state-transition-position-variance"]

# for the particle filters
resamplingRatio = parameters["resampling-ratio"]

# sleep time between updates when plotting the target and particles
sleepTime = parameters["sleep-time-between-plot-updates"]

# colors
centralizedPFcolor = parameters["color-centralized-PF"]
distributedPFcolor = parameters["color-distributed-PF"]

# DRNA related
drnaExchangePeriod = parameters["DRNA-exchange-period"]
drnaExchangePercentage = parameters["DRNA-exchange-percentage"]
drnaAggregatedWeights_c = parameters["DRNA-Aggregated-Weights-degeneration-c"]
drnaAggregatedWeights_epsilon = parameters["DRNA-Aggregated-Weights-degeneration-epsilon"]
drnaNormalizationPeriod = parameters["DRNA-normalization-period"]

# name of the output file for the MSE vs time plot
MSEvsTimeOutputFile = parameters["MSEvsTime-output-file"]

# a flag indicating whether target and particles evolution should be displayed
displayEvolution = parameters["display-evolution"]
displayParticlesEvolution = parameters["display-particles-evolution"]

# ---------------------------------------------

# number of PEs
M = PEs[0]['number']

# size of the grid when the network of PEs is a mesh
PEsnetworkMeshSize = PEs[0]['mesh size']

# it gives the width and height
roomDiagonalVector = roomTopRightCorner - roomBottomLeftCorner

#np.random.seed(9554)
np.random.seed(283627627)

#import pdb
#pdb.set_trace()

#import code
#code.interact(local=dict(globals(), **locals()))

# ----------------------------------------------------------- Processing Elements (PEs) ------------------------------------------------------------------

# a PEs network is created and used to get the exchange tuples

#drnaExchangeTuples = PEsNetwork.Customized(M,K,drnaExchangePercentage,[
	#[1,3],[0,2],[1,9],[0,4],[3,5],[4,6],[5,7],[6,8],[7,9],[2,8]
	#]).getExchangeTuples()

#drnaExchangeTuples = PEsNetwork.Ring(M,K,drnaExchangePercentage).getExchangeTuples()

drnaExchangeTuples = PEsNetwork.Mesh(M,K,drnaExchangePercentage,*PEsnetworkMeshSize).getExchangeTuples()

print('drnaExchangeTuples')
print(drnaExchangeTuples)

# ------------------------------------------------------------- sensors-related stuff --------------------------------------------------------------------

# an object for computing the positions of the sensors is created and used
sensorsPositions = Sensor.EquispacedOnRectangleSensorLayer(roomBottomLeftCorner,roomTopRightCorner).getPositions(nSensors)

# the actual number of sensor might not be equal to that requested
nSensors = sensorsPositions.shape[1]

# we build the array of sensors
sensors = [Sensor.Sensor(sensorsPositions[:,i:i+1],sensorRadius) for i in range(nSensors)]

# ----------------------------------------------------------------- dynamic model ------------------------------------------------------------------------

# a object that represents the prior distribution...
prior = State.UniformBoundedPositionGaussianVelocityPrior(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=priorVelocityVariance)

# ...and a different one for the transition kernel
transitionKernel = State.BouncingWithinRectangleTransitionKernel(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=stateTransitionVelocityVariance,noiseVariance=stateTransitionPositionVariance)

# ------------------------------------------------------------------- SMC stuff --------------------------------------------------------------------------

# a resampling algorithm...
resamplingAlgorithm = Resampling.MultinomialResamplingAlgorithm()

# ...and a resampling criterion are needed for the particle filters
#resamplingCriterion = Resampling.EffectiveSampleSizeBasedResamplingCriterion(resamplingRatio)
resamplingCriterion = Resampling.AlwaysResamplingCriterion()

# plain non-parallelized particle filter
pf = ParticleFilter.CentralizedTargetTrackingParticleFilter(K*M,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)

# distributed particle filter
distributedPf = ParticleFilter.TargetTrackingParticleFilterWithDRNA(
	M,drnaExchangePeriod,drnaExchangeTuples,drnaAggregatedWeights_c,drnaAggregatedWeights_epsilon,K,drnaNormalizationPeriod,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors
	)

#----------------------------------------------------------------- initialization ------------------------------------------------------------------------

initialState = prior.sample()

# the target is created...
target = Target.Target(transitionKernel,State.position(initialState),State.velocity(initialState))

print('initial position:\n',target.pos())
print('initial velocity:\n',target.velocity())

# initialization of the particle filters
pf.initialize()
distributedPf.initialize()

#-------------------------------------------------------------- plots initialization ---------------------------------------------------------------------

if displayEvolution:

	# this object will handle graphics
	painter = Painter.WithBorder(Painter.RoomPainter(sensorsPositions,sleepTime=sleepTime),roomBottomLeftCorner,roomTopRightCorner)

	# we tell it to draw the sensors
	painter.setupSensors()
	
	# the initial position is painted
	painter.updateTargetPosition(target.pos())
	
	# ...along with those estimated by the PFs (they should around the middle of the room...)
	painter.updateEstimatedPosition(State.position(pf.computeMean()),identifier='centralized',color=centralizedPFcolor)
	painter.updateEstimatedPosition(State.position(distributedPf.computeMean()),identifier='distributed',color=distributedPFcolor)

	if displayParticlesEvolution:

		# particles are plotted
		painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=centralizedPFcolor)
		painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=distributedPFcolor)

#------------------------------------------------------------- metrics initialization --------------------------------------------------------------------

# we store the computed mean square errors...
centralizedPF_MSE,distributedPF_MSE = np.empty(nTimeInstants),np.empty(nTimeInstants)

# ...and the aggregated weights
distributedPFaggregatedWeights = np.empty((nTimeInstants,M))

#-------------------------------------------------------------------- main loop --------------------------------------------------------------------------

for iTime in range(nTimeInstants):

	print('---------- iTime = ' + repr(iTime) + ' ---------------')

	# the target moves
	target.step()

	print('position:\n',target.pos())
	print('velocity:\n',target.velocity())
	
	# the observations (one per sensor) are computed
	observations = np.array(
			[float(sensor.detect(target.pos())) for sensor in sensors]
			)
	
	# particle filters are updated
	pf.step(observations)
	distributedPf.step(observations)
	
	# the mean computed by the centralized and distributed PFs
	centralizedPF_mean,distributedPF_mean = pf.computeMean(),distributedPf.computeMean()
	
	# MSE for both the centralized and distributed particle filters is computed
	centralizedPF_MSE[iTime],distributedPF_MSE[iTime] = ((State.position(centralizedPF_mean)-target.pos())**2).mean(),((State.position(distributedPF_mean)-target.pos())**2).mean()
	
	# the aggregated weights of the different PEs in the distributed PF are stored
	distributedPFaggregatedWeights[iTime,:] = distributedPf.getAggregatedWeights()
	
	print('centralized PF\n',centralizedPF_mean)
	print('distributed PF\n',distributedPF_mean)
	
	if displayEvolution:

		# the plot is updated with the position of the target...
		painter.updateTargetPosition(target.pos())
		
		# ...those estimated by the PFs
		painter.updateEstimatedPosition(State.position(centralizedPF_mean),identifier='centralized',color=centralizedPFcolor)
		painter.updateEstimatedPosition(State.position(distributedPF_mean),identifier='distributed',color=distributedPFcolor)

		if displayParticlesEvolution:

			# ...and those of the particles...
			painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=centralizedPFcolor)
			painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=distributedPFcolor)

# MSE vs time
Painter.plotMSEvsTime(centralizedPF_MSE,distributedPF_MSE,centralizedPFcolor,distributedPFcolor,'+','o',MSEvsTimeOutputFile)

## aggregated weights vs time in a stackbar diagram
#Painter.plotAggregatedWeightsDistributionVsTime(distributedPFaggregatedWeights)

# evolution of the largest aggregated weight over time
Painter.plotAggregatedWeightsSupremumVsTime(distributedPFaggregatedWeights,distributedPf.getAggregatedWeightsUpperBound())

if displayEvolution:
	painter.save()

import code
code.interact(local=dict(globals(), **locals()))