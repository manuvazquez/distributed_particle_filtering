#! /usr/bin/env python3

import math
import numpy as np
import json
import time
import matplotlib.pyplot as plt

import Target
import State
import Sensor
import Painter
from smc import ParticleFilter
from smc import Resampling
import PEsNetwork

# the parameters file is read to memory
with open('parameters.json') as jsonData:
	parameters = json.load(jsonData)

# number of particles per processing element (PE)
K = parameters["number-of-particles-per-PE"]

# number of PEs
M = parameters["number-of-PEs"]

# number of sensors
nSensors = parameters["number-of-sensors"]

# radious over which a sensor is able to detect the target
sensorRadius = parameters["sensor-radius"]

# number of time instants
nTimeInstants = parameters["number-of-time-instants"]

# arrays containing the coordinates that define the bounds of the room
roomBottomLeftCorner = np.array(parameters["bottom-left-corner-of-the-room"])
roomTopRightCorner = np.array(parameters["top-right-corner-of-the-room"])

# the variance of the noise that rules the evolution of the velocity vector
velocityVariance = parameters["velocity-variance"]

# for the particle filters
resamplingRatio = parameters["resampling-ratio"]

# sleep time between updates when plotting the target and particles
sleepTime = parameters["sleep-time-between-plot-updates"]

# colors
centralizedPFcolor = parameters["color-centralized-PF"]
distributedPFcolor = parameters["color-distributed-PF"]

# DRNA related
drnaExchangePeriod = parameters["DRNA-exchange-period"]
drnaExchangeTuples = parameters["DRNA-exchange-tuples"]
drnaAggregatedWeights_c = parameters["DRNA-Aggregated-Weights-degeneration-c"]
drnaAggregatedWeights_epsilon = parameters["DRNA-Aggregated-Weights-degeneration-epsilon"]

# name of the output file for the MSE vs time plot
MSEvsTimeOutputFile = parameters["MSEvsTime-output-file"]

# a flag indicating whether target and particles evolution should be displayed
displayEvolution = parameters["display-evolution"]

#import code
#code.interact(local=dict(globals(), **locals()))

# ---------------------------------------------

# a PEs network is created and used to get the exchange tuples
drnaExchangeTuples = PEsNetwork.PEsNetwork(M,K,0.05,[
	[1,3],[0,2],[1,9],[0,4],[3,5],[4,6],[5,7],[6,8],[7,9],[2,8]
	]).getExchangeTuples()

# it gives the width and height
roomDiagonalVector = roomTopRightCorner - roomBottomLeftCorner

# overall number of particles
N = K*M

sensorLayer = Sensor.EquispacedOnRectangleSensorLayer(roomBottomLeftCorner,roomTopRightCorner)
sensorsPositions = sensorLayer.getPositions(nSensors)

print('Sensors positions:\n',sensorsPositions)

# the actual number of sensor might not be equal to that requested
nSensors = sensorsPositions.shape[1]

# we build the array of sensors
sensors = [Sensor.Sensor(sensorsPositions[:,i:i+1],sensorRadius) for i in range(nSensors)]

# a object that represents the prior distribution...
prior = State.UniformBoundedPositionGaussianVelocityPrior(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)

# ...and a different one for the transition kernel
transitionKernel = State.UniformBoundedPositionGaussianVelocityTransitionKernel(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)

initialState = prior.sample()

# the target is created...
target = Target.Target(transitionKernel,State.position(initialState),State.velocity(initialState))

print('initial position:\n',target.pos())

# a resampling algorithm...
resamplingAlgorithm = Resampling.MultinomialResamplingAlgorithm()

# ...and a resampling criterion are needed for the particle filters
#resamplingCriterion = Resampling.EffectiveSampleSizeBasedResamplingCriterion(resamplingRatio)
resamplingCriterion = Resampling.AlwaysResamplingCriterion()

# plain non-parallelized particle filter
pf = ParticleFilter.CentralizedTargetTrackingParticleFilter(N,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)

# distributed particle filter
distributedPf = ParticleFilter.TargetTrackingParticleFilterWithDRNA(M,drnaExchangePeriod,drnaExchangeTuples,drnaAggregatedWeights_c,drnaAggregatedWeights_epsilon,K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)

# initialization of the particle filters
pf.initialize()
distributedPf.initialize()

if displayEvolution:

	# this object will handle graphics
	painter = Painter.WithBorder(Painter.RoomPainter(sensorsPositions,sleepTime=sleepTime),roomBottomLeftCorner,roomTopRightCorner)

	# we tell it to draw the sensors
	painter.setupSensors()

	# particles are plotted
	painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=centralizedPFcolor)
	painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=distributedPFcolor)

	# the initial position is painted
	painter.updateTargetPosition(target.pos())

# we store the computed mean square errors...
centralizedPF_MSE,distributedPF_MSE = np.empty(nTimeInstants),np.empty(nTimeInstants)

# ...and the aggregated weights
distributedPFaggregatedWeights = np.empty((nTimeInstants+1,M))

# the aggregated weights of the different PEs in the distributed PF at the initial time instant
distributedPFaggregatedWeights[0,:] = distributedPf.getAggregatedWeights()

for iTime in range(nTimeInstants):

	# the target moves
	target.step()
	
	# the observations (one per sensor) are computed
	observations = np.array([float(sensors[i].detect(target.pos())) for i in range(nSensors)])
	
	# particle filters are updated
	pf.step(observations)
	distributedPf.step(observations)

	if displayEvolution:

		# the plot is updated with the new positions of the particles...
		painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=centralizedPFcolor)
		painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=distributedPFcolor)
		
		# ...and the target
		painter.updateTargetPosition(target.pos())
	
	# MSE for both the centralized and distributed particle filters is computed
	centralizedPF_MSE[iTime],distributedPF_MSE[iTime] = ((State.position(pf.computeMean())-target.pos())**2).mean(),((State.position(distributedPf.computeMean())-target.pos())**2).mean()
	
	# the aggregated weights of the different PEs in the distributed PF are stored
	distributedPFaggregatedWeights[iTime+1,:] = distributedPf.getAggregatedWeights()

# MSE vs time
Painter.plotMSEvsTime(centralizedPF_MSE,distributedPF_MSE,centralizedPFcolor,distributedPFcolor,'+','o',MSEvsTimeOutputFile)

# aggregated weights vs time in a stackbar diagram
Painter.plotAggregatedWeightsDistributionVsTime(distributedPFaggregatedWeights)

import code
code.interact(local=dict(globals(), **locals()))