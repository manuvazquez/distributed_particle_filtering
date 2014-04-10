#! /usr/bin/env python3

import math
import numpy as np
import json

import Target
import State
import Sensor
import Painter
import ParticleFilter
import DRNA
import Resampling

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

# DRNA related
drnaExchangePeriod = parameters["DRNA-exchange-period"]
drnaExchangeMap = parameters["exchange-tuples"]

# ---------------------------------------------

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

# this object will handle graphics
painter = Painter.WithBorder(Painter.Painter(sensorsPositions),roomBottomLeftCorner,roomTopRightCorner)

# we tell it to draw the sensors
painter.setupSensors()

# a object that represents the prior distribution...
prior = State.UniformBoundedPositionGaussianVelocityPrior(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)

# ...and a different one for the transition kernel
transitionKernel = State.UniformBoundedPositionGaussianVelocityTransitionKernel(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)

initialState = prior.sample()

# the target is created...
target = Target.Target(transitionKernel,State.position(initialState),State.velocity(initialState))

print('initial position: ',target.pos())

# a resampling algorithm...
resamplingAlgorithm = Resampling.MultinomialResamplingAlgorithm()

# ...and a resampling criterion are needed for the particle filters
resamplingCriterion = Resampling.EffectiveSampleSizeBasedResamplingCriterion(resamplingRatio)
#resamplingCriterion = Resampling.AlwaysResamplingCriterion()

# plain non-parallelized particle filter
pf = ParticleFilter.TrackingParticleFilter(N,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)

# distributed particle filter
distributedPf = DRNA.ParticleFiltersCompoundWithDRNA(M,drnaExchangePeriod,drnaExchangeMap,K,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel,sensors)

# initialization of the particle filters
pf.initialize()
distributedPf.initialize()

# particles are plotted
painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized')
painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color='green')

# the initial position is painted
painter.updateTargetPosition(target.pos())

for iTime in range(nTimeInstants):

	# the target moves
	target.step()
	
	# the observations (one per sensor) are computed
	observations = np.array([float(sensors[i].detect(target.pos())) for i in range(nSensors)])
	
	# particle filters are updated
	pf.step(observations)
	distributedPf.step(observations)

	# the plot is updated with the new positions of the particles...
	painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized')
	painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color='green')
	
	# ...and the target
	painter.updateTargetPosition(target.pos())
	
	print('ENTER to continue...')
	input()