#! /usr/bin/env python3

import math
import numpy as np

import Target
import State
import Sensor
import Painter
import ParticleFilter
import Resampling

# number of particles per processing element (PE)
K = 10

# number of PEs
M = 20

# number of sensors
nSensors = 16

# radious over which a sensor is able to detect the target
sensorRadius = 5

# number of time instants
nTimeInstants = 15

# tuples containing the coordinates that define the bounds of the room
roomBottomLeftCorner = np.array([-10,-20])
roomTopRightCorner = np.array([10,20])

# the variance of the noise that rules the evolution of the velocity vector
velocityVariance = 0.25

# for the particle filters
resamplingRatio = 0.9

# ---------------------------------------------

# it gives the width and height
roomDiagonalVector = roomTopRightCorner - roomBottomLeftCorner

# overall number of particles
N = K*M

sensorLayer = Sensor.EquispacedOnRectangleSensorLayer(roomBottomLeftCorner,roomTopRightCorner)
sensorsPositions = sensorLayer.getPositions(nSensors)

print('Sensors positions:\n',sensorsPositions)

#import code
#code.interact(local=dict(globals(), **locals()))

# the actual number of sensor might not be equal to that requested
nSensors = sensorsPositions.shape[1]

# we build the array of sensors
sensors = [Sensor.Sensor(sensorsPositions[:,i:i+1],sensorRadius) for i in range(nSensors)]

# this object will handle graphics
painter = Painter.WithBorder(Painter.Painter(sensorsPositions),roomBottomLeftCorner,roomTopRightCorner)

# we tell them to draw the sensors
painter.setupSensors()

# a object that represents the prior distribution
prior = State.UniformBoundedPositionGaussianVelocityPrior(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)
transitionKernel = State.UniformBoundedPositionGaussianVelocityTransitionKernel(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)

initialState = prior.sample()

# the target is created...
target = Target.Target(transitionKernel,State.position(initialState),State.velocity(initialState))

print('initial position: ',target.pos())

# a resampling algorithm...
resamplingAlgorithm = Resampling.MultinomialResamplingAlgorithm()

# ...and a resampling criterion...
resamplingCriterion = Resampling.ResampleCriterion(resamplingRatio)

# ...are needed for the particle filter
pf = ParticleFilter.TrackingParticleFilter(20,resamplingAlgorithm,resamplingCriterion,prior,transitionKernel)

# initialization
pf.initialize()

# particles are plotted
painter.updateParticlesPositions(State.position(pf.getState()))

# the initial position is painted
painter.updateTargetPosition(target.pos())

for iTime in range(nTimeInstants):

	# the target moves
	target.step()
	
	print('position: ',target.pos())
	
	# we compute the observations
	observations = np.array([float(sensors[i].detect(target.pos())) for i in range(nSensors)])
	print('observations: ',observations)

	# the PF is updated
	pf.step(None)

	# the plot is updated
	painter.updateParticlesPositions(State.position(pf.getState()))

	print(target.pos())

	painter.updateTargetPosition(target.pos())
	
	print('ENTER to continue...')
	input()