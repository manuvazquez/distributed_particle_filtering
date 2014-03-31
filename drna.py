#! /usr/bin/env python3

import math
import numpy as np

import Target
import State
import Sensor
import Painter
import ParticleFilter

# number of particles per processing element (PE)
K = 10

# number of PEs
M = 20

# number of sensors
nSensors = 16

# number of time instants
nTimeInstants = 15

# tuples containing the coordinates that define the bounds of the room
roomBottomLeftCorner = np.array([-10,-20])
roomTopRightCorner = np.array([10,20])

# the variance of the noise that rules the evolution of the velocity vector
velocityVariance = 0.25

# ---------------------------------------------

# it gives the width and height
roomDiagonalVector = roomTopRightCorner - roomBottomLeftCorner

# overall number of particles
N = K*M

sensorLayer = Sensor.EquispacedOnRectangleSensorLayer(roomBottomLeftCorner,roomTopRightCorner)
sensorsPositions = sensorLayer.getPositions(nSensors)

painter = Painter.WithBorder(Painter.Painter(sensorsPositions),roomBottomLeftCorner,roomTopRightCorner)
#painter = Painter.Painter(sensorsPositions)
painter.go()

# a object that represents the prior distribution
prior = State.UniformBoundedPositionGaussianVelocityPrior(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)
transitionKernel = State.UniformBoundedPositionGaussianVelocityTransitionKernel(roomBottomLeftCorner,roomTopRightCorner,velocityVariance=velocityVariance)

initialState = prior.sample()

# the target is created...

# notice that samples returns 2D array and Target needs to receive a 1D array, hence the [:,0]
target = Target.Target(transitionKernel,State.position(initialState),State.velocity(initialState))

# particle filter is created
ParticleFilter.TrackingParticleFilter(20,prior,transitionKernel)

for iTime in range(nTimeInstants):
	
	print(target.pos())

	painter.update(target.pos())

	target.step()

	print(target.pos())

	painter.update(target.pos())
	
	print('ENTER to continue...')
	input()

sampleSensor = Sensor.Sensor(0.5,0.5,1)
#sampleSensor = Sensor.Sensor(14,14,1)

#for i in range(100):
	#print(sampleSensor.detect(target.pos()))

print(sampleSensor.detect(target.pos()))
