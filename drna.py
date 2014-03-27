#! /usr/bin/env python3

import Target
import State
import Sensor
import math
import numpy as np
import Painter

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
prior = State.BoundedUniformPrior(roomBottomLeftCorner,roomTopRightCorner)

# the target is created
#target = Target.Target(State.StraightTransitionKernel(10))
#target = Target.Target(State.BoundedRandomSpeedTransitionKernel(roomBottomLeftCorner,roomTopRightCorner))
target = Target.Target(State.BoundedRandomSpeedTransitionKernel(roomBottomLeftCorner,roomTopRightCorner),initialPosition=prior.sample(),initialSpeed=np.array([0,3]))

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
