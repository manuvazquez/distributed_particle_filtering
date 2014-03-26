#! /usr/bin/env python3

import Target
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

firstTarget = Target.Target(Target.RadialMotor(10,1))

print(firstTarget.pos())

firstTarget.step()

print(firstTarget.pos())

sampleSensor = Sensor.Sensor(0.5,0.5,1)
#sampleSensor = Sensor.Sensor(14,14,1)

#for i in range(100):
	#print(sampleSensor.detect(firstTarget.pos()))

print(sampleSensor.detect(firstTarget.pos()))

print('ENTER to close the figures and quit the program...')
input()
