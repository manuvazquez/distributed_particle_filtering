#! /usr/bin/env python3

import Target
import Sensor

# number of particles per processing element (PE)
K = 10

# number of PEs
M = 20

# overall number of particles
N = K*M

firstTarget = Target.Target(Target.RadialMotor(10,1))

print(firstTarget.pos())

firstTarget.step()

print(firstTarget.pos())

sampleSensor = Sensor.Sensor(0.5,0.5,1)
#sampleSensor = Sensor.Sensor(14,14,1)

#for i in range(100):
	#print(sampleSensor.detect(firstTarget.pos()))

print(sampleSensor.detect(firstTarget.pos()))
