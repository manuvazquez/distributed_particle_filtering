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

# ---------------------------------------------

# we rather have the coordinates of the corners stored as numpy arrays...
room["bottom left corner"] = np.array(room["bottom left corner"])
room["top right corner"] = np.array(room["top right corner"])

# it gives the width and height
roomDiagonalVector = room["top right corner"] - room["bottom left corner"]

if not parameters["ramdon seed?"]:
	#np.random.seed(9554)
	#np.random.seed(283627627)
	np.random.seed(parameters["seed"])

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
centralizedPF_MSE,distributedPF_MSE = np.empty(nTimeInstants),np.empty(nTimeInstants)

# ...and the aggregated weights
distributedPFaggregatedWeights = np.empty((nTimeInstants,PEs["number of PEs"][0]))

#------------------------------------------------------------------ PF estimation  -----------------------------------------------------------------------

# initialization of the particle filters
pf.initialize()
distributedPf.initialize()

if painterSettings["display evolution?"]:

	# this object will handle graphics
	painter = Painter.WithBorder(Painter.RoomPainter(sensorsPositions,sleepTime=painterSettings["sleep time between updates"]),room["bottom left corner"],room["top right corner"])

	# we tell it to draw the sensors
	painter.setupSensors()
	
	# the initial position is painted...
	painter.updateTargetPosition(target.pos())

	# ...along with those estimated by the PFs (they should around the middle of the room...)
	painter.updateEstimatedPosition(State.position(pf.computeMean()),identifier='centralized',color=painterSettings["color for the centralized PF"])
	painter.updateEstimatedPosition(State.position(distributedPf.computeMean()),identifier='distributed',color=painterSettings["color for the distributed PF"])

	if painterSettings["display particles evolution?"]:

		# particles are plotted
		painter.updateParticlesPositions(State.position(pf.getState()),identifier='centralized',color=painterSettings["color for the centralized PF"])
		painter.updateParticlesPositions(State.position(distributedPf.getState()),identifier='distributed',color=painterSettings["color for the distributed PF"])

for iTime in range(nTimeInstants):

	print('---------- iTime = ' + repr(iTime) + ' ---------------')

	print('position:\n',targetPosition[:,iTime:iTime+1])
	print('velocity:\n',targetVelocity[:,iTime:iTime+1])
	
	# particle filters are updated
	pf.step(observations[iTime])
	distributedPf.step(observations[iTime])
	
	# the mean computed by the centralized and distributed PFs
	centralizedPF_mean,distributedPF_mean = pf.computeMean(),distributedPf.computeMean()
	
	# MSE for both the centralized and distributed particle filters is computed
	centralizedPF_MSE[iTime],distributedPF_MSE[iTime] = ((State.position(centralizedPF_mean)-targetPosition[:,iTime:iTime+1])**2).mean(),((State.position(distributedPF_mean)-targetPosition[:,iTime:iTime+1])**2).mean()
	
	# the aggregated weights of the different PEs in the distributed PF are stored
	distributedPFaggregatedWeights[iTime,:] = distributedPf.getAggregatedWeights()
	
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

# MSE vs time
Painter.plotMSEvsTime(centralizedPF_MSE,distributedPF_MSE,painterSettings["color for the centralized PF"],painterSettings["color for the distributed PF"],'+','o',painterSettings["output file name for the MSEvsTime plot"])

## aggregated weights vs time in a stackbar diagram
#Painter.plotAggregatedWeightsDistributionVsTime(distributedPFaggregatedWeights)

# evolution of the largest aggregated weight over time
Painter.plotAggregatedWeightsSupremumVsTime(distributedPFaggregatedWeights,distributedPf.getAggregatedWeightsUpperBound())

if painterSettings["display evolution?"]:
	painter.save()

import code
code.interact(local=dict(globals(), **locals()))