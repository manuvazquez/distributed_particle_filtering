import abc
import numpy as np
import scipy.cluster.vq
import numpy.random

import networkx

def computePEsPositions(bottomLeftCorner,topRightCorner,nPEs,nPoints):
	
	# the seed of the Pseudo Random Numbers Generator to be used below (so that the positions obtained for the PEs stay the same through different runs)
	PRNG = numpy.random.RandomState(1234567)
	
	# ...is generated from a uniform distribution within the limits of the room
	points = np.vstack((PRNG.uniform(bottomLeftCorner[0],topRightCorner[0],(1,nPoints)),PRNG.uniform(bottomLeftCorner[1],topRightCorner[1],(1,nPoints))))
	
	# "nPEs" centroids for the above coordinates are computed using K-Means; initial random centroids are passed to the function so it does not generate them with its own random generator
	PEsPositions,_ = scipy.cluster.vq.kmeans(points.T,points.T[PRNG.choice(points.shape[1],nPEs),:])

	return orderedPEsPositions(PEsPositions,bottomLeftCorner,topRightCorner)

def computePEsPositionsFromSensorsPositions(bottomLeftCorner,topRightCorner,sensorsPositions,nPEs):
	
	# seed for the default numpy random generator (used by scipy)
	np.random.seed(123)
	
	# k-means over the sensors positions using "nPEs" centroids
	PEsPositions,_ = scipy.cluster.vq.kmeans(sensorsPositions.T,nPEs)
	
	# just in case K-means doesn't return the proper number of sensors
	assert(len(PEsPositions)==nPEs)
	
	return orderedPEsPositions(PEsPositions,bottomLeftCorner,topRightCorner)

def orderedPEsPositions(PEsPositions,bottomLeftCorner,topRightCorner):
	
	# the initial position is the upper-left corner
	previousPos = np.array([bottomLeftCorner[0],topRightCorner[1]])
	
	# we need to modify the "PEsPositions" array during the algorithm
	PEsPositionsCopy = PEsPositions.copy()
	
	# a numpy array to store the result (just like the sensors positions, every column contains the two coordinates for a position)
	ordered = np.empty_like(PEsPositions.T)
	
	for i in range(len(PEsPositions)):
		
		# the distance from the previuos position to ALL the PEs
		distances = np.linalg.norm(previousPos - PEsPositionsCopy,axis=1)
		
		# the index of the minimum distance...
		iMin = np.argmin(distances)
		
		# ...is used to pick the next PE in the "ordered" positions
		ordered[:,i] = PEsPositions[iMin,:]
		
		# we make sure this PE is not going to be selected again
		PEsPositionsCopy[iMin,0] = np.Inf
		
		# the previous position is the selected PE
		previousPos = ordered[:,i]
		
	return ordered

class SensorsPEsConnector(metaclass=abc.ABCMeta):
	
	def __init__(self,sensorsPositions,PEsPositions=None,parameters=None):
		
		self._sensorsPositions = sensorsPositions
		self._PEsPositions = PEsPositions
		self._parameters = parameters
		
		self._nSensors = self._sensorsPositions.shape[1]
	
	@abc.abstractmethod
	def getConnections(self,nPEs):
		
		return

class EverySensorWithEveryPEConnector(SensorsPEsConnector):
	
	def getConnections(self,nPEs):
		
		return [list(range(self._nSensors))]*nPEs

class SensorOrientedConnector(SensorsPEsConnector):
	
	def getConnections(self,nPEs):

		# each sensor is associated with "nPEsPerSensor" PEs
		sensorsDegrees = [self._parameters['number of PEs per sensor']]*self._nSensors
		
		# how many (at least) sensors should be connected to every PE (notice that the expresion between parenthesis is the overall number of connections)
		nSensorsPerPE = (self._parameters['number of PEs per sensor']*self._nSensors) // nPEs
		
		# each PE should be connected to the the number of sensors specified in the corresponding position of this list
		PEsDegrees = [nSensorsPerPE]*nPEs
		
		# if some connections remain, in order to satisfy that each sensor is connected to the given number of PEs...
		for iPE in range(self._parameters['number of PEs per sensor']*self._nSensors % nPEs):
			
			# ...the "first" PEs get the extra needed connections
			PEsDegrees[iPE] +=  1
	
		# a bipartite graph with one set of nodes given by the sensors and other by the PEs
		graph = networkx.bipartite_havel_hakimi_graph(sensorsDegrees,PEsDegrees)
		
		# we only "look" at the nodes from "self._nSensors" onwards, since the previous ones correspond to the sensors
		return [sorted(graph.neighbors(iPE+self._nSensors)) for iPE in range(nPEs)]

class ProximityBasedConnector(SensorsPEsConnector):
	
	def __init__(self,sensors,PEsPositions,parameters=None):
		
		super().__init__(sensors,PEsPositions,parameters)
		
	def getConnections(self,nPEs):
		
		# a number of samples proportional to the number of PEs...
		nPoints = self._parameters['number of uniform samples']*nPEs
		
		# the distance from each PE (whose position has been computed above) to each sensor [<PE>,<sensor>]
		distances = np.sqrt((np.subtract(self._PEsPositions[:,:,np.newaxis],self._sensorsPositions[:,np.newaxis,:])**2).sum(axis=0))
		
		# for each sensor, the index of the PE which is closest to it
		iClosestPEtoSensors = distances.argmin(axis=0)
		
		return [list(np.where(iClosestPEtoSensors==iPE)[0]) for iPE in range(nPEs)]