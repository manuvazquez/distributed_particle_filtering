import abc
import numpy as np
import scipy.cluster.vq
import numpy.random

import networkx

class SensorsPEsConnector(metaclass=abc.ABCMeta):
	
	def __init__(self,sensors,parameters=None):
		
		self._sensors = sensors
		self._parameters = parameters
		
		self._nSensors = len(sensors)
	
	@abc.abstractmethod
	def getConnections(self,nPEs):
		
		return

	def computePEsPositions(self,sensorsPositions,nPEs,nPoints):
		
		# the bottom leftmost and top right most position of the sensors are obtained...
		bottomLeftMostPosition = sensorsPositions.min(axis=1)
		topRightMostPosition = sensorsPositions.max(axis=1)
		
		# the seed of the Pseudo Random Numbers Generator to be used below (so that the positions obtained for the PEs stay the same through different runs)
		PRNG = numpy.random.RandomState(1234567)
		
		# ...is generated from a uniform distribution whose bounds are given by the rectangular space spanned by the sensors
		points = np.vstack((PRNG.uniform(bottomLeftMostPosition[0],topRightMostPosition[0],(1,nPoints)),PRNG.uniform(bottomLeftMostPosition[1],topRightMostPosition[1],(1,nPoints))))
		
		# "nPEs" centroids for the above coordinates are computed using K-Means; initial random centroids are passed to the function so it does not generate them with its own random generator
		PEsPositions,_ = scipy.cluster.vq.kmeans(points.T,points.T[PRNG.choice(points.shape[1],nPEs),:])
		
		# the transpose of the obtained positions is returned so that, just like the sensors positions, every column contains the two coordinates for a position
		return PEsPositions.T

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
	
	def getConnections(self,nPEs):
		
		sensorsPositions = np.hstack([s.position for s in self._sensors])
		
		# a number of samples proportional to the number of PEs...
		nPoints = self._parameters['number of uniform samples']*nPEs
		
		PEsPositions = self.computePEsPositions(sensorsPositions,nPEs,nPoints)
		
		# the distance from each PE (whose position has been computed above) to each sensor [<PE>,<sensor>]
		distances = np.sqrt((np.subtract(PEsPositions[:,:,np.newaxis],sensorsPositions[:,np.newaxis,:])**2).sum(axis=0))
		
		# for each sensor, the index of the PE which is closest to it
		iClosestPEtoSensors = distances.argmin(axis=0)
		
		return [list(np.where(iClosestPEtoSensors==iPE)[0]) for iPE in range(nPEs)]