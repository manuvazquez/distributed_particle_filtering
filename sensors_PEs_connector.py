import abc
import numpy as np

import networkx

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