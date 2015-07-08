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
	
	def __init__(self,sensorsPositions,PEsPositions,parameters=None):
		
		super().__init__(sensorsPositions,PEsPositions,parameters)
		
	def getConnections(self,nPEs):
		
		# the distance from each PE (whose position has been computed above) to each sensor [<PE>,<sensor>]
		distances = np.sqrt((np.subtract(self._PEsPositions[:,:,np.newaxis],self._sensorsPositions[:,np.newaxis,:])**2).sum(axis=0))
		
		# for each sensor, the index of the PE which is closest to it
		iClosestPEtoSensors = distances.argmin(axis=0)
		
		return [list(np.where(iClosestPEtoSensors==iPE)[0]) for iPE in range(nPEs)]

class ConstrainedProximityBasedConnector(ProximityBasedConnector):
	
	def getConnections(self,nPEs):
		
		# a list with the sensors associated with each PE withouth the fixed number of sensors constraint
		unconstrained = super().getConnections(nPEs)
		
		lengths = [len(s) for s in unconstrained]
		
		# the indexes of the PEs ordered by descending number of associated sensors
		iDescendingLength = np.argsort(lengths)[::-1]
		
		#import code
		#code.interact(local=dict(globals(), **locals()))
		
		nSensorsPerPE = self._sensorsPositions.shape[1]//self._PEsPositions.shape[1]
		
		for i in range(nPEs-1):
			
			# the index of the PE to be processed
			iCurrentPE = iDescendingLength[i]
			
			# number of sensor that should dettach from this PE
			nSensorsToDrop =  lengths[iCurrentPE] - nSensorsPerPE
			
			if nSensorsToDrop==0:
				
				print('pass')
				
				continue
			
			#if nSensorsToDrop < 0:
				
				#import code
				#code.interact(local=dict(globals(), **locals()))
			
			sensorsPositions = self._sensorsPositions[:,unconstrained[iCurrentPE]]
			
			remainingPEsPositions = self._PEsPositions[:,iDescendingLength[i+1:]]
			
			# the (i,j) element is the distance from the i-th sensor to the j-th remaining PE
			distances = np.sqrt(((sensorsPositions[:,:,np.newaxis] - remainingPEsPositions[:,np.newaxis,:])**2).sum(axis=0))
			
			for _ in range(nSensorsToDrop):
				
				iSensor,iPE = np.unravel_index(distances.argmin(),distances.shape)
		
				unconstrained[iDescendingLength[i+1+iPE]].append(unconstrained[iCurrentPE][iSensor])
				del unconstrained[iCurrentPE][iSensor]
				
				lengths[iCurrentPE] -= 1
				lengths[iDescendingLength[i+1+iPE]] += 1

		import code
		code.interact(local=dict(globals(), **locals()))
		
		return unconstrained