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

class SensorsPositionsBasedConnector(SensorsPEsConnector):
	
	def getConnections(self,nPEs):
		
		sensorsPositions = np.hstack([s.position for s in self._sensors])
		
		# the bottom leftmost and top right most position of the sensors are obtained...
		bottomLeftMostPosition = sensorsPositions.min(axis=1)
		topRightMostPosition = sensorsPositions.max(axis=1)
		
		# a number of samples proportional to the number of PEs...
		nPoints = self._parameters['number of uniform samples']*nPEs
		
		# ...is generated from a uniform distribution whose bounds are given by the rectangular space spanned by the sensors
		points = np.vstack((numpy.random.uniform(bottomLeftMostPosition[0],topRightMostPosition[0],(1,nPoints)),numpy.random.uniform(bottomLeftMostPosition[1],topRightMostPosition[1],(1,nPoints))))
		
		# "nPEs" centroids for the above coordinates are computed using K-Means
		PEsPositions,_ = scipy.cluster.vq.kmeans(points.T,nPEs)
		
		# more convenient so that (just like the sensors positions), every column contains the two coordinates for a position
		PEsPositions = PEsPositions.T
		
		# the distance from each PE (whose position has been computed above) to each sensor [<PE>,<sensor>]
		distances = np.sqrt((np.subtract(PEsPositions[:,:,np.newaxis],sensorsPositions[:,np.newaxis,:])**2).sum(axis=0))
		
		# for each sensor, the index of the PE which is closest to it
		iClosestPEtoSensors = distances.argmin(axis=0)
		
		#import matplotlib.pylab as plt
		#plt.ion()
		#figure = plt.figure()
		#ax = figure.gca()
		#figure.hold(True)
		##ax.plot(points[0,:],points[1,:],linewidth=0,marker='+',color='blue')
		#ax.plot(sensorsPositions[0,:],sensorsPositions[1,:],linewidth=0,marker='+',color='blue')
		#ax.plot(PEsPositions[0,:],PEsPositions[1,:],linewidth=0,marker='d',color='red')
		
		#for iSensor,iPE in enumerate(iClosestPEtoSensors):
			#ax.plot([sensorsPositions[0,iSensor],PEsPositions[0,iPE]],[sensorsPositions[1,iSensor],PEsPositions[1,iPE]],linewidth=2)
		
		#figure.show()
		
		#import code
		#code.interact(local=dict(globals(), **locals()))
		
		return [list(np.where(iClosestPEtoSensors==iPE)[0]) for iPE in range(nPEs)]
