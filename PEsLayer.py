import numpy as np
import scipy.cluster.vq

class PEsLayer:
	
	def __init__(self,bottomLeftCorner,topRightCorner,sensorsPositions=None,parameters=None):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		self._sensorsPositions = sensorsPositions
		self._parameters = parameters
		
	def orderedPEsPositions(self,PEsPositions):
		
		# the initial position is the upper-left corner
		previousPos = np.array([self._bottomLeftCorner[0],self._topRightCorner[1]])
		
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
	
	def getPositions(self,nPEs):
		
		pass

class RandomPEsLayer(PEsLayer):
	
	def getPositions(self,nPEs):
		
		# the required parameter is extracted from the dictionary
		nPoints = self._parameters['number of uniform samples']
		
		# the seed of the Pseudo Random Numbers Generator to be used below (so that the positions obtained for the PEs stay the same through different runs)
		PRNG = np.random.RandomState(1234567)
		
		# ...is generated from a uniform distribution within the limits of the room
		points = np.vstack((PRNG.uniform(self._bottomLeftCorner[0],self._topRightCorner[0],(1,nPoints)),PRNG.uniform(self._bottomLeftCorner[1],self._topRightCorner[1],(1,nPoints))))
		
		# "nPEs" centroids for the above coordinates are computed using K-Means; initial random centroids are passed to the function so it does not generate them with its own random generator
		PEsPositions,_ = scipy.cluster.vq.kmeans(points.T,points.T[PRNG.choice(points.shape[1],nPEs),:])

		return self.orderedPEsPositions(PEsPositions)

class ClusteringSensorsPEsLayer(PEsLayer):
	
	def getPositions(self,nPEs):

		# seed for the default numpy random generator (used by scipy)
		np.random.seed(123)
		
		# k-means over the sensors positions using "nPEs" centroids
		PEsPositions,_ = scipy.cluster.vq.kmeans(self._sensorsPositions.T,nPEs)
		
		# just in case K-means doesn't return the proper number of sensors
		assert(len(PEsPositions)==nPEs)
		
		return self.orderedPEsPositions(PEsPositions)