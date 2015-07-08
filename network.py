#import abc
import math

import numpy as np
import scipy.cluster.vq

class Network:
	
	def __init__(self,bottomLeftCorner,topRightCorner,nPEs,nSensors):
		
		self._bottomLeftCorner = bottomLeftCorner
		self._topRightCorner = topRightCorner
		self._nPEs = nPEs
		self._nSensors = nSensors

		self._PEsPositions = None
		self._sensorsPositions = None

	def orderPositions(self,positions):
		
		# the initial position is the upper-left corner
		previousPos = np.array([self._bottomLeftCorner[0],self._topRightCorner[1]])
		
		# we need to modify the "positions" array during the algorithm
		positionsCopy = positions.copy()
		
		# a numpy array to store the result (just like the sensors positions, every column contains the two coordinates for a position)
		ordered = np.empty_like(positions.T)
		
		for i in range(len(positions)):
			
			# the distance from the previuos position to ALL the PEs
			distances = np.linalg.norm(previousPos - positionsCopy,axis=1)
			
			# the index of the minimum distance...
			iMin = np.argmin(distances)
			
			# ...is used to pick the next position in the "ordered" array
			ordered[:,i] = positions[iMin,:]
			
			# we make sure this PE is not going to be selected again
			positionsCopy[iMin,0] = np.Inf
			
			# the previous position is the selected new one
			previousPos = ordered[:,i]
			
		return ordered
	
	def randomPositions(self,n,nSamples):
		
		# seed for the default numpy random generator (used by scipy)
		np.random.seed(123)
		
		# ...is generated from a uniform distribution within the limits of the room
		points = np.vstack((np.random.uniform(self._bottomLeftCorner[0],self._topRightCorner[0],(1,nSamples)),np.random.uniform(self._bottomLeftCorner[1],self._topRightCorner[1],(1,nSamples))))
		
		# "nPEs" centroids for the above coordinates are computed using K-Means; initial random centroids are passed to the function so it does not generate them with its own random generator
		positions,_ = scipy.cluster.vq.kmeans(points.T,n)

		return positions


	def equispacedPositions(self,n):
		
		# a vector representing the diagonal of the rectangle, from which we compute...
		diagonal = self._topRightCorner - self._bottomLeftCorner
		
		# ...the area
		area = diagonal.prod()
		
		# if the positions are equispaced, each one should "cover" an area equal to
		areaPerSensor = area/n
		
		# if the area "covered" by each sensor is a square, then its side is
		squareSide = math.sqrt(areaPerSensor)
		
		# number of "full" squares that fit in each dimension
		nSquaresInXdimension,nSquaresInYdimension = np.floor(diagonal[0]/squareSide),np.floor(diagonal[1]/squareSide)
		
		# if by adding one position in each dimension...
		nOverfittingSensors = (nSquaresInXdimension+1)*(nSquaresInYdimension+1)
		
		# ...we get closer to the number of requested sensors...
		if (n-(nSquaresInXdimension*nSquaresInYdimension)) > (nOverfittingSensors-n):
			
			# ...we repeat the computations with the "overfitting" number of sensors
			areaPerSensor = area/nOverfittingSensors
			
			squareSide = math.sqrt(areaPerSensor)
			nSquaresInXdimension,nSquaresInYdimension = np.floor(diagonal[0]/squareSide),np.floor(diagonal[1]/squareSide)
		
		# in each dimension there is a certain length that is not covered (using % "weird" things happen sometimes...)
		remainingInXdimension = diagonal[0] - nSquaresInXdimension*squareSide
		remainingInYdimension = diagonal[1] - nSquaresInYdimension*squareSide
		
		res = np.transpose(
				np.array(
					[
						[self._bottomLeftCorner[0] + (remainingInXdimension + squareSide)/2 + i*squareSide,self._bottomLeftCorner[1] + (remainingInYdimension + squareSide)/2 + j*squareSide]
						for i in range(int(nSquaresInXdimension)) for j in range(int(nSquaresInYdimension))]))

		return res

	@property
	def PEsPositions(self):
		
		return self._PEsPositions

	@property
	def sensorsPositions(self):
		
		return self._sensorsPositions

class FixedNumberOfSensorsPerPE(Network):
	
	def __init__(self,bottomLeftCorner,topRightCorner,nPEs,nSensors,radius=2,phase=0,nSamples=10000):
		
		super().__init__(bottomLeftCorner,topRightCorner,nPEs,nSensors)
		
		self._phase = phase
		self._radius = radius
		
		# there should be an integer number of sensors per PE...
		assert(nSensors % nPEs == 0)
		
		self._PEsPositions = self.orderPositions(self.randomPositions(nPEs,nSamples))
		#self._PEsPositions = self.equispacedPositions(self._nPEs)
		
		# ...which is
		nSensorsPerPE = nSensors // nPEs
		
		# the sensors will be positioned at these angles around each PE
		angles = self._phase + np.arange(0,2*np.pi,2*np.pi/nSensorsPerPE)
		
		self._sensorsPositions = np.empty((2,nSensors))
		iSensor = 0
		
		for PEposition in self._PEsPositions.T:
			
			for angle in angles:
				
				self._sensorsPositions[:,iSensor] = PEposition + np.array([self._radius*np.cos(angle),self._radius*np.sin(angle)])
				
				iSensor += 1
		
		assert(np.all((self._bottomLeftCorner[0] < self._sensorsPositions[0,:]) & (self._sensorsPositions[0,:] < self._topRightCorner[0])))
		assert(np.all((self._bottomLeftCorner[1] < self._sensorsPositions[1,:]) & (self._sensorsPositions[1,:] < self._topRightCorner[1])))

class PositionlessPEsEquispacedSensors(Network):
	
	def __init__(self,bottomLeftCorner,topRightCorner,nPEs,nSensors):
		
		super().__init__(bottomLeftCorner,topRightCorner,nPEs,nSensors)
		
		self._sensorsPositions = self.equispacedPositions(self._nSensors)

class RandomlyStrewnSensorsAndPEs(Network):
	
	def __init__(self,bottomLeftCorner,topRightCorner,nPEs,nSensors,nSamples):
		
		super().__init__(bottomLeftCorner,topRightCorner,nPEs,nSensors)
		
		self._sensorsPositions = self.randomPositions(nSensors,nSamples).T
		self._PEsPositions = self.orderPositions(self.randomPositions(nPEs,nSamples))