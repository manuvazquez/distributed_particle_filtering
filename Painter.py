import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plotMSEvsTime(centralizedPF_MSE,distributedPF_MSE,centralizedPFcolor,distributedPFcolor,centralizedPFmarker,distributedPFmarker,outputFile):

	# interactive mode on
	plt.ion()

	# a new figure is created to plot the MSE vs time
	mseVsTimeFigure = plt.figure('MSE vs Time')
	mseVsTimeAxes = plt.axes()

	mseVsTimeAxes.plot(centralizedPF_MSE,color=centralizedPFcolor,marker=centralizedPFmarker,label='Centralized PF')

	mseVsTimeAxes.hold(True)

	mseVsTimeAxes.plot(distributedPF_MSE,color=distributedPFcolor,marker=distributedPFmarker,label='Distributed PF')

	# the labes are shown
	mseVsTimeAxes.legend()

	plt.savefig(outputFile)

def plotAggregatedWeightsDistributionVsTime(aggregatedWeights,outputFile='aggregatedWeightsVsTime.eps',xticksStep=10):
	
	# interactive mode on
	plt.ion()

	# a new figure is created to plot the MSE vs time
	aggregatedWeightsVsTimeFigure = plt.figure('Aggregated Weights Evolution')
	aggregatedWeightsVsTimeAxes = plt.axes()
	
	# the shape of the array with the aggregated weights is used to figure out the number of PEs and time instants
	nTimeInstants,nPEs = aggregatedWeights.shape

	# the aggregated weights are represented normalized
	normalizedAggregatedWeights = np.divide(aggregatedWeights,aggregatedWeights.sum(axis=1)[np.newaxis].T)

	# in order to keep tabs on the sum of the aggregated weights already represented at each time instant
	accum = np.zeros(nTimeInstants)

	# positions for the bars corresponding to the different time instants
	t = range(nTimeInstants)

	# the colors associated to the different PEs are generated randomly
	PEsColors = np.random.rand(nPEs,3)

	for i in range(nPEs):
		
		aggregatedWeightsVsTimeAxes.bar(t,normalizedAggregatedWeights[:,i],bottom=accum,color=PEsColors[i,:])
		accum += normalizedAggregatedWeights[:,i]
	
	aggregatedWeightsVsTimeAxes.set_xticks(np.arange(0.5,nTimeInstants,xticksStep))
	aggregatedWeightsVsTimeAxes.set_xticklabels(range(0,nTimeInstants,xticksStep))
	aggregatedWeightsVsTimeAxes.set_xbound(upper=nTimeInstants)
	
	aggregatedWeightsVsTimeAxes.set_yticks([0,0.5,1])
	aggregatedWeightsVsTimeAxes.set_ybound(upper=1)

	plt.savefig(outputFile)

def plotAggregatedWeightsSupremumVsTime(maxWeights,upperBound,outputFile='maxAggregatedWeightVsTime.eps',xticksStep=10):
	
	# interactive mode on
	plt.ion()

	# a new figure is created to plot the MSE vs time
	maxAggregatedWeightVsTimeFigure = plt.figure('Aggregated Weights Supremum Vs Time')
	maxAggregatedWeightVsTimeAxes = plt.axes()

	# this is plotted along time
	maxAggregatedWeightVsTimeAxes.plot(maxWeights,label='Supremum')
	
	# the x-axis is adjusted so that it ends exactly at the last time instant
	maxAggregatedWeightVsTimeAxes.set_xbound(upper=len(maxWeights)-1)
	
	# the y-axis goes up to 1
	maxAggregatedWeightVsTimeAxes.set_ybound(upper=upperBound*4,lower=0)
	
	# the upper bound is plotted
	maxAggregatedWeightVsTimeAxes.axhline(y=upperBound,linewidth=2, color='red',linestyle='dashed',label='$c/M^{1-{\\varepsilon}}$')
	
	# in order to show the legend
	maxAggregatedWeightVsTimeAxes.legend(loc='upper right')

	plt.savefig(outputFile)

class RoomPainter:
	
	def __init__(self,sensorsPositions,sleepTime=0.5,outputFile='trajectory.eps'):
		
		self._sensorsPositions = sensorsPositions
		self._sleepTime = sleepTime
		self._outputFile = outputFile
		
		self._figure = plt.figure('Room')
		self._ax = plt.axes()
		
		# in order to draw a segment from the previous position to the current one, we need to remember the former
		self._previousPosition = None
		
		# the same for the estimates, but since there can be more than one, we use a dictionary
		self._previousEstimates = {}
		
		# used to erase the previous particles and paint the new ones
		self._particles = {}
		
		# in order to avoid a legend entry per plotted segment
		self._legendEntries = []
		
		# so that the program doesn't wait for the open windows to be closed in order to continue
		plt.ion()
		
	def setupSensors(self):
		
		self._ax.plot(self._sensorsPositions[0,:], self._sensorsPositions[1,:],color='red',marker='+',linewidth=0)
		self._ax.set_aspect('equal', 'datalim')
		
		# plot now...
		plt.show()
		
	def updateTargetPosition(self,position):
		
		plt.hold(True)
		
		# if this is not the first update (i.e., there exists a previous position)...
		if self._previousPosition is not None:
			# ...plot the step taken
			self._ax.plot(np.array([self._previousPosition[0],position[0]]),np.array([self._previousPosition[1],position[1]]),linestyle='-',color='red')
		# if this is the first update...
		else:
			# ...just plot the position keeping the handler...
			p, = self._ax.plot(position[0],position[1],color='red')
			
			# we add this to the list of entries in the legend (just once!!)
			self._legendEntries.append(p)
		
		self._previousPosition = position
		
		# plot now...
		plt.draw()

		# ...and wait...
		plt.pause(self._sleepTime)
	
	def updateEstimatedPosition(self,position,identifier='unnamed',color='blue'):
		
		plt.hold(True)
		
		# if this is not the first update (i.e., there exists a previous estimate)...
		if identifier in self._previousEstimates:
			# ...plot the step taken
			self._ax.plot(np.array([self._previousEstimates[identifier][0],position[0]]),np.array([self._previousEstimates[identifier][1],position[1]]),linestyle='-',color=color)
		# if this is the first update...
		else:
			# ...just plot the position keeping the handler...
			p, = self._ax.plot(position[0],position[1],color=color)
			
			# ...to add it to the legend
			self._legendEntries.append(p)
		
		self._previousEstimates[identifier] = position
		
		# plot now...
		plt.draw()

	def updateParticlesPositions(self,positions,identifier='unnamed',color='blue'):

		# if previous particles are being displayed...
		if identifier in self._particles:
			# ...we erase them
			self._ax.lines.remove(self._particles[identifier])

		self._particles[identifier], = self._ax.plot(positions[0,:],positions[1,:],color=color,marker='o',linewidth=0)
		
		# plot now...
		plt.draw()
		
	def save(self):
		
		# just in case...the current figure is set to the proper value
		plt.figure(self._figure.number)
		
		
		self._ax.legend(self._legendEntries,['real'] + list(self._previousEstimates.keys()),ncol=3)
		
		#self._ax.set_ybound(lower=-8)
		
		plt.savefig(self._outputFile)
		
	def close(self):
		
		plt.close(self._figure)

class RoomPainterDecorator(RoomPainter):
	
	def __init__(self,decorated):

		# the superclass constructor is not called so that we don't duplicate attributes
		
		# we keep a reference to the object being decorated
		self._decorated = decorated

	# let the decorated class do its stuff when the methods are not redefined in subclasses:
	def setupSensors(self):

		self._decorated.setupSensors()

	def updateTargetPosition(self,position):
		
		self._decorated.updateTargetPosition(position)

	def updateParticlesPositions(self,positions,identifier='unnamed',color='blue'):
		
		self._decorated.updateParticlesPositions(positions,identifier=identifier,color=color)
		
	def updateEstimatedPosition(self,position,identifier='unnamed',color='blue'):
		
		self._decorated.updateEstimatedPosition(position,identifier=identifier,color=color)
		
	def save(self):
		
		#self._decorated._ax.set_ybound(lower=self._roomBottomLeftCorner[1]+5)
		#self._decorated._ax.set_ybound(upper=self._roomTopRightCorner[1]-5)
		
		self._decorated.save()
		
	def close(self):
		
		self._decorated.close()
		
class WithBorder(RoomPainterDecorator):
	
	def __init__(self,decorated,roomBottomLeftCorner,roomTopRightCorner):
		
		super().__init__(decorated)
		
		self._roomBottomLeftCorner = roomBottomLeftCorner
		self._roomTopRightCorner = roomTopRightCorner
		self._roomDiagonalVector = self._roomTopRightCorner - self._roomBottomLeftCorner
		
	def setupSensors(self):
		
		# let the decorated class do its thing
		self._decorated.setupSensors()
		
		# we define a rectangular patch...
		roomEdge = matplotlib.patches.Rectangle((self._roomBottomLeftCorner[0],self._roomBottomLeftCorner[1]), self._roomDiagonalVector[0], self._roomDiagonalVector[1], fill=False, color='blue')
		
		# ...and added to the axes
		self._decorated._ax.add_patch(roomEdge)