import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plotDistributedAgainstCentralizedVsTime(centralizedPF,distributedPF,centralizedPFcolor,distributedPFcolor,centralizedPFmarker,distributedPFmarker,outputFile,iFrom=0,
											centralizedPFparameters={'label':'Centralized PF'},distributedPFparameters={'label':'Distributed PF'},figureId='vs Time'):

	# interactive mode on
	plt.ion()

	# a new figure is created to plot the MSE vs time
	vsTimeFigure = plt.figure(figureId)
	
	# ...cleared (just in case this method is called several times)
	plt.clf()
	
	# ...and the corresponding axes created
	metricVsTimeAxes = plt.axes()
	
	# range specifying the time instants being plotted
	iTime = np.arange(iFrom,centralizedPF.shape[0])
	
	metricVsTimeAxes.plot(iTime,centralizedPF[iTime],color=centralizedPFcolor,marker=centralizedPFmarker,**centralizedPFparameters)

	metricVsTimeAxes.hold(True)

	metricVsTimeAxes.plot(iTime,distributedPF[iTime],color=distributedPFcolor,marker=distributedPFmarker,**distributedPFparameters)

	# the labes are shown
	metricVsTimeAxes.legend()
	
	# the x axis is adjusted so that no empty space is left before the beginning of the plot
	metricVsTimeAxes.set_xbound(lower=iFrom)

	plt.savefig(outputFile)

def plotAggregatedWeightsDistributionVsTime(aggregatedWeights,outputFile='aggregatedWeightsVsTime.eps',xticksStep=10):
	
	# interactive mode on
	plt.ion()

	# a new figure is created to plot the aggregated weights fmp vs time
	aggregatedWeightsVsTimeFigure = plt.figure('Aggregated Weights Evolution')
	
	# ...cleared (just in case this method is called several times)
	plt.clf()
	
	# ...and the corresponding axes created
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

def plotAggregatedWeightsSupremumVsTime(maxWeights,upperBound,
										outputFile='maxAggregatedWeightVsTime.eps',
										stepExchangePeriod=1,addMarksOnStepExchangeInstants=False,
										ylabel='$c/M^{1-{\\varepsilon}}$',figureId='Aggregated Weights Supremum Vs Time'):
	
	nTimeInstants = len(maxWeights)
	
	# interactive mode on
	plt.ion()

	# a new figure is created to plot the aggregated weights' supremum vs time...
	maxAggregatedWeightVsTimeFigure = plt.figure(figureId)

	# ...cleared (just in case this method is called several times)
	plt.clf()
	
	# ...and the corresponding axes created
	maxAggregatedWeightVsTimeAxes = plt.axes()
	
	if addMarksOnStepExchangeInstants:
		
		# for the x-axis
		t = np.arange(nTimeInstants)
		
		# this is plotted along time
		maxAggregatedWeightVsTimeAxes.plot(t,maxWeights[t],label='Supremum',linestyle=':')
		
		# the time instants at which step exchanges occur...
		tExchangeSteps = np.arange(stepExchangePeriod-1,nTimeInstants,stepExchangePeriod)
		
		# ...are plotted with different markers
		maxAggregatedWeightVsTimeAxes.plot(tExchangeSteps,maxWeights[tExchangeSteps],label='Exchange steps',linestyle='.',marker='D',color='black')
		
	else:
	
		# for the x-axis
		t = np.arange(stepExchangePeriod-1,nTimeInstants,stepExchangePeriod)

		# this is plotted along time
		maxAggregatedWeightVsTimeAxes.plot(t,maxWeights[t],label='Supremum')
	
	# the x-axis is adjusted so that it ends exactly at the last time instant
	maxAggregatedWeightVsTimeAxes.set_xbound(lower=t[0],upper=t[-1])
	
	# the y-axis goes up to 1
	maxAggregatedWeightVsTimeAxes.set_ybound(upper=upperBound*4,lower=0)
	
	# the upper bound is plotted
	maxAggregatedWeightVsTimeAxes.axhline(y=upperBound,linewidth=2, color='red',linestyle='dashed',label=ylabel)
	
	# in order to show the legend
	maxAggregatedWeightVsTimeAxes.legend(loc='upper right')

	plt.savefig(outputFile)

def plotTrajectory(filename,nTimeInstants=-1):
	
	import Sensor
	import pickle
	import os
	
	# data file is loaded
	with np.load(filename) as data:
		
		position = np.append(data['targetInitialPosition'],data['targetPosition'],axis=1)
	
	# parameters are loaded
	with open(os.path.splitext(filename)[0] + '.parameters',"rb") as f:
		
		# ...is loaded
		parameters = pickle.load(f)[0]
	
	# the positions of the sensors are computed
	sensorsPositions = Sensor.EquispacedOnRectangleSensorLayer(parameters["room"]["bottom left corner"],parameters["room"]["top right corner"]).getPositions(parameters["sensors"]["number"])
	
	# a Painter object is created to do the dirty work
	painter = TightRectangularRoomPainter(parameters["room"]["bottom left corner"],parameters["room"]["top right corner"],sensorsPositions)
	
	painter.setup()
	
	# if the number of time instants to be plotted received is not within the proper limits...
	if not (0<nTimeInstants<=position.shape[1]):
		
		# ...the entire trajectory is plotted
		nTimeInstants = position.shape[1]
		
		print('plotTrajectory: the number of time instants to be plotted is not within the limits...plotting the entire sequence...')
	
	for i in range(nTimeInstants):
		
		painter.updateTargetPosition(position[:,i])
	
	painter.save()

class RoomPainter:
	
	def __init__(self,sensorsPositions,sleepTime=0.5):
		
		self._sensorsPositions = sensorsPositions
		self._sleepTime = sleepTime
		
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
		
	def setup(self,sensorsLineProperties = {'marker':'+','color':'red'}):
		
		# linewidth=0 so that the points are not joined...
		self._ax.plot(self._sensorsPositions[0,:], self._sensorsPositions[1,:],linewidth=0,**sensorsLineProperties)

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
		
	def save(self,outputFile='trajectory.eps'):
		
		# just in case...the current figure is set to the proper value
		plt.figure(self._figure.number)
		
		
		self._ax.legend(self._legendEntries,['real'] + list(self._previousEstimates.keys()),ncol=3)
		
		#self._ax.set_ybound(lower=-8)
		
		plt.savefig(outputFile)
		
	def close(self):
		
		plt.close(self._figure)

class RectangularRoomPainter(RoomPainter):
	
	def __init__(self,roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime=0.5):
		
		super().__init__(sensorsPositions,sleepTime=sleepTime)
		
		self._roomBottomLeftCorner = roomBottomLeftCorner
		self._roomTopRightCorner = roomTopRightCorner
		self._roomDiagonalVector = self._roomTopRightCorner - self._roomBottomLeftCorner

	def setup(self,borderLinePropertis={'color':'blue'},sensorsLineProperties = {'marker':'+','color':'red'}):
		
		# let the parent class do its thing
		super().setup(sensorsLineProperties=sensorsLineProperties)
		
		# we define a rectangular patch...
		roomEdge = matplotlib.patches.Rectangle((self._roomBottomLeftCorner[0],self._roomBottomLeftCorner[1]), self._roomDiagonalVector[0], self._roomDiagonalVector[1], fill=False,**borderLinePropertis)

		# ...and added to the axes
		self._ax.add_patch(roomEdge)

class TightRectangularRoomPainter(RectangularRoomPainter):
	
	def __init__(self,roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime=0.1):
		
		super().__init__(roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime=sleepTime)
		
		# the figure created by the superclass is discarded
		self.close()
		
		self._figure = plt.figure('Room',figsize=tuple(self._roomDiagonalVector//4))
		self._ax = self._figure.add_axes((0,0,1,1))
		
	def setup(self,borderLinePropertis={'color':'black','linewidth':4},sensorsLineProperties = {'marker':'x','color':(116/255,113/255,209/255),'markersize':10,'markeredgewidth':5}):
		
		# let the parent class do its thing
		super().setup(borderLinePropertis=borderLinePropertis,sensorsLineProperties=sensorsLineProperties)
		
		# axis are removed
		plt.axis('off')
		
	def save(self,outputFile='trajectory.eps'):
		
		# just in case...the current figure is set to the proper value
		plt.figure(self._figure.number)
		
		plt.savefig(outputFile,bbox_inches="tight")