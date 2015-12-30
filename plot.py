import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def setup_axes(figure_id, clear_figure=True):
	
	# interactive mode on
	plt.ion()

	# a new figure is created...
	fig = plt.figure(figure_id)
	
	# ...and, if requested,...
	if clear_figure:
		# ...cleared, just in case this method is called several times with the same "figure_id"
		plt.clf()
	
	# ...and the corresponding axes created
	axes = plt.axes()
	
	return axes, fig


def markers_list():

	# all the available markers in matplotlib
	available_markers = list(matplotlib.markers.MarkerStyle.markers)

	# the "dummy" ones are removed
	available_markers.remove('None')
	available_markers.remove('')
	available_markers.remove(' ')

	return available_markers


def distributed_against_centralized_particle_filter(
		x, centralize_particle_filter, distributed_particle_filter, extra_line=None, output_file=None,
		centralized_particle_filter_parameters={'label': 'Centralized PF'},
		distributed_particle_filter_parameters={'label': 'Distributed PF'}, extra_line_parameters={},
		figure_id='vs Time', axes_properties={}):

	# a new pair of axes is set up
	ax, fig = setup_axes(figure_id)
	
	ax.plot(x, centralize_particle_filter, **centralized_particle_filter_parameters)

	ax.plot(x, distributed_particle_filter, **distributed_particle_filter_parameters)
	
	if extra_line is not None:
		
		ax.plot(x, extra_line, **extra_line_parameters)

	# the labels are shown
	ax.legend()

	# the x axis is adjusted so that no empty space is left before the beginning of the plot
	ax.set_xbound(lower=x[0], upper=x[-1])
	
	# set any additional properties for the axes
	ax.set(**axes_properties)
	
	if output_file:
	
		plt.savefig(output_file)
	
	return ax, fig


def particle_filters(
		x, y, output_file, algorithms_parameters, figure_id='vs Time', axes_properties={}, maximized=False,
		colormap=None):

	# a new pair of axes is set up
	ax, fig = setup_axes(figure_id)
	
	assert y.shape[0] == len(x)
	assert y.shape[1] == len(algorithms_parameters)
	
	# this is needed since x may be a "range" rather than a numpy array
	try:
		
		several_x = (x.ndim > 1)
		
	except AttributeError:
		
		several_x = False

	# if a colormap is passed...
	if colormap:

		# a new dictionary of parameters is built for every algorithm dropping "color"
		algorithms_parameters = [{k: v for k, v in d.items() if k is not 'color'} for d in algorithms_parameters]

		# the requested color map is obtained
		color_map = plt.get_cmap(colormap)

		# the number of colors needed
		n_colors = len(algorithms_parameters)

		# colors are automatically picked up equally spaced from the color map
		ax.set_color_cycle([color_map(1.*i/n_colors) for i in range(n_colors)])

	# if several sets of x coordinates were passed
	if several_x:
	
		for xcol, ycol, param in zip(x.T, y.T, algorithms_parameters):
			
			ax.plot(xcol, ycol, **param)
			
	else:
		
		for data, param in zip(y.T, algorithms_parameters):
			
			ax.plot(x, data, **param)
	
	# the labels are shown
	ax.legend()

	# the x axis is adjusted so that no empty space is left before the beginning of the plot
	if several_x:
		
		ax.set_xbound(lower=x[0, :].min(), upper=x[-1, :].max())
		
	else:

		ax.set_xbound(lower=x[0], upper=x[-1])
	
	# set any additional properties for the axes
	ax.set(**axes_properties)
	
	if maximized:
		
		# the window is maximized through the figure manager
		plt.get_current_fig_manager().window.showMaximized()
	
	# show the plot...now!!
	fig.show()
	
	if output_file:
	
		plt.savefig(output_file)
	
	return ax, fig


def aggregated_weights_distribution_vs_time(
		aggregated_weights, output_file='aggregatedWeightsVsTime.pdf', xticks_step=10):

	# the corresponding axes are created
	ax, _ = setup_axes('Aggregated Weights Evolution')
	
	# the shape of the array with the aggregated weights is used to figure out the number of PEs and time instants
	n_time_instants, n_processing_elements = aggregated_weights.shape

	# the aggregated weights are represented normalized
	normalized_aggregated_weights = np.divide(aggregated_weights, aggregated_weights.sum(axis=1)[np.newaxis].T)

	# in order to keep tabs on the sum of the aggregated weights already represented at each time instant
	accum = np.zeros(n_time_instants)

	# positions for the bars corresponding to the different time instants
	t = range(n_time_instants)

	# the colors associated to the different PEs are generated randomly
	processing_elements_colors = np.random.rand(n_processing_elements, 3)

	for i in range(n_processing_elements):
		
		ax.bar(t, normalized_aggregated_weights[:, i], bottom=accum, color=processing_elements_colors[i, :])
		accum += normalized_aggregated_weights[:, i]
	
	ax.set_xticks(np.arange(0.5, n_time_instants, xticks_step))
	ax.set_xticklabels(range(0, n_time_instants, xticks_step))
	ax.set_xbound(upper=n_time_instants)
	
	ax.set_yticks([0, 0.5, 1])
	ax.set_ybound(upper=1)

	plt.savefig(output_file)


def aggregated_weights_supremum_vs_time(
		max_weights, upper_bound, output_file='maxAggregatedWeightVsTime.pdf',
		step_exchange_period=1, supremum_line_properties={'label': 'Supremum', 'linestyle': ':'},
		supremum_at_exchange_steps_line_properties={
			'label': 'Exchange steps', 'linestyle': '.', 'marker': 'D', 'color': 'black'},
		upper_bound_line_properties={
			'label': '$c^q/M^{q-\\varepsilon}$', 'linestyle': 'dashed', 'color': 'red', 'linewidth': 2},
		figure_id='Aggregated Weights Supremum Vs Time', axes_properties={}, plot_everything=True, ybound_factor=4):
	
	n_time_instants = len(max_weights)

	# the corresponding axes are created
	ax, fig = setup_axes(figure_id)
	
	# for the x-axis
	t = np.arange(n_time_instants)

	if plot_everything:
		# this is plotted along time
		ax.plot(t, max_weights[t], **supremum_line_properties)
	
	# the time instants at which step exchanges occur...
	t_exchange_steps = np.arange(step_exchange_period-1, n_time_instants, step_exchange_period)
	
	# ...are plotted with different markers
	ax.plot(t_exchange_steps, max_weights[t_exchange_steps], **supremum_at_exchange_steps_line_properties)
	
	# the x-axis is adjusted so that it ends exactly at the last time instant
	ax.set_xbound(lower=t[0], upper=t[-1])
	
	# the y-axis goes up to 1
	ax.set_ybound(upper=upper_bound*ybound_factor, lower=0)
	
	# the upper bound is plotted
	ax.axhline(y=upper_bound, **upper_bound_line_properties)
	
	# in order to show the legend
	ax.legend(loc='upper right')
	
	# set any additional properties for the axes
	ax.set(**axes_properties)
	
	if output_file:

		plt.savefig(output_file)
	
	return ax, fig


def aggregated_weights_supremum_vs_n_processing_elements(
		Ms, max_weights, upper_bounds=None, output_file='maxAggregatedWeightVsM.pdf', supremum_line_properties={},
		upper_bound_line_properties={
			'color': 'red', 'label': '$c^q/M^{q-\\varepsilon}$', 'marker': '+', 'markersize': 10,
			'markeredgewidth': 2, 'linestyle': ':'}, figure_id='Aggregated Weights Supremum Vs M', axes_properties={}):
	
	# the corresponding axes are created
	ax, fig = setup_axes(figure_id)
	
	# this is plotted along time
	ax.loglog(Ms, max_weights, **supremum_line_properties)
	
	if upper_bounds:
	
		# the bound
		ax.loglog(Ms, upper_bounds, **upper_bound_line_properties)
		
	# only the ticks corresponding to the values of M
	ax.set_xticks(Ms)
	
	# so that the ticks show up properly when the x axis is logarithmic
	ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	
	# the x-axis is adjusted so that it ends exactly at the last time instant
	ax.set_xbound(lower=Ms[0], upper=Ms[-1])

	if upper_bounds:
		
		# in order to show the legend
		ax.legend(loc='upper right')
	
	# set any additional properties for the axes
	ax.set(**axes_properties)

	if output_file:

		plt.savefig(output_file)
	
	return ax, fig


def trajectory(filename, i_trajectory=0, n_time_instants=-1, ticks_font_size=12):
	
	import network_nodes
	import pickle
	import os
	import scipy.io
	
	position = scipy.io.loadmat(filename)['targetPosition'][..., i_trajectory]
	
	# parameters are loaded
	with open(os.path.splitext(filename)[0] + '.parameters', "rb") as f:
		
		# ...is loaded
		parameters = pickle.load(f)[0]

	n_processing_elements = parameters['topologies'][parameters['topologies']['type'][0]]['number of PEs']
	n_sensors = parameters['sensors']['number']

	# the positions of the sensors are computed
	sensors_positions = network_nodes.PositionlessPEsEquispacedSensors(
		parameters["room"]["bottom left corner"], parameters["room"]["top right corner"], n_processing_elements, n_sensors
	).sensors_positions

	# a Painter object is created to do the dirty work
	painter = TightRectangularRoomPainter(
		parameters["room"]["bottom left corner"], parameters["room"]["top right corner"],
		sensors_positions, ticksFontSize=ticks_font_size)
	
	painter.setup()
	
	# if the number of time instants to be plotted received is not within the proper limits...
	if not (0 < n_time_instants <= position.shape[1]):
		
		# ...the entire trajectory is plotted
		n_time_instants = position.shape[1]

		print(
			'trajectory: the number of time instants to be plotted is not within the limits...plotting the entire sequence...')
	
	for i in range(n_time_instants):
		
		painter.updateTargetPosition(position[:, i])
	
	painter.save()


def trajectory_from_hdf5(filename, i_trajectory=0, n_time_instants=-1, ticks_font_size=12):

	import pickle
	import os
	import h5py

	import simulation

	data_file = h5py.File(filename,'r')
	position = simulation.Mposterior.parse_hdf5(data_file)[0][..., i_trajectory]

	# parameters are loaded
	with open(os.path.splitext(filename)[0] + '.parameters', "rb") as f:

		# ...is loaded
		parameters = pickle.load(f)[0]

	if isinstance(parameters['topologies types'], list):

		n_processing_elements = parameters['topologies'][parameters['topologies types'][0]]['number of PEs']

	else:

		n_processing_elements = parameters['topologies'][parameters['topologies types']]['number of PEs']

	# the positions of the sensors are extracted
	sensors_positions = data_file['sensors/positions'][...]

	proc_elem_positions = np.zeros((2, n_processing_elements))
	proc_elem_sensors_connections = []
	proc_elem_neighbours = []

	for i_PE in range(n_processing_elements):

		proc_elem_positions[:,i_PE] = data_file['PEs/{}/position'.format(i_PE)][...]
		proc_elem_sensors_connections.append(list(data_file['PEs/{}/connected sensors'.format(i_PE)][...]))
		proc_elem_neighbours.append(list(data_file['PEs/{}/neighbours'.format(i_PE)][...]))

	painter = TightRectangularRoomPainterWithPEs(
		data_file['room/bottom left corner'][...], data_file['room/top right corner'][...], sensors_positions,
		proc_elem_positions, proc_elem_sensors_connections, proc_elem_neighbours,
		sleepTime=parameters["painter"]["sleep time between updates"])

	painter.setup()

	# if the number of time instants to be plotted received is not within the proper limits...
	if not (0 < n_time_instants <= position.shape[1]):

		# ...the entire trajectory is plotted
		n_time_instants = position.shape[1]

		print(
			'trajectory: the number of time instants to be plotted is not within the limits...plotting the entire sequence...')

	for i in range(n_time_instants):

		painter.updateTargetPosition(position[:, i])

	painter.save()

	# HDF5 must be closed
	data_file.close()

class RoomPainter:
	
	def __init__(self, sensorsPositions, sleepTime=0.5):
		
		self._sensorsPositions = sensorsPositions
		self._sleepTime = sleepTime
		
		# in order to draw a segment from the previous position to the current one, we need to remember the former
		self._previousPosition = None
		
		# the same for the estimates, but since there can be more than one, we use a dictionary
		self._previousEstimates = {}
		
		# used to erase the previous particles and paint the new ones
		self._particles = {}
		
		# in order to avoid a legend entry per plotted segment
		self._legendEntries = []
		
		# a new pair of axes is set up
		self._ax, self._figure = setup_axes('Room', clear_figure=False)
		
	def setup(self, sensorsLineProperties={'marker':'+','color':'red'}):
		
		# linewidth=0 so that the points are not joined...
		self._ax.plot(self._sensorsPositions[0,:], self._sensorsPositions[1,:], linewidth=0, **sensorsLineProperties)

		self._ax.set_aspect('equal', 'datalim')
		
		# show...now!
		self._figure.show()
		
	def updateTargetPosition(self,position):
		
		# if this is not the first update (i.e., there exists a previous position)...
		if self._previousPosition is not None:
			# ...plot the step taken
			self._ax.plot(np.array([self._previousPosition[0],position[0]]),np.array([self._previousPosition[1],position[1]]),linestyle='-',color='red')
		# if this is the first update...
		else:
			# ...just plot the position keeping the handler...
			p, = self._ax.plot(position[0],position[1],color='red',marker='d',markersize=10)
			
			# we add this to the list of entries in the legend (just once!!)
			self._legendEntries.append(p)
		
		self._previousPosition = position
		
		# plot now...
		self._figure.canvas.draw()

		# ...and wait...
		plt.pause(self._sleepTime)
	
	def updateEstimatedPosition(self,position,identifier='unnamed',color='blue'):
		
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
		self._figure.canvas.draw()

	def updateParticlesPositions(self,positions,identifier='unnamed',color='blue'):

		# if previous particles are being displayed...
		if identifier in self._particles:
			# ...we erase them
			self._ax.lines.remove(self._particles[identifier])

		self._particles[identifier], = self._ax.plot(positions[0,:],positions[1,:],color=color,marker='o',linewidth=0)
		
		# plot now...
		self._figure.canvas.draw()
		
	def save(self, outputFile='trajectory.pdf'):
		
		# just in case...the current figure is set to the proper value
		plt.figure(self._figure.number)
		
		self._ax.legend(self._legendEntries,['real'] + list(self._previousEstimates.keys()),ncol=3)
		
		plt.savefig(outputFile)
		
	def close(self):
		
		plt.close(self._figure)


class RectangularRoomPainter(RoomPainter):
	
	def __init__(self,roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime=0.5):
		
		super().__init__(sensorsPositions,sleepTime=sleepTime)
		
		self._roomBottomLeftCorner = roomBottomLeftCorner
		self._roomTopRightCorner = roomTopRightCorner
		self._roomDiagonalVector = self._roomTopRightCorner - self._roomBottomLeftCorner
		
	def setup(self,borderLineProperties={'color':'blue'},sensorsLineProperties = {'marker':'+','color':'red'}):
		
		# let the parent class do its thing
		super().setup(sensorsLineProperties=sensorsLineProperties)
		
		# we define a rectangular patch...
		roomEdge = matplotlib.patches.Rectangle((self._roomBottomLeftCorner[0],self._roomBottomLeftCorner[1]), self._roomDiagonalVector[0], self._roomDiagonalVector[1], fill=False,**borderLineProperties)

		# ...and added to the axes
		self._ax.add_patch(roomEdge)


class TightRectangularRoomPainter(RectangularRoomPainter):
	
	def __init__(self,roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime=0.1,ticksFontSize=14):
		
		super().__init__(roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime=sleepTime)
		
		self._ticksFontSize = ticksFontSize
		
		# the figure created by the superclass is discarded
		self.close()
		
		self._figure = plt.figure('Room',figsize=tuple(self._roomDiagonalVector//4))
		self._ax = self._figure.add_axes((0,0,1,1))
		
	def setup(self,borderLineProperties={'color':'black','linewidth':4},sensorsLineProperties = {'marker':'x','color':(116/255,113/255,209/255),'markersize':10,'markeredgewidth':5}):
		
		# let the parent class do its thing
		super().setup(borderLineProperties=borderLineProperties,sensorsLineProperties=sensorsLineProperties)
		
		# axis are removed
		#plt.axis('off')
		
		# the font size of the ticks in both axes is set
		self._ax.tick_params(axis='both',labelsize=self._ticksFontSize)
		
	def save(self,outputFile='trajectory.pdf'):
		
		# just in case...the current figure is set to the proper value
		plt.figure(self._figure.number)
		
		plt.savefig(outputFile,bbox_inches="tight")


class TightRectangularRoomPainterWithPEs(TightRectangularRoomPainter):
	
	def __init__(self,roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,PEsPositions,connections,PEsPEsConnections,sleepTime=0.1,ticksFontSize=14):
		
		super().__init__(roomBottomLeftCorner,roomTopRightCorner,sensorsPositions,sleepTime,ticksFontSize)
		
		self._PEsPositions = PEsPositions
		self._connections = connections
		self._PEsPEsConnections = PEsPEsConnections
	
	def setup(self,borderLineProperties={'color':'black','linewidth':4},sensorsLineProperties = {'marker':'x','color':(116/255,113/255,209/255),'markersize':10,'markeredgewidth':5}):
		
		super().setup(borderLineProperties=borderLineProperties,sensorsLineProperties=sensorsLineProperties)

		self._ax.plot(self._PEsPositions[0,:],self._PEsPositions[1,:],linewidth=0,marker='s',color='red')
		
		for iPE,pos in enumerate(self._PEsPositions.T):
			
			self._ax.annotate('#{}'.format(iPE),xy=tuple(pos))
		
		# in "self._connections", for every PE there is a list of sensors associated
		for iPE,sensorsIndexes in enumerate(self._connections):
			
			# for every sensor associated with the PE being processed
			for iSensor in sensorsIndexes:
			
				self._ax.plot([self._sensorsPositions[0,iSensor],self._PEsPositions[0,iPE]],[self._sensorsPositions[1,iSensor],self._PEsPositions[1,iPE]],linewidth=2,linestyle='--')
				
		for iPE,iNeighbours in enumerate(self._PEsPEsConnections):
			
			for i in iNeighbours:
				
				self._ax.plot([self._PEsPositions[0,iPE],self._PEsPositions[0,i]],[self._PEsPositions[1,iPE],self._PEsPositions[1,i]],linewidth=1,linestyle=':',color='gray')