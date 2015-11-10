import numpy as np
import math
import networkx as nx
import operator


class Topology:
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		self._n_processing_elements = n_processing_elements
		self._topology_specific_parameters = topology_specific_parameters

	@property
	def n_processing_elements(self):
		
		return self._n_processing_elements
	
	def get_neighbours(self):
		
		"""Get the list of neighbours of every PE.
		
		Returns
		-------
		samples: list of lists
			Each list contains the indexes of the neighbours of the corresponding PE.
		"""
		
		return self._neighbours

	@property
	def distances_between_processing_elements(self):

		# an empty graph
		G = nx.Graph()

		# a node per PE is added
		G.add_nodes_from(range(self._n_processing_elements))

		# this is a list to store tuples defining the edges of the graph
		edges = []

		# for every PE...
		for iPE, this_processing_element_neighbours in enumerate(self._neighbours):

			# an edge is added for every one of its neighbours
			edges.extend([(iPE, n) for n in this_processing_element_neighbours])

		# all the edges are added
		G.add_edges_from(edges)

		# paths between nodes
		paths = nx.all_pairs_shortest_path(G)

		# a numpy array in which each row yields the distance (in hops) from each node to every other node
		distances = np.empty((self._n_processing_elements, self._n_processing_elements), dtype=int)

		for iPE in range(self._n_processing_elements):

			for iNeigh in range(self._n_processing_elements):

				# the number of hops is the number of nodes in the path minus one
				distances[iPE, iNeigh] = len(paths[iPE][iNeigh])-1

		return distances

	def i_neighbours_within_hops(self, n_hops, lower_bound=0):

		distances_between_processing_elements = self.distances_between_processing_elements

		exchanging_processing_elements = []

		for neighbours in distances_between_processing_elements:

			i_within_radius, = np.where((neighbours > lower_bound) & (neighbours <= n_hops))

			exchanging_processing_elements.append(list(i_within_radius))

		return exchanging_processing_elements


class Customized(Topology):
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		super().__init__(n_processing_elements, topology_specific_parameters)
		
		# each element in the list is another list specifying the neighbours of the corresponding PE
		self._neighbours = self._topology_specific_parameters["neighbourhoods"]


class Ring(Topology):
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		super().__init__(n_processing_elements, topology_specific_parameters)
		
		self._neighbours = [
			[(i-1) % n_processing_elements, (i+1) % n_processing_elements] for i in range(n_processing_elements)]


class Mesh(Topology):
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		super().__init__(n_processing_elements, topology_specific_parameters)
		
		potential_neighbours_relative_position = self._topology_specific_parameters["neighbours"]
		n_rows, n_cols = self._topology_specific_parameters["geometry"]
		
		assert n_rows*n_cols == n_processing_elements
		
		# for the sake of clarity, and in order to avoid some computations...
		arrayed_processing_elements = np.arange(n_processing_elements).reshape((n_rows, n_cols), order='F')
		
		self._neighbours = []
		
		for j in range(n_cols):
			for i in range(n_rows):
				
				# here we store the neighbours of the PE being processed
				current_processing_element_neighbours = []
				
				# for every potential neighbour
				for neighbour_relative_position in potential_neighbours_relative_position:
					
					# we compute its position...
					
					# if the neighbours of a certain PE are allowed to "wrap around" the other side (either horizontally
					# or vertically) of the mesh...
					if topology_specific_parameters['wraparound']:
					
						i_neighbour = (i+neighbour_relative_position[0]) % n_rows
						j_neighbour = (j+neighbour_relative_position[1]) % n_cols
						
					else:
						
						i_neighbour, j_neighbour = i+neighbour_relative_position[0], j+neighbour_relative_position[1]
					
						# if the position does not corresponds to that of a PE (i.e., it is NOT within the PEs array)
						if not ((0 <= i_neighbour < n_rows) and (0 <= j_neighbour < n_cols)):
							
							continue
						
					# we add this neighbour to the list
					current_processing_element_neighbours.append(arrayed_processing_elements[i_neighbour, j_neighbour])
				
				# the list of neighbours of this PE is added to the list of lists of neighbours
				self._neighbours.append(current_processing_element_neighbours)


class ConstantDegreeSimpleGraph(Topology):
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		super().__init__(n_processing_elements, topology_specific_parameters)
		
		n_neighbours = math.ceil(
			topology_specific_parameters['number of neighbours as a percentage of the number of PEs']*n_processing_elements)
		
		# the Havel-Hakimi algorithm is used to obtain a simple graph with the requested degrees
		graph = nx.havel_hakimi_graph([n_neighbours]*n_processing_elements)
		
		self._neighbours = [graph.neighbors(i) for i in range(n_processing_elements)]


class FullyConnected(Topology):
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		super().__init__(n_processing_elements, topology_specific_parameters)
		
		self._neighbours = [[j for j in range(n_processing_elements) if j != i] for i in range(n_processing_elements)]


class LOSbased(Topology):
	
	def __init__(self, n_processing_elements, topology_specific_parameters):
		
		super().__init__(n_processing_elements, topology_specific_parameters)
		
		# for the sake of clarity...
		processing_elements_positions = topology_specific_parameters['PEs positions']
		
		# a list of sets, each one meant to store the neighbours of the corresponding PE
		res = [set() for _ in range(processing_elements_positions.shape[1])]
		
		for iPE in range(processing_elements_positions.shape[1]):
			
			# the difference vectors between the position of the current PE and the positions of ALL the PEs,...
			diff = processing_elements_positions - processing_elements_positions[:, iPE:iPE+1]
			
			# ... which allow to compute the angles
			angles = np.degrees(np.arctan2(diff[1, :], diff[0, :]))
			
			# ...and the distances
			distances = np.linalg.norm(diff, axis=0)
			
			# we need to use the "and" operator for all the comparisons except the one for the left, which requires the "or"
			for (low, up), op in zip(
					topology_specific_parameters['ranges of vision for right, up, left and down'],
					[operator.and_, operator.and_, operator.or_, operator.and_]):
			
				# the index of the PEs that are within the range of angles specified
				i_processing_elements_within_range, = np.where(op(angles >= low, angles <= up) & (distances > 0))
				
				# if any PE found within that "range of vision"
				if i_processing_elements_within_range.size:
				
					# the neighbour is chosen to be that which is closer to the current PE
					i_neighbour = i_processing_elements_within_range[np.argmin(distances[i_processing_elements_within_range])]
					
					# it is added to the list of neighbours of the current PE...
					res[iPE].add(i_neighbour)
					
					# ...and the current PE to the list of neighbours of the selected neighbour (if it wasn't yet)
					res[i_neighbour].add(iPE)
					
		self._neighbours = [list(s) for s in res]
