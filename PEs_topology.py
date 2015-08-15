import numpy as np
import math
import collections
import networkx as nx

class Topology:
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		self._nPEs = nPEs
		self._topologySpecificParameters = topologySpecificParameters
		
	def getNumberOfPEs(self):
		
		return self._nPEs
	
	def getNeighbours(self):
		
		"""Get the list of neighbours of every PE.
		
		Returns
		-------
		samples: list of lists
			Each list contains the indexes of the neighbours of the corresponding PE.
		"""
		
		return self._neighbours

	def distances_between_PEs(self):

		# an empty graph
		G = nx.Graph()

		# a node per PE is added
		G.add_nodes_from(range(self._nPEs))

		# this is a list to store tuples defining the edges of the graph
		edges = []

		# for every PE...
		for iPE,thisPEneighbours in enumerate(self._neighbours):

			# an edge is added for every one of its neighbours
			edges.extend([(iPE,n) for n in thisPEneighbours])

		# all the edges are added
		G.add_edges_from(edges)

		# paths between nodes
		paths = nx.all_pairs_shortest_path(G)

		# a numpy array in which each row yields the distance (in hops) from each node to every other node
		distances = np.empty((self._nPEs,self._nPEs))

		for iPE in range(self._nPEs):

			for iNeigh in range(self._nPEs):

				# the number of hops is the number of nodes in the path minus one
				distances[iPE,iNeigh] = len(paths[iPE][iNeigh])-1

class Customized(Topology):
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		super().__init__(nPEs,topologySpecificParameters)
		
		# each element in the list is another list specifying the neighbours of the corresponding PE
		self._neighbours = self._topologySpecificParameters["neighbourhoods"]
		
class Ring(Topology):
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		super().__init__(nPEs,topologySpecificParameters)
		
		self._neighbours = [[(i-1) % nPEs,(i+1) % nPEs] for i in range(nPEs)]

class Mesh(Topology):
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		super().__init__(nPEs,topologySpecificParameters)
		
		potentialNeighboursRelativePosition = self._topologySpecificParameters["neighbours"]
		nRows,nCols = self._topologySpecificParameters["geometry"]
		
		assert nRows*nCols == nPEs
		
		# for the sake of clarity, and in order to avoid some computations...
		arrayedPEs = np.arange(nPEs).reshape((nRows,nCols),order='F')
		
		self._neighbours = []
		
		for j in range(nCols):
			for i in range(nRows):
				
				# here we store the neighbours of the PE being processed
				currentPEneighbours = []
				
				# for every potential neighbour
				for neighbourRelativePosition in potentialNeighboursRelativePosition:
					
					# we compute its position...
					
					# if the neighbours of a certain PE are allowed to "wrap around" the other side (either horizontally or vertically) of the mesh...
					if topologySpecificParameters['wraparound']:
					
						iNeighbour,jNeighbour = (i+neighbourRelativePosition[0])%nRows,(j+neighbourRelativePosition[1])%nCols
						
					else:
						
						iNeighbour,jNeighbour = i+neighbourRelativePosition[0],j+neighbourRelativePosition[1]
					
						# if the position does not corresponds to that of a PE (i.e., it is NOT within the PEs array)
						if not ( (0 <= iNeighbour < nRows) and (0 <= jNeighbour < nCols) ):
							
							continue
						
					# we add this neighbour to the list
					currentPEneighbours.append(arrayedPEs[iNeighbour,jNeighbour])
				
				# the list of neighbours of this PE is added to the list of lists of neighbours
				self._neighbours.append(currentPEneighbours)

class ConstantDegreeSimpleGraph(Topology):
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		super().__init__(nPEs,topologySpecificParameters)
		
		nNeighbours = math.ceil(topologySpecificParameters['number of neighbours as a percentage of the number of PEs']*nPEs)
		
		import networkx as nx
		
		# the Havel-Hakimi algorithm is used to obtain a simple graph with the requested degrees
		graph = nx.havel_hakimi_graph([nNeighbours]*nPEs)
		
		self._neighbours = [graph.neighbors(i) for i in range(nPEs)]
		
class FullyConnected(Topology):
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		super().__init__(nPEs,topologySpecificParameters)
		
		self._neighbours = [[j for j in range(nPEs) if j!=i] for i in range(nPEs)]

class LOSbased(Topology):
	
	def __init__(self,nPEs,topologySpecificParameters):
		
		super().__init__(nPEs,topologySpecificParameters)
		
		import operator
		
		# for the sake of clarity...
		PEsPositions = topologySpecificParameters['PEs positions']
		
		# a list of sets, each one meant to store the neighbours of the corresponding PE
		res = [set() for i in range(PEsPositions.shape[1])]
		
		for iPE in range(PEsPositions.shape[1]):
			
			# the difference vectors between the position of the current PE and the positions of ALL the PEs,...
			diff = PEsPositions - PEsPositions[:,iPE:iPE+1]
			
			# ... which allow to compute the angles
			angles = np.degrees(np.arctan2(diff[1,:],diff[0,:]))
			
			# ...and the distances
			distances = np.linalg.norm(diff,axis=0)
			
			# we need to use the "and" operator for all the comparisons except the one for the left, which requires the "or"
			for (low,up),op in zip(topologySpecificParameters['ranges of vision for right, up, left and down'],
						  [operator.and_,operator.and_,operator.or_,operator.and_]):
			
				# the index of the PEs that are within the range of angles specified
				iPEsWithinRange, = np.where(op(angles >= low,angles<=up) & (distances>0))
				
				# if any PE found within that "range of vision"
				if iPEsWithinRange.size:
				
					# the neighbour is chosen to be that which is closer to the current PE
					iNeighbour = iPEsWithinRange[np.argmin(distances[iPEsWithinRange])]
					
					# it is added to the list of neighbours of the current PE...
					res[iPE].add(iNeighbour)
					
					# ...and the current PE to the list of neighbours of the selected neighbour (if it wasn't yet)
					res[iNeighbour].add(iPE)
					
		self._neighbours = [list(s) for s in res]