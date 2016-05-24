import abc
import numpy as np

import networkx


def sensors_PEs_mapping(PEs_sensors_mapping):
	n_sensors = sum([len(l) for l in PEs_sensors_mapping])

	res = [
		[i_PE for i_PE, PE_sensors in enumerate(PEs_sensors_mapping) if i_sensor in PE_sensors]
		for i_sensor in range(n_sensors)]

	# the list of lists is flattened before returning it
	return [e for l in res for e in l]


class SensorsPEsConnector(metaclass=abc.ABCMeta):
	
	def __init__(self, sensors_positions, PEs_positions=None, parameters=None):
		
		self._sensorsPositions = sensors_positions
		self._PEsPositions = PEs_positions
		self._parameters = parameters
		
		self._nSensors = self._sensorsPositions.shape[1]
	
	@abc.abstractmethod
	def get_connections(self, n_PEs):
		
		return


class EverySensorWithEveryPEConnector(SensorsPEsConnector):
	
	def get_connections(self, n_PEs):
		
		return [list(range(self._nSensors))] * n_PEs


class SensorOrientedConnector(SensorsPEsConnector):
	
	def get_connections(self, n_PEs):

		# each sensor is associated with "nPEsPerSensor" PEs
		sensors_degrees = [self._parameters['number of PEs per sensor']] * self._nSensors
		
		# how many (at least) sensors should be connected to every PE
		# (the expression between parenthesis is the overall number of connections)
		n_sensors_per_PE = (self._parameters['number of PEs per sensor']*self._nSensors) // n_PEs
		
		# each PE should be connected to the the number of sensors specified in the corresponding position of this list
		PEs_degrees = [n_sensors_per_PE] * n_PEs
		
		# if some connections remain, in order to satisfy that each sensor is connected to the given number of PEs...
		for iPE in range(self._parameters['number of PEs per sensor']*self._nSensors % n_PEs):
			
			# ...the "first" PEs get the extra needed connections
			PEs_degrees[iPE] +=  1
	
		# a bipartite graph with one set of nodes given by the sensors and other by the PEs
		graph = networkx.bipartite_havel_hakimi_graph(sensors_degrees, PEs_degrees)
		
		# we only "look" at the nodes from "self._n_sensors" onwards, since the previous ones correspond to the sensors
		return [sorted(graph.neighbors(iPE+self._nSensors)) for iPE in range(n_PEs)]


class ProximityBasedConnector(SensorsPEsConnector):
	
	def __init__(self, sensors_positions, PEs_positions, parameters=None):
		
		super().__init__(sensors_positions, PEs_positions, parameters)
		
	def get_connections(self, n_PEs):
		
		# the distance from each PE (whose position has been computed above) to each sensor [<PE>,<sensor>]
		distances = np.sqrt(
			(np.subtract(self._PEsPositions[:, :, np.newaxis], self._sensorsPositions[:, np.newaxis, :])**2).sum(axis=0))
		
		# for each sensor, the index of the PE which is closest to it
		i_closest_pe_to_sensors = distances.argmin(axis=0)
		
		return [list(np.where(i_closest_pe_to_sensors == iPE)[0]) for iPE in range(n_PEs)]


class ConstrainedProximityBasedConnector(ProximityBasedConnector):
	
	def get_connections(self, n_PEs):
		
		# a list with the sensors associated with each PE without the fixed number of sensors constraint
		unconstrained_PEs_sensors = super().get_connections(n_PEs)
		
		lengths = [len(s) for s in unconstrained_PEs_sensors]
		
		# the indexes of the PEs ordered by descending number of associated sensors
		i_PEs_decreasing_n_sensors = np.argsort(lengths)[::-1]
		
		# number of sensors that SHOULD be assigned to each PE
		n_sensors_per_PE = self._sensorsPositions.shape[1]//self._PEsPositions.shape[1]
		
		for i in range(n_PEs-1):
			
			# the index of the PE to be processed
			i_current_PE = i_PEs_decreasing_n_sensors[i]
			
			# number of sensor that should detach from this PE
			n_sensors_to_drop = lengths[i_current_PE] - n_sensors_per_PE
			
			if n_sensors_to_drop == 0:
				
				continue
			
			elif n_sensors_to_drop > 0:
			
				# the positions of the sensors associated with the current PE
				sensors_positions = self._sensorsPositions[:, unconstrained_PEs_sensors[i_current_PE]]
				
				# the positions of subsequente (not processed yet) PEs
				remaining_PEs_positions = self._PEsPositions[:, i_PEs_decreasing_n_sensors[i+1:]]
				
				# the (i,j) element is the distance from the i-th sensor to the j-th remaining PE
				distances = np.sqrt(((sensors_positions[:, :, np.newaxis] - remaining_PEs_positions[:, np.newaxis, :])**2).sum(axis=0))
				
				for _ in range(n_sensors_to_drop):
					
					# the index of the sensor that is sent away and that of the PE that is going to attach to
					i_local_sensor, i_hosting_PE = np.unravel_index(distances.argmin(),distances.shape)
			
					unconstrained_PEs_sensors[i_PEs_decreasing_n_sensors[i+1+i_hosting_PE]].append(unconstrained_PEs_sensors[i_current_PE][i_local_sensor])
					del unconstrained_PEs_sensors[i_current_PE][i_local_sensor]
					
					lengths[i_current_PE] -= 1
					lengths[i_PEs_decreasing_n_sensors[i+1+i_hosting_PE]] += 1
			
			else:

				raise Exception('not implemented!!')
		
		return unconstrained_PEs_sensors
