import matplotlib
import matplotlib.pyplot as plt

class Painter:
	def __init__(self,sensorsPositions):
		self._figure = plt.figure('room')
		self._ax = plt.axes()
		self._sensorsPositions = sensorsPositions
		
		# so that the program doesn't wait for the open windows to be closed in order to continue
		plt.ion()
		
	def go(self):
		plt.plot(self._sensorsPositions[:,0], self._sensorsPositions[:,1], '+')
		self._ax.set_aspect('equal', 'datalim')
		plt.show()
		
class PainterDecorator(Painter):
	def __init__(self,decorated):
		# let the superclass do its stuff
		super().__init__(decorated._sensorsPositions)
		
		# we keep a reference to the object being decorated
		self._decorated = decorated
		
class WithBorder(PainterDecorator):
	def __init__(self,decorated,roomBottomLeftCorner,roomTopRightCorner):
		super().__init__(decorated)
		self._roomBottomLeftCorner = roomBottomLeftCorner
		self._roomTopRightCorner = roomTopRightCorner
		self._roomDiagonalVector = self._roomTopRightCorner - self._roomBottomLeftCorner
		
	def go(self):
		# let the superclass do its stuff
		super().go()
		
		# we define a rectangular patch...
		roomEdge = matplotlib.patches.Rectangle((self._roomBottomLeftCorner[0],self._roomBottomLeftCorner[1]), self._roomDiagonalVector[0], self._roomDiagonalVector[1], fill=False, color='blue')
		
		# ...and added to the axes
		self._ax.add_patch(roomEdge)
		
		