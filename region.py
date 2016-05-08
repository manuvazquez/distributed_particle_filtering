import numpy as np


class Rectangle:

	def __init__(self, bottom_left_corner, top_right_corner):

		self._bottom_left_corner = np.array(bottom_left_corner).reshape(2, -1)
		self._top_right_corner = np.array(top_right_corner).reshape(2,-1)

		# top right, top left, bottom left, and bottom right corners stored as column vectors
		self.tr_corner = np.array(top_right_corner).reshape(2, -1)
		self.tl_corner = np.array([bottom_left_corner[0], top_right_corner[1]]).reshape(2, -1)
		self.bl_corner = np.array(bottom_left_corner).reshape(2, -1)
		self.br_corner = np.array([top_right_corner[0], bottom_left_corner[1]]).reshape(2, -1)

		self.x_range = (bottom_left_corner[0], top_right_corner[0])
		self.y_range = (bottom_left_corner[1], top_right_corner[1])

		self.tr_tl_bl_br_corners = np.hstack((self.tr_corner, self.tl_corner, self.bl_corner, self.br_corner))

	def belong(self, positions):

		return np.logical_and(
			(positions > self._bottom_left_corner).all(axis=0),
			(positions < self._top_right_corner).all(axis=0)
		)