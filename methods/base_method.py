import tensorflow as tf

class BaseMethod(object):
	def __init__(self):

	def train(self, inputs, targets):
		raise NotImplementedError()

	def validation(self, inputs):
		raise NotImplementedError()