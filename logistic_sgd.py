import theano
import theano.tensor as T
import numpy
import os

import gzip
import cPickle


class LogisticRegression(object):
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(
					value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
					name='W')
		self.b = theano.shared(
					numpy.zeros(n_out, dtype=theano.config.floatX),
					name='b')

		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
		''' likelihood = the probability of classifying right '''
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y shold have the same shape as self.y_pred',
				('y', y.type, 'y_pred', self.y_pred.type))

		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

def load_data(dataset):
	if not os.path.isfile(dataset):
		raise IOError('file: ' + dataset + ' not found')
	
	f = gzip.open(dataset)
	train_set, valid_set, test_set = cPickle.load(f)

	def share_data(data_xy):
		data_x, data_y = data_xy
		shared_x = theano.shared(
				numpy.asarray(data_x, dtype=theano.config.floatX),
				borrow=True)
		shared_y = theano.shared(numpy.asarray(
				data_y, dtype=theano.config.floatX),
				borrow=True)

		return shared_x, T.cast(shared_y, 'int32')

	train_set_x, train_set_y = share_data(train_set)
	valid_set_x, valid_set_y = share_data(valid_set)
	test_set_x, test_set_y = share_data(test_set)

	rval = [(train_set_x, train_set_y),
			(valid_set_x, valid_set_y),
			(test_set_x, test_set_y)] 

	return rval

def train_model(learning_rate=0.1, dataset='mnist.pkl.gz',
				n_in=28*28, n_out=10, batch_size=600, n_epochs=1000):

	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')
	
	print 'loading data ...'
	data = load_data(dataset)
	train_set_x, train_set_y = data[0]
	valid_set_x, valid_set_y = data[1]
	test_set_x, test_set_y = data[2]

	# calc number of batches
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]/batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0]/batch_size


	print 'building model ...'
	logreg = LogisticRegression(x, n_in, n_out)
	# build test model
	test_model = theano.function(
		inputs=[index],
		outputs=logreg.errors(y),
		givens={
			x: test_set_x[index*batch_size: (index+1)*batch_size],
			y: test_set_y[index*batch_size: (index+1)*batch_size]
		}
	)

	# build valid model
	valid_model = theano.function(
		inputs=[index],
		outputs=logreg.errors(y),
		givens={
			x: valid_set_x[index*batch_size: (index+1)*batch_size],
			y: valid_set_y[index*batch_size: (index+1)*batch_size]
		}
	)

	# build train model
	cost = logreg.negative_log_likelihood(y)
	g_W, g_b = T.grad(logreg.negative_log_likelihood(y), logreg.params)
	updates=[(logreg.W, logreg.W-learning_rate*g_W),
			 (logreg.b, logreg.b-learning_rate*g_b)]

	train_model = theano.function(
		inputs=[index],
		outputs=cost,
		updates=updates,
		givens={
			x: train_set_x[index*batch_size: (index+1)*batch_size],
			y: train_set_y[index*batch_size: (index+1)*batch_size]
		}
	)
	
	print 'training model ...'

