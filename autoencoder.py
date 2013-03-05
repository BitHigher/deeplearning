import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import os
import time
from logistic_sgd import load_data
from utils import tile_raster_images
import PIL.Image

class AutoEncoder(object):
	def __init__(self, numpy_rng, theano_rng=None, input=None, 
					n_visible=784, n_hidden=500,
					W=None, bhid=None, bvis=None):
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if not theano_rng:
			theano_rng = RandomStreams(rng.randint(2 ** 30))
	
		if not W:
			initial_W = numpy.asarray(numpy_rng.uniform(
				low = -4*numpy.sqrt(6./(n_hidden+n_visible)),
				high = 4*numpy.sqrt(6./(n_hidden+n_visible)),
				size=(n_visible, n_hidden)),
				dtype=theano.config.floatX)
			W = theano.shared(value=initial_W, name='W')
		
		if not bvis:
			bvis = theano.shared(value=numpy.zeros(n_visible, 
										dtype=theano.config.floatX),
									name='bvis')

		if not bhid:
			bhid = theano.shared(value=numpy.zeros(n_hidden,
										dtype=theano.config.floatX),
									name='bhid')

		self.W = W
		self.b = bhid
		self.b_prime = bvis
		self.W_prime = self.W.T
		self.theano_rng = theano_rng

		if input == None:
			self.x = T.dmatrix(name='input')
		else:
			self.x = input

		self.params = [self.W, self.b, self.b_prime]

	def get_corrupted_input(self, input, corruption_level):
		return self.theano_rng.binomial(
						size=input.shape, n=1, 
						p=1-corruption_level) * input
	
	def get_hidden_values(self, input):
		return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

	def get_reconstructed_input(self, hidden):
		return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)
		
	def get_cost_updates(self, corruption_level, learning_rate):
		''' computes the cost and the updates for one training '''
		tilde_x = self.get_corrupted_input(self.x, corruption_level)
		y = self.get_hidden_values(tilde_x)
		z = self.get_reconstructed_input(y)

		L = -T.sum(self.x * T.log(z) + (1-self.x) * T.log(1-z), axis=1)

		cost = T.mean(L)

		gparams = T.grad(cost, self.params)

		updates = []
		for param, gparam in zip(self.params, gparams):
			updates.append([param, param-learning_rate*gparam])

		return (cost, updates)

def test_da(learning_rate=0.1, training_epochs=15,
			dataset='mnist.pkl.gz', batch_size=20,
			output_folder='da_plots'):
	if not os.path.isdir(output_folder):
		os.makedirs(output_folder)

	print 'loading data ... '
	datasets = load_data(dataset)
	train_set_x , train_set_y = datasets[0]
	n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batch_size

	print 'building model ... '
	index = T.lscalar()
	x = T.matrix('x')

	rng = numpy.random.RandomState(123)
	theano_rng = RandomStreams(rng.randint(2 ** 30))

	da = AutoEncoder(numpy_rng=rng, theano_rng=theano_rng, input=x, 
						n_visible=28*28, n_hidden=500)

	cost, updates = da.get_cost_updates(
				corruption_level = 0.3, 
				learning_rate = learning_rate)
	
	train_model = theano.function(
					[index], cost,
					updates = updates,
					givens = {
						x: train_set_x[index*batch_size: (index+1)*batch_size]
					})

	print 'training model ... '
	start_time = time.clock()

	for epoch in xrange(training_epochs):
		c = []
		for batch_index in xrange(n_train_batches):
			c.append(train_model(batch_index))

		print 'Training epoch %d, cost %f' % (epoch, numpy.mean(c))

	end_time = time.clock()
	training_time = (end_time - start_time)
	print 'Training took %f minutes' % (training_time/60.)

	image = PIL.Image.fromarray(tile_raster_images(
		X=da.W.get_value(borrow=True).T,
		img_shape=(28,28), tile_shape=(10, 10),
		tile_spacing=(1,1)))

	os.chdir(output_folder)
	image.save('filters_corrupution_30.png')
	os.chdir('../')

if __name__ == '__main__':
	test_da()
