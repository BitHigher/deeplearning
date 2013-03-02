import theano.tensor as T
import numpy
import time
import os

from logistic_sgd import *

class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
		self.input = input
		W_values = numpy.asarray(rng.uniform(
							low = -numpy.sqrt(6./(n_in+n_out)),
							high = numpy.sqrt(6./(n_in+n_out)),
							size = (n_in, n_out)),
						dtype=theano.config.floatX)
		if activation == T.nnet.sigmoid:
			W_values *= 4
		self.W = theano.shared(value=W_values, name='W')
		
		b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
		self.b = theano.shared(value=b_values, name='b')

		self.output = activation(T.dot(input, self.W) + self.b)

		self.params = [self.W, self.b]

class MLP(object):
	def __init__(self, rng, input, n_in, n_hidden, n_out):
		# hidden layer
		self.hiddenLayer = HiddenLayer(rng, input, n_in, n_hidden)

		# output layer
		self.logRegressionLayer = LogisticRegression(
									input = self.hiddenLayer.output,
									n_in = n_hidden,
									n_out = n_out)

		# L1 norm
		self.L1 = abs(self.hiddenLayer.W).sum() \
				+ abs(self.logRegressionLayer.W).sum()

		# L2 norm
		self.L2_sqr = (self.hiddenLayer.W ** 2).sum() \
					+ (self.logRegressionLayer.W ** 2).sum()

		# negative log likelihood
		self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood

		# errors
		self.errors = self.logRegressionLayer.errors

		#params
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
				dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
	
	print 'loading data ... '
	datasets = load_data(dataset)
	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	print 'building model ... '
	index = T.lscalar()
	x = T.matrix('x')
	y = T.ivector('y')

	rng = numpy.random.RandomState(1234)
	
	classifier = MLP(input=x, rng=rng, n_in=28*28, n_hidden=n_hidden, n_out=10)
	
	# define the cost function
	cost = classifier.negative_log_likelihood(y) \
		 + L1_reg * classifier.L1 \
		 + L2_reg * classifier.L2_sqr

	# build test model
	test_model = theano.function(
		inputs = [index],
		outputs = classifier.errors(y),
		givens = {
			x: test_set_x[index*batch_size: (index+1)*batch_size],
			y: test_set_y[index*batch_size: (index+1)*batch_size]
		}
	)

	# build valid model
	valid_model = theano.function(
		inputs = [index],
		outputs = classifier.errors(y), 
		givens = {
			x: valid_set_x[index*batch_size: (index+1)*batch_size],
			y: valid_set_y[index*batch_size: (index+1)*batch_size]
		}
	)


	# build train model
	gparams = []
	for param in classifier.params:
		gparam = T.grad(cost, param)
		gparams.append(gparam)

	updates = []
	for param, gparam in zip(classifier.params, gparams):
		updates.append((param, param - learning_rate*gparam))

	train_model = theano.function(
		inputs = [index],
		outputs = cost,
		updates = updates,
		givens = {
			x: train_set_x[index*batch_size: (index+1)*batch_size],
			y: train_set_y[index*batch_size: (index+1)*batch_size]
		}
	)

	print 'training model ... '
	# early-stopping parameters ?
	patience = 10000
	patience_increase = 2

	improvement_threshold = 0.995
	valid_frequency = min(n_train_batches, patience/3)

	best_params = None
	best_validation_loss = numpy.inf
	best_iter = 0
	best_score = 0.
	start_time = time.clock()

	epoch = 0
	done_looping = False


	while(epoch < n_epochs) and (not done_looping):
		epoch += 1
		for minibatch_index in xrange(n_train_batches):
			minibatch_avg_cost = train_model(minibatch_index)
			# iteration number
			iter = epoch * n_train_batches + minibatch_index

			if (iter+1) % valid_frequency == 0:
				validation_losses = [valid_model(i) for i in 
										xrange(n_valid_batches)]
				this_validation_loss = numpy.mean(validation_losses)

				print('epoch %i, minibatch %i/%i, validation error %f %%' %
					(epoch, minibatch_index+1, n_train_batches,
					this_validation_loss*100.))

				if this_validation_loss < best_validation_loss:
					# improve patience if loss imporvement is good enough
					if this_validation_loss < \
							best_validation_loss * improvement_threshold:
						patience = max(patience, iter * patience_increase)

					best_validation_loss = this_validation_loss
					best_iter = iter

					# test it on the test set
					test_losses = [test_model(i) for i in
									xrange(n_test_batches)]
					test_score = numpy.mean(test_losses)
					print (('	epoch %i, minibatch %i/%i, test error of '
							'best model %f %%') % 
							(epoch, minibatch_index+1, n_train_batches,
							test_score * 100.))
			if patience <= iter:
				done_looping = True
				break

	end_time = time.clock()
	print(('Optimization complete. Best validation score of %f %% '
			'obtained at iteration %i, with test performance % %%') %
		  (best_validation_loss*100., best_iter, test_score * 100.))

	print >> sys.stderr, ('The code for file ' + 
							os.path.split(__file__)[1] + 
						  ' ran for %.2fm' % ((end_time-start_time) / 60.))

if __name__ == '__main__':
	test_mlp()
