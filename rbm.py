import numpy
import theano
import theano.tensor as T

from theano.tensor.shared_randomstreams import RandomStreams


class RBM(object):
	"""Restricted Boltzmann Machine """
	
	def __init__(self, input=None, n_visible=784, 
				n_hidden=500, W=None, hbias=None,
				vbias=None, numpy_rng=None,
				theano_rng=None):
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			numpy_rng = numpy.random.RandomState(1234)

		if theano_rng is None:
			theano_rng = RandomStreams(numpy_rng.randint(2**30))

		if W is None:
			initial_W = numpy.asarray(numpy.random.uniform(
				low = -4*numpy.sqrt(6.0/(n_hidden+n_visible)),
				high = -4*numpy.sqrt(6.0/(n_hidden+n_visible)),
				size=(n_visible, n_hidden)),
				dtype=theano.config.floatX)
			W = theano.shared(value=initial_W, name='W')



		if hbias is None:
			hbias = theano.shared(value=numpy.zeros(
						n_hidden, dtype=theano.config.floatX), name='hbias')

		if vbias is None:
			vbias = theano.shared(value=numpy.zeros(
						n_visible, dtype=theano.config.floatX), name='vbias')

		self.input = input if input else T.dmatrix('input')

		self.W = W
		self.hbias = hbias
		self.vbias = vbias
		self.theano_rng = theano_rng

		self.params = [self.W, self.hbias, self.vbias]


	def propup(self, vis):
		''' propagates the visible units activation upwards to the hidden units'''
		pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_h_given_v(self, v0_sample):
		''' infers state of hidden units given visible units '''
		pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
		h1_sample = self.theano_rng.binomial(
					size=h1_mean.shape, n=1, p=h1_mean,
					dtype=theano.config.floatX)

		return [pre_sigmoid_h1, h1_mean, h1_sample]

	def propdown(self, hid):
		''' propagates the hidden units activation downwards to the visible units '''
		pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
		return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

	def sample_v_given_h(self, h0_sample):
		''' interfs state of visible units given hidden units '''
		pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
		v1_sample = self.theano_rng.binomial(
					size = v1_mean.shape, n=1, p=v1_mean,
					dtype=theano.config.floatX)

		return [pre_sigmoid_v1, v1_mean, v1_sample]

	def gibbs_hvh(self, h0_sample):
		''' one step of Gibbs sampling, starting from the hidden state '''
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample):
		''' one step of Gibbs sampling, starting from the visible state '''
		pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

	def free_energy(self, v_sample):
		wx_b = T.dot(v_sample, self.W) + self.hbias
		vbias_term = T.dot(v_sample, self.vbias)
		hidden_term = T.sum(T.log(1+T.exp(wx_b)), axis=1)
		return -hidden_term - vbias_term

	def get_cost_updates(self, lr=0.1, persistent=None, k=1):
		''' implements one step of CD-k or PCD-k 
		    :param lr: learning rate used to train RBM
			:param k: number of Gibbs steps to do in CD-k/PCD-k'''
		
		# compute positive phase
		pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

		# decide how to initialize persistent chain:
		# for CD, we use the newly generate hidden sample
		# for PCD, we initialize from the old state of the chain
		if persistent is None:
			chain_start = ph_sample
		else:
			chain_start = persistent

		# perform actual negative phase
		[pre_sigmoid_nvs, nv_means, nv_samples, 
		pre_sigmoid_nhs, nh_means, nh_samples] , updates = theano.scan(
			self.gibbs_hvh,
			outputs_info = [None, None, None, None, None, chain_start],
			n_steps = k)

		chain_end = nv_samples[-1]
		cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
		# must not compute the gradient through the gibbs sampling
		gparams = T.grad(cost, self.params, consider_constant=[chain_end])

		for gparam, param in zip(gparams, self.params):
			updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

		if persistent:
			updates[persistent] = nh_samples[-1]
			monitoring_cost = self.get_pseudo_likelihood_cost(updates)
		else:
			monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])

		return monitoring_cost, updates


