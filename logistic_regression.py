import numpy.random as rng
import theano.tensor as T
from theano import shared
from theano import function


def main(steps=1000):
	N = 400
	feats = 666
	
	# generate data
	D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

	# construct expression
	x = T.matrix('x')
	y = T.vector('y')
	w = shared(rng.randn(feats), name='w')
	b = shared(0.0, name='b')

	p = 1/(1+T.exp(-T.dot(x,w)-b))
	prediction = p > 0.5

	# the training error is defined as cross entropy
	err = -y*T.log(p) - (1-y)*T.log(1-p)
#	cost = err.mean() + 0.01*(w**2).sum()
	cost = err.mean()

	# using gradient descent to approximate w and b
	gw, gb = T.grad(cost, [w, b])

	# compile
	train = function(
				inputs=[x, y],
				outputs=[prediction, err],
				updates=[(w,w-0.1*gw), (b,b-0.1*gb)])

	predict = function([x], prediction)

	for i in range(steps):
		train(D[0], D[1])

	return D[1], predict(D[0])
