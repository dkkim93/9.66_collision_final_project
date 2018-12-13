
# import sys
# print (sys.version)
# print (‘hello world’)
# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random


## Experiments


def plot(X,Z):
	plt.plot(X)
	# plt.plot(Z)
	plt.ylim((0,7))
	plt.legend(loc='best')
	plt.show()

# get directions by time steps
def get_direction(pos):
	"""
	pos is a n x 2 vector
	"""
	dirs = []

	for i in range(1, len(pos)):
		print(pos[i], pos[i-1])

		delta1 = (pos[i]-pos[i-1])
		print(delta1)
		dirs.append(delta1.tolist())

	return dirs


# get direction vector
def get_direction_vector(directions):
	"""
	dir = [1,0]
	"""
	to_be_returned = []
	for i in range(len(directions)):
		res = np.zeros(8)
		order = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
		ind = np.where((order==tuple(directions[i])).all(axis=1)) 
		to_be_returned.append((ind[0]+ random.random()).tolist())
	return to_be_returned


# positions
a1 = np.array([[0,0], [0,1], [0,2], [0,3], [0,4]])
a2 = np.array([[4, 0], [3, 0], [2, 0], [1, 0], [0, 0]])

dir1 = get_direction(a1)
dir2 = get_direction(a2)

dir1 = get_direction_vector(dir1)
dir2 = get_direction_vector(dir2)

model_exp = hmm.GaussianHMM(n_components=8, covariance_type="full")
startprob = np.ones(model_exp.n_components) / model_exp.n_components
means = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
covars = .5 * np.tile(np.identity(2), (8, 1, 1))

model_exp.startprob_ = startprob
# model_exp.transmat_ = transmat
# model_exp.means_ = means
model_exp.covars_ = covars
print(dir1,dir2)


model_exp.fit(dir1+ dir1)


X, Z = model_exp.sample(500)
print(X,Z)
plot(X,Z)