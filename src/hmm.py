# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random

np.random.seed(0)
class Agent:
	def __init__(self, pos, num_dir=8):
		self.pos = pos
		self.dir_vectors = np.random.uniform(8)

class GridWorld:
	def __init__(self, w=10, h=10):
		self.width = w
		self.heigth = h
		self.agents = []

	def display(self):
		for agent in self.agents:
			plt.plot(agent.pos[0],agent.pos[1], 'bo')

		plt.axis([0, self.width, 0, self.height])
		plt.show()


def infer(hmm_model, X):
	model.fit(X)
	return model.predict(X)

def sample_model(model, sample_size=1):
	return model.sample(sample_size)

def update_probs(model, obs):
	# TODO(rewang): merge with transition updates of the hmm model (basically the same idea)
	# based on previous and current state, update probability vectors under
	# agent.dir_vector
	raise NotImplementedError

def init_start_prob(model):
	model.startprob_ = np.random.uniform(model.n_components)
	return model


def create_multisequence_model(P1, P2):
	"""
	should be direction observations

	positions ==get==> directions ==infer==> next step ==get==> new position
	"""
	P = np.concatenate([P1, P2])
	lengths = [len(P1), len(P2)]
	return hmm.GaussianHMM(n_components=3).fit(P, lengths)

### PLOTTING DATA

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
	find indices
	"""
	to_be_returned = []
	for i in range(len(directions)):
		res = np.zeros(8)
		order = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
		ind = np.where((order==tuple(directions[i])).all(axis=1)) 
		to_be_returned.append((ind[0]+ random.random()).tolist())
	return to_be_returned

def analyze(a1, a2): 
	# positions
	a1 = np.array([[0,0], [0,1], [0,2], [0,3], [0,4]])
	a2 = np.array([[4, 0], [3, 0], [2, 0], [1, 0], [0, 0]])

	dir1 = get_direction(a1)
	dir2 = get_direction(a2)

	dir1 = get_direction_vector(dir1)
	dir2 = get_direction_vector(dir2)

	model_exp = hmm.GaussianHMM(n_components=8, covariance_type="full")
	startprob = np.ones(model_exp.n_components) / model_exp.n_components
	covars = .5 * np.tile(np.identity(2), (8, 1, 1))
	model_exp.startprob_ = startprob
	model_exp.covars_ = covars

	# hmm library: needs samples >= n_components
	model_exp.fit(dir1+dir1)
	X, Z = sample_model(model_exp, 500)
	plot(X,Z)

###


### Saving for later evaluation ###

def save_model(model,filename="hmm_model.pkl"):
	joblib.dump(remodel, filename)

def get_model(filename="hmm_model.pkl"):
	return joblib.load(filename)
