# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random

DIRECTIONS = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
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

### Saving for later evaluation ###

def save_model(model,filename="hmm_model.pkl"):
	joblib.dump(remodel, filename)

def get_model(filename="hmm_model.pkl"):
	return joblib.load(filename)
