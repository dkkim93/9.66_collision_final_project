# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt


np.random.seed(0)

class Agent:
	def __init__(self, pos): 
		self.pos = pos

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
	# TODO(rewang)
	raise NotImplementedError

def init_start_prob(model):
	model.startprob_ = np.random.uniform(model.n_components)
	return model

def init_transition_prob(model, agent): 
	raise Not NotImplementedError
	model.transmat_ = np.array([[0.7, 0.2, 0.1],
	                            [0.3, 0.5, 0.2],
	                            [0.3, 0.3, 0.4]])
	return model


def create_multisequence_model(P1, P2):
	"""
	P1 and P2 are 2x5 np arrays
	"""
	P = np.concatenate([P1, P2])
	lengths = [len(P1), len(P2)]
	return hmm.GaussianHMM(n_components=3).fit(P, lengths)  

### Saving for later evaluation ### 

def save_model(model,filename="hmm_model.pkl"): 
	joblib.dump(remodel, filename)


def get_model(filename="hmm_model.pkl"): 
	return joblib.load(filename)  
