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
		delta1 = (pos[i]-pos[i-1])
		dirs.append(delta1.tolist())
	return dirs


## FOR DATA EXTRACTION

def radian_to_degree(sample): 
	return sample * 180 / np.pi

def get_dir_indices_weight(sample): 
	deg = radian_to_degree(sample)
	if deg <= 90.0 and deg >= 0.0:
		return 0, 1, (90-deg)/45., 1-(90-deg)/45.
	elif deg <= 45.0 and deg >= 0.0:
		return 1, 2, (45-deg)/45., 1-(45-deg)/45.
	elif deg >= 315.0 and deg <= 360.0:
		return 2, 3, (360-deg)/45., 1-(360-deg)/45.
	elif deg <= 315.0 and deg >= 270.0:
		return 3,4, (315-deg)/45., 1-(315-deg)/45.
	elif deg <= 270.0 and deg >= 225.0:
		return 4,5, (270-deg)/45., 1-(270-deg)/45.
	elif deg <= 225.0 and deg >= 180.0:
		return 5,6, (225-deg)/45., 1-(225-deg)/45.
	elif deg <= 180.0 and deg >= 135.0:
		return 6,7, (180-deg)/45., 1-(180-deg)/45.
	elif deg <= 135.0 and deg >= 90.0:
		return 7,0, (135-deg)/45., 1-(135-deg)/45.


def radian_to_dir(data, pos): 
	cur_pos = pos
	pos_res = []
	for d in data:
		i,j, wi, wj = get_dir_indices_weight(d) 
		cur_pos = DIRECTIONS[i]*wi + DIRECTIONS[j]*wj
		pos_res.append(cur_pos)
	return pos_res

## for experiments

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


def get_distance_model(dists): 
	model = hmm.GaussianHMM(n_components=5, covariance_type="full")
	model.fit(dists)
	return model


def get_euclidean_dists(pos1, pos2):
	return np.linalg.norm(pos1-pos2)

def get_dir_vector(dir_sample):
	# should be a float between 0, 8
	ind = round(dir_sample)
	if dir_sample - ind > 0: # round down 
		if ind <= 6: 
			return (dir_sample - ind)*DIRECTIONS[ind] + (1-(dir_sample - ind))*DIRECTIONS[ind+1]
		else: 
			return (dir_sample - ind)*DIRECTIONS[ind]
	else: 
		if ind >1:
			return (ind-dir_sample)*DIRECTIONS[ind] + (1-(ind-dir_sample))*DIRECTIONS[ind-1]
		else:
			return (ind-dir_sample)*DIRECTIONS[ind] 

def predict_position(dir_samples, pos):
	pos_res = []
	cur_pos = pos

	for i in dir_samples:
		dir_vec = get_dir_vector(i)
		# new position
		cur_pos += dir_vec
		pos_res.append(cur_pos)
	return pos_res

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
# Pipeline: get pos => derive directions => create dir model => sample directions => get new positions => derive euclidean dist => eucl. dist model 

### Saving for later evaluation ###

def save_model(model,filename="hmm_model.pkl"):
	joblib.dump(remodel, filename)

def get_model(filename="hmm_model.pkl"):
	return joblib.load(filename)
