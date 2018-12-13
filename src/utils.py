# Hidden parameter: euclidean distance
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import random

DIRECTIONS = np.array([[0,1],[1,1],[1,0], [1,-1], [0,-1], [-1,-1],[-1,0], [-1,1]])
np.random.seed(0)

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
	deg =sample
	# comes around
	if deg < 0:
		deg = abs(deg)

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
	else:
		raise Error("something is wrong with the degree")

def radian_to_dir(data, pos): 
	cur_pos = pos
	pos_res = np.array([pos])
	data = radian_to_degree(data)
	for d in data:
		i, j, wi, wj = get_dir_indices_weight(d) 
		cur_pos += DIRECTIONS[i]*wi + DIRECTIONS[j]*wj
		pos_res = np.vstack((pos_res, cur_pos))
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
	return np.linalg.norm(pos1-pos2,axis=1)

def get_dir_vector(dir_sample):
	# should be a float between 0, 8
	ind = round(round(float(dir_sample[0])))
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
	pos_res = np.array([pos])
	cur_pos = pos
	for i in dir_samples:
		dir_vec = get_dir_vector(i)
		# new position
		cur_pos += dir_vec
		pos_res = np.vstack((pos_res, cur_pos))
	return pos_res

def create_model(obs, n_components=8):
	model_exp = hmm.GaussianHMM(n_components=8, covariance_type="full")
	startprob = np.ones(model_exp.n_components) / model_exp.n_components
	covars = .5 * np.tile(np.identity(2), (8, 1, 1))
	model_exp.startprob_ = startprob
	model_exp.covars_ = covars
	# hmm library: needs samples >= n_components
	model_exp.fit(obs)
	return model_exp
