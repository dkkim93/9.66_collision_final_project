# Hidden parameter: euclidean distance
import numpy as np
import pickle
from hmmlearn import hmm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random
import read_data
import utils

def analyze_fake_data(a1, a2): 
	# positions
	a1 = np.array([[0,0], [0,1], [0,2], [0,3], [0,4]])
	a2 = np.array([[4, 0], [3, 0], [2, 0], [1, 0], [0, 0]])

	# extract directions
	dir1 = utils.get_direction(a1)
	dir2 = utils.get_direction(a2)

	dir1 = utils.get_direction_vector(dir1)
	dir2 = utils.get_direction_vector(dir2)

	# models for directions
	model1 = utils.create_model(dir1+dir1)
	model2 = utils.create_model(dir2+dir2)

	X1, Z1 = model1.sample(10)
	a1_pos = utils.predict_position(X1, np.array(a1.astype('float64').tolist()[-1]))

	X2, Z2 = model2.sample(10)
	a2_pos = utils.predict_position(X2, np.array(a2.astype('float64').tolist()[-1]))

	# Create distance model
	dists = utils.get_euclidean_dists(a1_pos, a2_pos)
	dists = [[d] for d in dists]
	# print(dists, type(dists))
	model = utils.get_distance_model(dists)

	Xd,Zd = model.sample(100)
	utils.plot(Xd,Zd)


#### IMPORTANT CODE 
#### Change data indices here
# i = 0

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def analyze_sim_data(): 
	file_name = "../data/logs/obs_history_1"
	data= read_data.load_data(file_name)

	for i in range(20): 
		# positions, only want the first 5 samples
		a1 = data["agent1_pos"][i:i+1][:5]
		a2 = data["agent2_pos"][i:i+1][:5]

		# extract directions
		dir1 = data["agent1_heading"][i:i+1]
		dir2 = data["agent2_heading"][i:i+1]

		a1_pos = utils.radian_to_dir(dir1[0][:5], a1[0][-1])
		a2_pos = utils.radian_to_dir(dir2[0][:5], a2[0][-1])

		# Create distance model
		dists = utils.get_euclidean_dists(a1_pos, a2_pos)
		dists = [[d] for d in dists]
		print("Here are the distances between the two agents: {}".format(dists))

		model = utils.get_distance_model(dists)
		# log_prob = model._compute_log_likelihood(np.array([[0.0],[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]]))
		log_prob = model.score_samples(np.array([[0.0],[1.0],[2.0],[3.0],[4.0],[5.0],[6.0]]))
		# print("Transition matrix for experiment {}: \n {} ".format(i, normalized(np.exp(log_prob[1]))))
		nm = normalized(np.exp(log_prob[1]))

		with open('e_{}.txt'.format(i), 'wb') as f:
			pickle.dump(nm, f)


		# sampling distance model
		Xd,Zd = model.sample(100)
		utils.plot(Xd,Zd, heading=i)
		# plt.plot(Zd)

analyze_sim_data()
###
# Pipeline: get pos => derive directions => create dir model => sample directions => get new positions => derive euclidean dist => eucl. dist model 
