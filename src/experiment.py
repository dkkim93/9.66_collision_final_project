# Hidden parameter: euclidean distance
import numpy as np
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
	plot(Xd,Zd)


def analyze_sim_data(): 
	# As a reference: Information inside `data`
	# agent1_pos = data["agent1_pos"]
	# agent1_heading = data["agent1_heading"] 
	# agent2_pos = data["agent2_pos"]
	# agent2_heading = data["agent2_heading"]
	# is_collided = data["is_collided"]
	# full_obs = data["full_obs"] 
	file_name = "../data/logs/obs_history_1"
	data= load_data(file_name)


	# positions
	a1 = data["agent1_pos"][:1]
	a2 = data["agent2_pos"][:1]

	# extract directions
	dir1 = data["agent1_heading"][:1]
	dir2 = data["agent2_heading"][:1]

	dir1 = get_direction_vector(dir1)
	dir2 = get_direction_vector(dir2)

	# models for directions
	model1 = create_model(dir1+dir1)
	model2 = create_model(dir2+dir2)

	X1, Z1 = model1.sample(10)
	a1_pos = predict_position(X1, np.array(a1.astype('float64').tolist()[-1]))

	X2, Z2 = model2.sample(10)
	a2_pos = predict_position(X2, np.array(a2.astype('float64').tolist()[-1]))

	# Create distance model
	dists = get_euclidean_dists(a1_pos, a2_pos)
	dists = [[d] for d in dists]
	# print(dists, type(dists))
	model = get_distance_model(dists)

	Xd,Zd = model.sample(100)
	plot(Xd,Zd)

	print(data["agent2_pos"][:1])
	print(data["agent2_heading"][:1])

###
# Pipeline: get pos => derive directions => create dir model => sample directions => get new positions => derive euclidean dist => eucl. dist model 
