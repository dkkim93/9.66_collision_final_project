import numpy as np
import pickle # I think my pickle file requires python3

  
def load_data(file_name):
  # Load data
  pickle_name = file_name
  pickle_file = open(pickle_name,'rb')
  data = pickle.load(pickle_file)
  pickle_file.close()

  # Unnecessary. Just to show whats in the data.
  agent1_pos = data["agent1_pos"]
  agent1_heading = data["agent1_heading"]
  agent2_pos = data["agent2_pos"]
  agent2_heading = data["agent2_heading"]
  is_collided = data["is_collided"]
  full_obs = data["full_obs"]

  return data

if __name__ == '__main__':
  file_name = "logs/obs_history_1"
  load_data(file_name)
  print(data)