import pickle
import numpy as np


def load_data(file_name):
    pickle_name = file_name
    pickle_file = open(pickle_name, 'rb')
    data = pickle.load(pickle_file)
    pickle_file.close()
    
    # # Unnecessary. Just to show whats in the data.
    # agent1_pos = data["agent1_pos"]
    # agent1_heading = data["agent1_heading"]
    # agent2_pos = data["agent2_pos"]
    # agent2_heading = data["agent2_heading"]
    # is_collided = data["is_collided"]
    # full_obs = data["full_obs"]
    
    return data


def process_data(data):
    data_size = data["agent1_pos"].shape[0]
    n_episode = 5

    input_data, label_data = [], []
    for i_data in range(data_size):
        # Process input data
        data_point_agent1 = data["agent1_pos"][i_data, 0:n_episode, :].flatten()
        data_point_agent2 = data["agent2_pos"][i_data, 0:n_episode, :].flatten()

        data_point = np.concatenate([data_point_agent1, data_point_agent2])
        if np.sum(data_point) == 0:
            pass
        else:
            input_data.append(data_point)

            # Process label
            label = np.max(data["is_collided"][i_data])
            if label == 0.:
                label_data.append([1., 0.])  # No collision one hot
            elif label == 1.:
                label_data.append([0., 1.])  # Collision one hot
            else:
                raise ValueError()

    input_data = np.squeeze(np.array([input_data]), axis=0)
    label_data = np.squeeze(np.array([label_data]), axis=0)

    print("input_data.shape:", input_data.shape)
    print("label_data.shape:", label_data.shape)

    return input_data, label_data
