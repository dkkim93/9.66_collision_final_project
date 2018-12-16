import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_human_data():
    # Set parameters
    IMG_PATH = "../data/imgs/"
    N_EPISODE = 89
    MAX_TIME = 5

    # Show image and receive user input
    collision_prob = []
    for i_episode in range(N_EPISODE):
        filename = IMG_PATH + "course_966_e_" + "%02d" % (i_episode) + "_t_" + "%02d" % (MAX_TIME) + "_label"

        with open(filename, "rb") as input_file:
            label = pickle.load(input_file)

        label = float(label)
        assert (label >= 0 and label <= 10) is True

        # Normalize label to prob
        label = label / 10.
        collision_prob.append(label)

    return collision_prob


def read_key_from_log(path, key, index, flip=False):
    with open(path) as f:
        content = f.read().splitlines()

    data = []
    for line in content:
        if key in line:
            target_data = line.split()[index]
            if target_data[0] == "[" or target_data[0] == "(":
                target_data = target_data[1:-1]

            if flip:
                data.append(-float(target_data))
            else:
                data.append(float(target_data))

    assert len(data) > 0

    return data


if __name__ == "__main__":
    # Read human data
    human_collision_prob = read_human_data()

    # Read NN data
    nn_collision_prob = read_key_from_log(
        path="result_nn.txt",
        key="collision prob",
        index=-1)

    # Read ensemble data
    ensemble_collision_prob = read_key_from_log(
        path="result_ensemble.txt",
        key="collision prob",
        index=-2)
    ensemble_collision_uncertainty = read_key_from_log(
        path="result_ensemble.txt",
        key="collision prob",
        index=-1)

    plt.plot(ensemble_collision_uncertainty)
    plt.show()

    # Convert to panda
    data = {}
    data["human"] = np.array(human_collision_prob)
    data["nn"] = np.array(nn_collision_prob)
    data["ensemble"] = np.array(ensemble_collision_prob)
    data_panda = pd.DataFrame(data=data)

    # Compute corr
    corr = data_panda["human"].corr(data_panda["nn"])
    print(corr)
    corr = data_panda["nn"].corr(data_panda["ensemble"])
    print(corr)

    # plt.figure()
    # plt.plot(human_collision_prob)
    # plt.plot(nn_collision_prob)
    # plt.show()

    # ensemble_collision_prob = read_key_from_log(
    #     path="result_ensemble.txt",
    #     key="collision prob",
    #     index=5)
