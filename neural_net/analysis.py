import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sns.set_style("darkgrid")
sns.set_style("whitegrid")
# sns.set_style("ticks")
plt.rc('text', usetex=True)                                                                                     
plt.rc('font', family='serif')


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


def vis_human_data(human_collision_prob):
    plt.hist(
        human_collision_prob,
        alpha=0.7, rwidth=0.85,
        bins=10)
    plt.xlim([0, 10])
    plt.xlabel(r"\textbf{Rating ($0$: No collision, $10$: Collision)}", size=14)
    plt.ylabel(r"\textbf{Frequency}", size=14)
    plt.title(r"\textbf{Histogram of Human Data Collection (Total $89$ Samples)}", size=15)
    plt.show()


def vis_nn_data(nn_collision_prob):
    # nn_collision_prob = np.round(nn_collision_prob, decimals=1)
    plt.hist(
        nn_collision_prob,
        alpha=0.7, rwidth=0.85,
        bins=20)
    # plt.xlim([0, 1.])
    plt.xlabel(r"\textbf{Collision Probability}", size=14)
    plt.ylabel(r"\textbf{Frequency}", size=14)
    plt.title(r"\textbf{Histogram of NN Prediction (Total $89$ Samples)}", size=15)
    plt.show()


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

    # Hisogram on NN data
    vis_nn_data(nn_collision_prob)

    # Convert to panda
    data = {}
    data["human_prob"] = np.array(human_collision_prob)
    data["nn_prob"] = np.array(nn_collision_prob)
    data["ensemble_prob"] = np.array(ensemble_collision_prob)
    data_panda = pd.DataFrame(data=data)

    # Compute corr: human prob vs nn prob
    corr = data_panda["human_prob"].corr(data_panda["nn_prob"])
    print("Corr between human prob and nn prob:", corr)

    # # Plot human prob vs nn prob
    # plt.plot(data["human_prob"], label="Human Response")
    # plt.plot(data["nn_prob"], label="Neural Network Model")
    # legend = plt.legend(
    #     bbox_to_anchor=(0., 1.07, 1., .102),
    #     loc=3,
    #     ncol=2,
    #     mode="expand",
    #     borderaxespad=0.,
    #     prop={"size": 11})
    # plt.xlabel(r'\textbf{Data}', size=14)
    # plt.ylabel(r'\textbf{Collision Probability}', size=14)
    # plt.title(r'\textbf{Comparison between Human and Neural Network Data}', size=15)
    # plt.ylim([-0.03, 1.03])
    # plt.show()

    # Histogram
