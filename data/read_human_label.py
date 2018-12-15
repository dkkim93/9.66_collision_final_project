import pickle


if __name__ == "__main__":
    # Set parameters
    IMG_PATH = "imgs/"
    N_EPISODE = 89
    MAX_TIME = 5

    # Show image and receive user input
    for i_episode in range(N_EPISODE):
        filename = IMG_PATH + "course_966_e_" + "%02d" % (i_episode) + "_t_" + "%02d" % (MAX_TIME) + "_label"

        with open(filename, "rb") as input_file:
            label = pickle.load(input_file)

        label = float(label)
        assert (label >= 0 and label <= 10) is True

        # Normalize label to prob
        label = label / 10.

        print("Data {}: Probability of collision is {}".format(i_episode, label))
