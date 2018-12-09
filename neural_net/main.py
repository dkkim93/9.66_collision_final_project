from normal_nn import NormalNN
# from ensemble_nn import EnsembleNN
from utils import *


if __name__ == "__main__":
    # Process data
    data = load_data("../data/logs/obs_history_1")
    input_data, label_data = process_data(data)

    # Set up model
    input_dim = input_data.shape[1]
    output_dim = label_data.shape[1]
    model = NormalNN(
        input_dim=input_dim,
        output_dim=output_dim)

    # Train model
    model.train(input_data, label_data)
