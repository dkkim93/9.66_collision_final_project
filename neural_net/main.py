from normal_nn import NormalNN
from ensemble_nn import EnsembleNN
from utils import *

MODEL = "normal"  # Either "normal" or "ensemble"


if __name__ == "__main__":
    # Process data
    data = load_data("../data/logs/10000e/obs_history")
    input_data, label_data = process_data(data)

    # Set up model
    input_dim = input_data.shape[1]
    output_dim = label_data.shape[1]
    if MODEL == "normal":
        model = NormalNN(
            input_dim=input_dim,
            output_dim=output_dim)
    elif MODEL == "ensemble":
        model = EnsembleNN(
            input_dim=input_dim,
            output_dim=output_dim,
            ensemble_size=5)
    else:
        raise ValueError()

    # Train model
    model.train(input_data, label_data)

    # Test model
    test_data = load_data("../data/logs/obs_history_1")
    input_data, label_data = process_data(test_data)
    model.prediction(input_data)
