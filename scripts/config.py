from path import Path

device = 'cuda:0'
p = 0.61
n = 10

untrained_model_save_file = Path(f"./saved_models/untrained_model_{n}.pt")
trained_model_save_file = Path(f"./saved_models/trained_model_{n}.pt")
validation_data_save_file = Path(f"./validation_data/validation_data_{n}.npy")