import pandas as pd
import numpy as np
from preprocess import encode_and_set_decoder

def write_predictions(path, predictions_np, losses_np=None):
    # Create a copy of the DataFrame to avoid modifying the original
    df = pd.read_csv(path)
    _ , decoder = encode_and_set_decoder(df)



    df_length = len(df)



    # Flatten predictions to a 1D array
    predictions_flat = predictions_np.flatten()
    losses_flat = losses_np.flatten()

    # Unscale the predictions to match the original encrypted_lengths scale
    unscaled_lengths = predictions_flat.reshape(-1, 1)

    unscaled_lengths_flat = unscaled_lengths.flatten()

    decoded_array = np.array([decoder[val] for val in unscaled_lengths_flat])
    num_predictions = min(len(decoded_array), df_length)

    # Assign the unscaled lengths to the corresponding entries
    df['predicted_next_length'] = 0  # Initialize the column with zeros or any default value
    df.iloc[:num_predictions, df.columns.get_loc('predicted_next_length')] = pd.Series(decoded_array[:num_predictions])

    df['loss'] = 0
    df.iloc[:num_predictions, df.columns.get_loc('loss')] = pd.Series(
        losses_flat[:num_predictions])
    df.to_csv(path, index=False)