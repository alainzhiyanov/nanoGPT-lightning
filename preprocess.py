from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd


class TimeSeriesDataset(Dataset):
    """
        Custom PyTorch Dataset for time series data.

        Args:
            interpacket_time_normalized (torch.Tensor): Normalized interpacket time feature.
            packet_lengths (torch.Tensor): Packet length feature.
            targets (torch.Tensor): Target values for prediction.
        """
    def __init__(self, interpacket_time_normalized, packet_lengths, targets):
        self.interpacket_time_normalized = interpacket_time_normalized
        self.packet_lengths = packet_lengths
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        #return self.interpacket_time_normalized[idx], self.packet_lengths[idx], self.targets[idx],
        return self.packet_lengths[idx], self.targets[idx],




def preprocess_time(df):
    """
    Preprocess the data for training by reading, normalizing, and creating sequences.

    Args:
        path (str): Path to the CSV file containing the data.
        block_size (int): Sequence length for the time series.

    Returns:
        TimeSeriesDataset: Dataset ready for PyTorch training.
    """
    scaler = StandardScaler()
    df['interpacket_time_normalized'] = scaler.fit_transform(df[['interpacket_time']])
    return df, scaler


def add_next_length_feature(df):
    """
    Add a feature for the next packet length by shifting the 'encrypted_lengths' column upward.

    Args:
        df (pd.DataFrame): DataFrame containing the 'encrypted_lengths' column.

    Returns:
        pd.DataFrame: Updated DataFrame with the 'next_length' column added.
    """
    # Shift the column upwards to align the next entry with the current row
    df['next_length'] = df['mapped_lengths'].shift(-1)
    df = df.dropna()

    return df

def create_sequences(data, sequence_length, device):
    """
    Create sequences for time series prediction suitable for PyTorch. The entire trace is partitioned into sequence_length sequences.

    Args:
        data (pd.DataFrame): Input DataFrame containing the features.
        sequence_length (int): Length of the input sequences.
        device (torch.device): The device (CPU or GPU) to store the tensors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            Tensors for interpacket time, packet lengths, and targets.
    """
    interpacket_time_normalized = []
    packet_lengths = []
    targets = []

    for i in range(0, len(data) - sequence_length, sequence_length):
        seq = data.iloc[i:i + sequence_length]  # Get the current sequence
        targets.append(seq['next_length'].values)
        interpacket_time_normalized.append(seq['interpacket_time_normalized'].values)
        packet_lengths.append(seq['mapped_lengths'].values)


    # Convert to PyTorch tensors
    interpacket_time_normalized_tensor = torch.tensor(np.array(interpacket_time_normalized),
                                                      dtype=torch.long)  # Input sequences
    packet_lengths_tensor = torch.tensor(np.array(packet_lengths), dtype=torch.long)  # Length normalized
    targets_tensor = torch.tensor(np.array(targets), dtype=torch.long)

    return interpacket_time_normalized_tensor.to(device), packet_lengths_tensor.to(device), targets_tensor.to(device)


def create_dataset(X_interpacket_time, X_packet_lengths, Y):
    return TimeSeriesDataset(
        X_interpacket_time,
        X_packet_lengths,
        Y,
    )


def encode_and_set_decoder(df):
    ##TODO mappings cannot be ambiguous. hardcoded for now
    value_to_int = {26: 0,  27: 1}
    #unique_values = df['encrypted_lengths'].unique()
    #value_to_int = {val: idx for idx, val in enumerate(unique_values)}

    # Add the mapping to the dataframe
    df['mapped_lengths'] = df['encrypted_lengths'].map(value_to_int)
    decoder = {idx: val for val, idx in value_to_int.items()}
    return df, decoder


def preprocess(path, block_size):
    df = read(path)
    df, _ = encode_and_set_decoder(df)
    df = add_next_length_feature(df)
    df, time_scaler = preprocess_time(df)
    sequence_length = block_size
    X_interpacket_time, X_packet_lengths, Y = create_sequences(df, sequence_length, torch.device('cpu'))
    return create_dataset(X_interpacket_time, X_packet_lengths, Y)


def read(path):
    df = pd.read_csv(path)
    df_filtered = df[['interpacket_time', 'Protocol', 'encrypted_lengths']].copy()
    df_filtered = df_filtered.head(int(len(df_filtered)))
    return df_filtered

