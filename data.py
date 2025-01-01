import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from preprocess import preprocess


class LanguageModelDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=64, block_size=256, num_workers=1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download dataset if necessary
        pass

    def setup(self, stage=None):
        self.train_dataset = preprocess('data/generated_packets.csv', self.block_size)
        self.val_dataset = self.train_dataset
        self.test_dataset = preprocess('data/generated_packets.csv', self.block_size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
