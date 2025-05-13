import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ct_dataset(Dataset):
    def __init__(
        self,
        mode,
        saved_path=None,
        dataset=None,
    ):
        assert mode in ["train", "test"], "mode is 'train' or 'test'"

        # if mode == "train":
        #     self.input_path = "/mnt/new_ssd/dataset/restormer/XCAT90/train/Low_dose"
        #     self.target_path = "/mnt/new_ssd/dataset/restormer/XCAT90/train/High_dose"
        #     # self.input_path = "/mnt/new_ssd/dataset/restormer/GE512/Train/AAPM/Low_dose"
        #     # self.target_path = (
        #     #     "/mnt/new_ssd/dataset/restormer/GE512/Train/AAPM/High_dose"
        #     # )
        # elif mode == "test":
        #     self.input_path = "/mnt/new_ssd/dataset/restormer/XCAT90/test/Low_dose"
        #     self.target_path = "/mnt/new_ssd/dataset/restormer/XCAT90/test/High_dose"
        #     # self.input_path = "/mnt/new_ssd/dataset/restormer/GE512/Test/AAPM/Low_dose"
        #     # self.target_path = (
        #     #     "/mnt/new_ssd/dataset/restormer/GE512/Test/AAPM/High_dose"
        #     # )

        # self.input_files = os.listdir(self.input_path)
        # self.target_files = os.listdir(self.target_path)

        assert saved_path is not None or dataset is not None, (
            "saved_path or dataset must be provided"
        )
        if dataset is not None:
            data_path = dataset
        else:
            data_path = saved_path

        data = torch.load(data_path)
        print(f"loaded dataset from {data_path} shape {data.shape}")
        self.input_files = [f"{i}.npy" for i in range(data.shape[0])]
        self.target_files = [f"{i}.npy" for i in range(data.shape[0])]
        self.input_data = data[:, 0]
        self.target_data = data[:, 1]
        self.image_size = self.input_data.shape[-1]
        self.input_data = self.input_data.view(
            self.input_data.shape[0], 1, self.image_size, self.image_size
        )
        self.target_data = self.target_data.view(
            self.target_data.shape[0], 1, self.image_size, self.image_size
        )

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return (
            self.input_data[idx],
            self.target_data[idx],
            self.input_files[idx],
            self.target_files[idx],
        )


def get_loader(
    mode="train",
    saved_path=None,
    dataset=None,
    batch_size=32,
    shuffle=True,
    num_workers=4,
):
    dataset_ = ct_dataset(mode, saved_path, dataset)
    data_loader = DataLoader(
        dataset=dataset_,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_loader
