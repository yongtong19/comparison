import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat


class CTDataset(Dataset):
    def __init__(
        self,
        dataset: str,
    ):
        assert os.path.exists(dataset), "dataset does not exist"

        self.dataset = dataset
        self.input_files = sorted(glob.glob(os.path.join(dataset, "img", "*.mat")))
        self.target_files = sorted(glob.glob(os.path.join(dataset, "label", "*.mat")))

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        if input_file.endswith(".mat"):
            input_data = loadmat(os.path.join(self.dataset, "img", input_file)).get(
                "imagesparse"
            )

        if target_file.endswith(".mat"):
            target_data = loadmat(os.path.join(self.dataset, "label", target_file)).get(
                "imagef"
            )

        return (
            torch.from_numpy(input_data).reshape(
                1, input_data.shape[-2], input_data.shape[-1]
            ),
            torch.from_numpy(target_data).reshape(
                1, target_data.shape[-2], target_data.shape[-1]
            ),
        )

    @property
    def image_size(self):
        return self.__getitem__(0).shape[-1]


def get_loader(
    dataset: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
):
    dataset_ = CTDataset(dataset)
    data_loader = DataLoader(
        dataset=dataset_,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_loader
