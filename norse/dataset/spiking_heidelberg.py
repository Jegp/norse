"""
The Spiking Heidelberg Digits (SHD) audio dataset
https://compneuro.net/posts/2019-spiking-heidelberg-digits/
Licensed under CC A 4.0

Cramer, B., Stradmann, Y., Schemmel, J., and Zenke, F. (2019). 
The Heidelberg spiking datasets for the systematic evaluation of spiking neural networks. 
ArXiv:1910.07407 [Cs, q-Bio]. https://arxiv.org/abs/1910.07407
"""

import os
import h5py
import tqdm

import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class SpikingHeidelberg(torch.utils.data.Dataset):
    """
    Initialises, but does not download by default, the
    `Spiking Heidelberg audio dataset <https://compneuro.net/posts/2019-spiking-heidelberg-digits/>`_.

    Parameters:
        root (str): The root of the dataset directory
        sparse (bool): Whether or not to load the data as sparse tensors. True by default
        train (bool, optional): If True, creates dataset from training set, otherwise
            we only use te test set.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    test_file, test_file_md5 = "shd_test.h5", "207ad1295a0ab611e22d1734989c254e"
    test_url = "https://compneuro.net/datasets/shd_test.h5.gz"
    test_checksum = "3062a80ec0c5719404d5b02e166543b1"
    train_file, train_file_md5 = "shd_train.h5", "8e9877c85f29a28dd353ea6a54402667"
    train_url = "https://compneuro.net/datasets/shd_train.h5.gz"
    train_checksum = "d47c9825dee33347913e8ce0f2be08b0"
    n_units = 700  # 700 channels

    def __init__(self, root, dt=0.001, sparse=True, train=True, download=False):
        super(SpikingHeidelberg).__init__()
        self.root = root
        self.train = train
        self.dt = dt

        self.data_dir = os.path.join(self.root, "shd_data")
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)
        self.test_data_file = os.path.join(self.data_dir, "test.dat")
        self.train_data_file = os.path.join(self.data_dir, "train.dat")

        if download:
            self.download()

        self.data = []
        if train:
            self.data = self.data + torch.load(self.train_data_file)
        self.data = self.data + torch.load(self.test_data_file)

    def _check_integrity(self):
        return os.path.isfile(self.train_data_file) and os.path.isfile(
            self.test_data_file
        )

    def _generate_data(self, h5file):
        with h5py.File(h5file, "r") as fp:
            points = []
            for times, units, label in tqdm.tqdm(
                zip(fp["spikes"]["times"], fp["spikes"]["units"], fp["labels"]),
                total=len(fp["labels"]),
            ):
                points.append((self._bin_spikes(times, units), label))
            return points

    def _bin_spikes(self, times, units):
        assert len(times) == len(units), "Spikes and units must have same length"
        length = int(times.max() // self.dt + 1)
        data = torch.zeros((length, self.n_units), dtype=torch.uint8)
        for index in range(len(times)):
            time_index = int(times[index] // self.dt)
            unit_index = units[index]
            data[time_index][unit_index] = 1
        return data.to_sparse()

    def download(self):
        if not self._check_integrity():
            if self.train:
                download_and_extract_archive(
                    self.train_url, self.root, md5=self.train_checksum
                )
                data_train = self._generate_data(self.train_file)
                torch.save(data_train, self.train_data_file)
            download_and_extract_archive(
                self.test_url, self.root, md5=self.test_checksum
            )
            data_test = self._generate_data(self.test_file)
            torch.save(data_test, self.test_data_file)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)