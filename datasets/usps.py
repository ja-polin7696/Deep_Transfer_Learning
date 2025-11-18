"""Dataset setting and data loader for USPS.

Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py
"""

import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms

import params


class USPS(data.Dataset):
    """USPS Dataset."""

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        self.transform = transform
        self.dataset_size = None

        # Download if needed
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it.")

        # Load data
        self.train_data, self.train_labels = self.load_samples()

        # Shuffle if training
        if self.train:
            indices = np.arange(self.dataset_size)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices]
            self.train_labels = self.train_labels[indices]

        # Original data is 0..1 → multiply to 0..255
        # (we will fix normalization in transform)
        self.train_data = (self.train_data * 255.0).astype(np.uint8)

        # Convert to HWC (height, width, channel)
        self.train_data = self.train_data.transpose(0, 2, 3, 1)

    def __getitem__(self, index):
        img = self.train_data[index]
        label = self.train_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __len__(self):
        return self.dataset_size

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        if os.path.isfile(filename):
            return

        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")

    def load_samples(self):
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()

        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]

        self.dataset_size = labels.shape[0]
        return images, labels


def get_usps(train):
    """Get USPS dataset loader."""

    pre_process = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))   # IMPORTANT: 0–255 → normalized to 0–1
    ])

    usps_dataset = USPS(
        root=params.data_root,
        train=train,
        transform=pre_process,
        download=True
    )

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=params.batch_size,
        shuffle=True
    )

    return usps_data_loader
