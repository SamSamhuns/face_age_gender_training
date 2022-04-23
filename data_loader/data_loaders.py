import os
import random
import logging
import cv2
import torch
import numpy as np

from base import BaseDataLoader
from utils import recursively_get_file_paths


class BatchNormalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[
            None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[
            None, :, None, None]
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor.float()


class GenderDataset(torch.utils.data.Dataset):
    """

    There are in in total of three datasets available. They are Adience, wiki,
    and imdb. wiki and imdb often go togther so basically only two datasets
    should be considered (Adience and imdb_wiki)

    You should extract the arcface 512-D face feature vectors. If you haven't,
    go read README.md first.

    Note that Adience is loaded differently from imdb_wiki since Adience
    officially has test split while imdb and wiki don't.

    male is labeled as 0 and female is labeled as 1.

    """

    def __init__(self, data_dir: str = 'data', dataset: str = None,
                 training: bool = True,
                 test_cross_val: int = None, limit_data: int = None):
        logging.info(f"test cross val is {test_cross_val}")
        if dataset.lower() == 'adience':
            data = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                           allow_pickle=True).item()
            if training:
                data = [data[i] for i in range(5) if i != test_cross_val]
                data = [d for da in data for d in da]
            else:
                data = data[test_cross_val]
        elif "gender" in dataset.lower():
            data = recursively_get_file_paths(os.path.join(data_dir, dataset))
        elif dataset.lower() in ['wiki', 'imdb']:
            data = np.load(os.path.join(data_dir, f"{dataset.lower()}/data.npy"),
                           allow_pickle=True)
        elif dataset.lower() == 'imdb_wiki':
            data_imdb = np.load(os.path.join(data_dir, "imdb/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, "wiki/data.npy"),
                                allow_pickle=True).tolist()
            data = data_imdb + data_wiki
            del data_imdb, data_wiki
        elif dataset.lower() == 'imdb_wiki_adience':
            data_imdb = np.load(os.path.join(data_dir, "imdb/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, "wiki/data.npy"),
                                allow_pickle=True).tolist()
            data_adience = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                                   allow_pickle=True).item()
            data_adience = [sample for _, samples in data_adience.items()
                            for sample in samples]
            data = data_imdb + data_wiki + data_adience
            del data_imdb, data_wiki, data_adience
        else:
            raise NotImplementedError

        if limit_data is not None:
            logging.info(
                f"reducing data samples from {len(data)} to {len(data[:limit_data])} ...")
            random.shuffle(data)
            self.data = data[:limit_data]
        else:
            self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return an feature and a target label (gender).
        """
        # x = self.data[idx]['feature']
        # y = {'m': 0, 'f': 1}[self.data[idx]['gender']]
        data = np.load(self.data[idx], allow_pickle=True)
        X = data.item().get('feature')
        y = data.item().get('label')

        return X, y


class AgeDataset(torch.utils.data.Dataset):
    """
    There are in in total of three datasets available. They are Adience, wiki,
    and imdb. wiki and imdb often go togther so basically only two datasets
    should be considered (Adience and imdb_wiki)

    You should extract the 512-D face feature vectors. If you haven't,
    go read README.md first.

    Note that Adience is loaded differently from imdb_wiki since Adience
    officially has test split while imdb and wiki don't.

    The Adience dataset has 8 age classes while the imdb and wiki datasets have
    101 age classes.

    """

    def __init__(self, data_dir: str = 'data', dataset: str = None, training: bool = True,
                 test_cross_val: int = None, num_classes: int = None, limit_data: int = None):
        logging.info(f"test cross val is {test_cross_val}")

        if num_classes == 5:
            self.age_map = {0: 0, 7: 1, 15: 2, 35: 3, 75: 4}
        elif num_classes == 8:
            self.age_map = {0: 0, 5.0: 1, 10.0: 2, 17.5: 3, 28.5: 4,
                            40.5: 5, 50.5: 6, 80.0: 7}
        elif num_classes == 101:
            self.age_map = {i: i for i in range(101)}
        elif num_classes == 1:
            logging.warning("Attempting a regression task instead of "
                            "classification. Discouraged as previous exps show it "
                            "doesn't perform so well.")
            self.age_map = None
        elif num_classes == 4:
            logging.info("Starting age classification for 4 age groups")
        else:
            raise NotImplementedError(
                f"num_classes {num_classes} for ages not implemented.")

        if dataset == 'Adience':
            data = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                           allow_pickle=True).item()
            if training:
                data = [data[i] for i in range(5) if i != test_cross_val]
                data = [d for da in data for d in da]
            else:
                data = data[test_cross_val]
        elif "age" in dataset.lower():
            data = recursively_get_file_paths(os.path.join(data_dir, dataset))
        elif dataset.lower() in ['wiki', 'imdb']:
            data = np.load(os.path.join(data_dir, f"{dataset.lower()}/data.npy"),
                           allow_pickle=True)
        elif dataset.lower() == 'imdb_wiki':
            data_imdb = np.load(os.path.join(data_dir, "imdb/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, "wiki/data.npy"),
                                allow_pickle=True).tolist()
            data = data_imdb + data_wiki
            del data_imdb, data_wiki
        elif dataset.lower() == 'imdb_wiki_adience':
            data_imdb = np.load(os.path.join(data_dir, "imdb/data.npy"),
                                allow_pickle=True).tolist()
            data_wiki = np.load(os.path.join(data_dir, "wiki/data.npy"),
                                allow_pickle=True).tolist()
            data_adience = np.load(os.path.join(data_dir, "Adience/data-aligned.npy"),
                                   allow_pickle=True).item()
            data_adience = [sample for _, samples in data_adience.items()
                            for sample in samples]
            data = data_imdb + data_wiki + data_adience
            del data_imdb, data_wiki, data_adience
        else:
            raise NotImplementedError

        if limit_data is not None:
            logging.info(
                f"reducing data samples from {len(data)} to {len(data[:limit_data])} ...")
            random.shuffle(data)
            self.data = data[:limit_data]
        else:
            self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def _get_closest_age(self, num: float) -> float:
        """
        Get the closest age from the possible age range.
        """
        possible_ages = np.array([age for age in list(self.age_map.keys())])
        idx = int(np.argmin(np.abs(possible_ages - num)))

        return possible_ages[idx]

    def __getitem__(self, idx: int) -> tuple:
        """
        Return an feature and a target label (age).
        """
        # x = self.data[idx]['feature']

        # In case you want to do regression in classification, the target should be
        # a floating point number. I tried this and the result is worse with
        # regression. Juse do classification with cross entropy loss.
        # if self.age_map is None:
        #     y = np.float32([self.data[idx]['age']])
        # else:
        #     y = self.age_map[self._get_closest_age(self.data[idx]['age'])]

        data = np.load(self.data[idx], allow_pickle=True)
        X = data.item().get('feature')
        y = data.item().get('label')
        return X, y


class FeaturesDataset(torch.utils.data.Dataset):
    """
    Dataset for loading feature vectors for training. Data point must be npy files containing
    python objects with 'feature' and 'label' parameters
    """

    def __init__(self, data_dir: str = 'data', dataset: str = None,
                 training: bool = True, limit_data: int = None, **kwargs):

        if "feat" in dataset:
            data_paths = recursively_get_file_paths(
                os.path.join(data_dir, dataset))
        else:
            raise NotImplementedError(f"{dataset} is not implemented.")

        if limit_data is not None:
            logging.info(
                f"reducing data samples from {len(data_paths)} to {len(data_paths[:limit_data])} ...")
            random.shuffle(data_paths)
            self.data_paths = data_paths[:limit_data]
        else:
            self.data_paths = data_paths

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return an feature and a target label (video classification).
        """
        data = np.load(self.data_paths[idx], allow_pickle=True)
        X = data.item().get('feature')
        # X = X.reshape(60, 128)
        # X = X[:45]
        y = data.item().get('label')
        return X, y


class VideoFramesDataset(torch.utils.data.Dataset):
    """
    Dataset for loading video numpy frames for training.
    Data point must be npy.npz file containing images in npy format (batch, height, width, channel)
    """

    def __init__(self, data_dir: str = 'data', dataset: str = None,
                 training: bool = True, limit_data: int = None, **kwargs):

        if "frames" in dataset:
            data_paths = recursively_get_file_paths(
                os.path.join(data_dir, dataset), ext="npz")
        else:
            raise NotImplementedError(f"{dataset} is not implemented.")

        self.cname_to_label = {'female_n_male': 0,
                               'male': 1,
                               'female_with_kids': 2,
                               'male_with_kids': 3,
                               'female': 4,
                               'family': 5,
                               'none': 6,
                               'young_kid': 7}
        self.transforms = BatchNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True)

        if limit_data is not None:
            logging.info(
                f"reducing data samples from {len(data_paths)} to {len(data_paths[:limit_data])} ...")
            random.shuffle(data_paths)
            self.data_paths = data_paths[:limit_data]
        else:
            self.data_paths = data_paths

    def __len__(self) -> int:
        return len(self.data_paths)

    def __getitem__(self, idx: int) -> tuple:
        """
        Return an feature and a target label (video classification).
        """
        data = np.load(self.data_paths[idx])
        X = data["arr"]
        X = np.asarray([cv2.resize(img, (299, 299)) for img in X])
        b, h, w, c = X.shape
        # switch from HWC to CHW
        X = X.reshape(b, c, h, w)
        if self.transforms:
            X = torch.as_tensor(X)
            X = self.transforms(X)
        # get label name from data path
        y = self.cname_to_label[self.data_paths[idx].split('/')[-2]]
        return X, y


class GenderDataLoader(BaseDataLoader):
    """
    Note that this data loader class is sub-classes the BaseDataLoader class,
    which again sub-classes the vanilla pytorch DataLoader class. See
    data_loader/data_loaders.py for the details.

    """

    def __init__(self, data_dir: str, batch_size: int, shuffle: bool, validation_split: float,
                 num_workers: int, dataset: str, num_classes: int, training: bool, test_cross_val: int = None,
                 limit_data: int = None, **kwargs):

        assert num_classes == 2
        self.dataset = GenderDataset(data_dir=data_dir, dataset=dataset,
                                     test_cross_val=test_cross_val,
                                     training=training, limit_data=limit_data)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)


class AgeDataLoader(BaseDataLoader):
    """
    Note that this data loader class is sub-classes the BaseDataLoader class,
    which again sub-classes the vanilla pytorch DataLoader class. See
    data_loader/data_loaders.py for the details.

    """

    def __init__(self, data_dir: str, batch_size: int, shuffle: bool, validation_split: float,
                 num_workers: int, dataset: str, num_classes: int, training: bool, test_cross_val: int = None,
                 limit_data: int = None, **kwargs):

        self.dataset = AgeDataset(data_dir=data_dir, dataset=dataset,
                                  test_cross_val=test_cross_val,
                                  training=training, num_classes=num_classes,
                                  limit_data=limit_data)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)


class FeaturesDataLoader(BaseDataLoader):
    """
    DataLoader for Features
    """

    def __init__(self, data_dir: str, batch_size: int, shuffle: bool, validation_split: float,
                 num_workers: int, dataset: str, num_classes: int, training: bool, limit_data: int = None, **kwargs):

        self.dataset = FeaturesDataset(
            data_dir=data_dir, dataset=dataset, limit_data=limit_data, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class VideoFramesDataLoader(BaseDataLoader):
    """
    DataLoader for Video Frames
    """

    def __init__(self, data_dir: str, batch_size: int, shuffle: bool, validation_split: float,
                 num_workers: int, dataset: str, num_classes: int, training: bool, limit_data: int = None, **kwargs):

        self.dataset = VideoFramesDataset(
            data_dir=data_dir, dataset=dataset, limit_data=limit_data, **kwargs)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
