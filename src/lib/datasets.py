import os

import torch
import torchvision

import requests
import zipfile
import io
from typing import Optional, Dict, List, Tuple


def get_size(start_path = '.'):
    """
    Got from:
        https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


def download_fg_dataset(path: str, url: str, can_skip_download: bool = False):
    """"
    Downloads and extracts the dataset from a given `url` into a given `path`

    NOTE: this functions has been thought for the FG dataset, so it has not
    been tested against other datasets that can be found in URLs
    """

    # Create the dir if it does not exist
    if os.path.exists(path) is False:
        print(f"Dir {path} does not exist, creating that dir")
        os.mkdir(path)

    # If the dir has a filesize bigger than 42.2MB, then it should be already
    # downloaded, and we can skip this step. However, the user can tell this
    # function to do not skip, so they assure there has not been any data
    # corruption
    file_B = get_size(path)
    file_MB = file_B / (1024 * 1024)

    if file_MB > 44.2 and can_skip_download is True:
        print("Skipping the download, files are already downloaded")
        return

    # Download the dataset
    try:
        print("Downloading the dataset")
        req = requests.get(url)
    except Exception as e:
        print("ERROR: could not download data from url")
        print(f"ERROR: error is:\n{e}")
        return

    # Extract the dataset contents
    print("Extracting the dataset contents")
    zip_file = zipfile.ZipFile(io.BytesIO(req.content))
    zip_file.extractall(path)

    print("Succesful download")

# TODO -- properly document this class
# TODO -- say that we are not using all the metadata that the dataset has
class FGDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, transform = None):

        # Path of the dir where all the images are stored
        # NOTE: This is the path of the image dir, and not the dataset dir, where
        # some metadata is also stored
        self.path = path

        # Transformation to apply to the images of the dataset
        # Items of this dataset are made up of: image, id and age
        # But the transformation is done only to the image
        self.transform = transform

        # Dict containing the following data:
        # - Keys are the ids of the individuals
        # - Values are lists with all the ages associated with that individual
        self.individuals: Optional[Dict[int, List[int]]] = None

        # `self.individuals` already have the information about the targets
        # However, even though pytorch has no documented this (AFAIK), most of
        # the functions expect `torch.utils.data.Dataset` to have a `targets`
        # attribute
        self.targets: Optional[List[int]] = None

        # Number of images stored in this class
        self.__number_images: Optional[int] = None

        # All the filenames of the images stored in `self.path` dir
        self.file_names: Optional[List] = None

        # Get the data from the dir and instanciate all the attributes of this class
        # Before that, all the attrs are None
        self.__generate_dataset()

        # Check that the dataset is properly created
        self.__check_integrity()

        # In this dataset, each element has an image, an id and the age
        # In the EDA we want to work with the tree, but in the training, most
        # of the methods expect `__getitem__` to return (image, target)
        #
        # So with this attribute, and `set_exploration_mode()`, we can control
        # the behaviour of `__getitem__`
        self.exploration_mode: bool = False

        super(FGDataset, self).__init__()

    def __len__(self) -> int:

        # Check that we have the number of images of the dataset
        if self.__number_images is None:
            raise Exception(
                "Dataset is not initialized, thus, number of images is unknown"
            )

        return self.__number_images

    def __getitem__(self, idx):
        """
        NOTE: returns (image, id) if `self.exploration_mode == False`, otherwise
        it returns a dict with the image, id and age
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image from the index
        img_name = os.path.join(self.path, self.file_names[idx])
        image = torchvision.io.read_image(img_name)

        # Get the id and age from the file_name
        id, age = self.__id_and_age_from_file_name(self.file_names[idx])

        # Put together all the info
        sample = {
            "image": image,
            "id": id,
            "age": age,
        }

        # Items are made up of: image, id and age; as the prev code shows
        # But the transform is only made to the image, that is the only
        # part of the dict where it makes sense
        if self.transform:
            sample["image"] = self.transform(sample["image"])


        # Check if we are or not in exploration mode. As above-mentioned, return
        # one type of data or another depending on the mode
        if self.exploration_mode is True:
            return sample

        return sample["image"], sample["id"]

    def __generate_dataset(self):

        # Get all the names of the files
        self.file_names = os.listdir(self.path)

        # Use that for computing the size
        self.__number_images = len(self.file_names)

        # Instanciate some attributes
        self.individuals = dict()
        self.targets = []

        # Use the names to get the persons IDs and their ages
        for file_name in self.file_names:

            # Split into id and age
            id, age = self.__id_and_age_from_file_name(file_name)

            # ID's are what we want in the targets
            self.targets.append(id)

            # If there is not already a instance for this id, create it
            # and set the initial value for the list
            if self.individuals.get(id) is None:
                self.individuals[id] = [age]
                continue

            # This individual already has a list (we have checked before)
            # so append to that list
            self.individuals[id].append(age)

    def __id_and_age_from_file_name(self, file_name: str) -> Tuple[str, str]:
        # Remove file extension
        file_name_no_extension = file_name.split(".JPG")[0]

        # Split into id and age
        id, age = file_name_no_extension.split("A")

        # Age can contain trailing letters, for example, the entries corresponding to:
        # `068A10a.JPG` and `068A10b.JPG`
        age = ''.join(filter(str.isdigit, age))

        # Now it is safe to cast both id and age to an int
        id, age = int(id), int(age)

        return id, age

    def __check_integrity(self):

        # Check that we have the proper number of images
        assert self.__len__() == 1002
        assert len(self.targets) == 1002

    def set_exploration_mode(self, mode: bool):
        self.exploration_mode = mode
