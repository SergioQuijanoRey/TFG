import torch
import torchvision
import numpy as np
from tqdm import tqdm
from collections import Counter
from typing import Callable


# utils import depend on enviroment (local or remote), so we can do two try-except blocks
# for dealing with that
try:
    import utils
except Exception as e:
    pass
try:
    import src.lib.utils as utils
except Exception as e:
    pass



import logging
file_logger = logging.getLogger("MAIN_LOGGER")

# TODO -- TEST -- augmentate_dataset with min_number_of_images = 1 should do nothing
# TODO -- TEST -- check that original dataset is not modified
# TODO -- TEST -- how many images have at least K images should be 100%

class AugmentatedDataset(torch.utils.data.Dataset):
    """
    Dataset class that gets a base dataset class, and augmentates it

    Augmentation only is done on classes that have less than a given number of
    images
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        min_number_of_images: int,
        transform
    ):

        super(AugmentatedDataset, self).__init__()

        # Store base dataset as we're going to use its information
        # This way, we don't need to copy all the information, which is very slow
        # for large enough datasets
        self.base_dataset = base_dataset

        # A transformation that is random
        # That's to say, each time is called on an image, the transformation performed is different
        # This way we can use a single "value" instead of having to use a generator or something
        # similar that would make things slow
        self.transform = transform

        # Create new images (with associated labels)
        self.new_images, self.new_targets = self.__augmentate_base_dataset(min_number_of_images)

        # Now we can create the targets list
        self.targets = self.base_dataset.targets + self.new_targets

    def __getitem__(self, index: int):

        # If the index is from the original dataset, return from that
        if index < len(self.base_dataset):
            return self.base_dataset.__getitem__(index)

        else:

            # Index in the lists stored in the class
            resized_index = index - len(self.base_dataset)

            # Return the data from new generated images and labels
            return self.new_images[resized_index], self.new_targets[resized_index]


    def __len__(self) -> int:
        return len(self.base_dataset) + len(self.new_images)

    def __augmentate_base_dataset(self, min_number_of_images: int):
        """
        Creates two lists, `new_images` and `new_targets`, storing the new images and new targets
        that the `self.base_dataset` needs. That's to say, the augmented data
        """

        # Initialize the two lists that we're going to build (list of images
        # and list of their targets)
        new_images = []
        new_targets = []

        # Get the counter of how many images have each class
        how_many_images_per_class = Counter(self.base_dataset.targets)

        # Get the classes that have less than the min number of images
        classes_with_less_than_min = [
            curr_class
            for (curr_class, images_of_that_class) in how_many_images_per_class.items()
            if images_of_that_class < min_number_of_images
        ]

        file_logger.debug(f"{len(classes_with_less_than_min)} classes need data augmentation")

        # Before doing the augmentation, we pre-compute dict of classes for speeding
        # up the computations
        dict_of_classes = utils.precompute_dict_of_classes(self.base_dataset.targets)

        # Iterate over all classes that have less than min_number_of_images images
        # This process might be slow, so plot the progress using tqdm
        for small_class in tqdm(classes_with_less_than_min):

            # Compute how many images we need to create to reach min_number_of_images
            number_images_needed = min_number_of_images - how_many_images_per_class[small_class]

            # Create number_images_needed images and put them in the new lists
            for _ in range(number_images_needed):

                # Select a random image of the class
                img_idx = int(np.random.choice(len(dict_of_classes[small_class]), size = 1))
                img_idx = dict_of_classes[small_class][img_idx]
                img, _ = self.base_dataset[img_idx]

                # Create a new random img
                new_img = self.transform(img)

                # Add that img to the new lists
                new_images.append(new_img)
                new_targets.append(small_class)

        return new_images, new_targets
