import numpy as np
import tensorflow as tf

import os

from tensorflow.keras.layers import StringLookup
from typing import Tuple, List
from tqdm import tqdm


class DataPreperation:
    """
    This class prepares the IAM Words dataset
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.train_samples = None
        self.test_samples = None
        self.validation_samples = None
        self.words = list()
    
    def __split_dataset(self) -> None:
        """
        Create train, test and validation from given dataset
        :param dataset_path: (str) path to the dataset base folder
        :return: None
        """
        words = open(f"{self.dataset_path}/words.txt", "r").readlines()
        for line in words:
            if line[0] == "#":
                continue
            if line.split(" ")[1] != "err":
                self.words.append(line)

        np.random.shuffle(self.words)

        split_idx = int(0.9 * len(self.words))
        self.train_samples = self.words[:split_idx]
        self.test_samples = self.words[split_idx:]

        val_split_idx = int(0.5 * len(self.test_samples))
        self.validation_samples = self.test_samples[:val_split_idx]
        self.test_samples = self.test_samples[val_split_idx:]
    
    def __paths_and_labels(self, samples):
        base_image_path = os.path.join(self.dataset_path, "words")
        paths = []
        corrected_samples = []
        for (i, file_line) in tqdm(enumerate(samples), total=len(samples)):
            line_split = file_line.strip()
            line_split = line_split.split(" ")

            image_name = line_split[0]
            partI = image_name.split("-")[0]
            partII = image_name.split("-")[1]
            img_path = os.path.join(
                base_image_path, partI, partI + "-" + partII, image_name + ".png"
            )
            if os.path.getsize(img_path):
                paths.append(img_path)
                corrected_samples.append(file_line.split("\n")[0])
            
        return paths, corrected_samples
    
    @staticmethod
    def clean_labels(labels):
        cleaned_labels = []
        for label in labels:
            label = label.split(" ")[-1].strip()
            cleaned_labels.append(label)
        
        return cleaned_labels
    
    def __call__(self):
        self.__split_dataset()
        print("Preparing train samples")
        self.train_img_paths, self.train_labels = self.__paths_and_labels(self.train_samples)
        print("Preparing Validation samples")
        self.validation_img_paths, self.validation_labels = self.__paths_and_labels(self.validation_samples)
        print("Preparing test samples")
        self.test_img_paths, self.test_labels = self.__paths_and_labels(self.test_samples)

        characters = set()
        self.max_len = 0

        clean_lables = []

        for label in self.train_labels:
            label = label.split(" ")[-1].strip()
            for char in label:
                characters.add(char)

            self.max_len = max(self.max_len, len(label))
            clean_lables.append(label)

        self.characters = sorted(list(characters))

        self.train_labels_cleaned = clean_lables
        self.validation_labels_cleaned = DataPreperation.clean_labels(self.validation_labels)
        self.test_labels_cleaned = DataPreperation.clean_labels(self.test_labels)
