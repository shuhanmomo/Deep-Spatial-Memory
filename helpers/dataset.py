import os
import numpy as np
import torch
from torch.utils.data import Dataset
import zipfile
import tempfile
import re
from tqdm import tqdm
from helpers.utils import read_image
import matplotlib.pyplot as plt


def format_name(zip_file_name):
    """
    reads zip file name and format it into lower case
    """
    formatted_name = zip_file_name.lower().replace(".zip", "").split("_")
    return formatted_name


class BldgDataset(Dataset):
    def __init__(
        self,
        data_path="./data/experiments/casestudy.zip",
        mode="train",
        transform=None,
        seq_len=5,
        num_seq=6,
        num_frame=30,
        seed=42,
    ):
        super(BldgDataset, self).__init__()

        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.building_names = []
        self.num_frame = num_frame  # 30 frames each path

        # Load dataset
        self.data = []
        self.mean = 0.0
        self.std = 0.0
        self.load_dataset()
        self.split_data(seed)

    def load_dataset(self):
        sum_pixels = np.float64(0)
        sum_pixels_squared = np.float64(0)
        pixel_count = np.float64(0)
        # Extract the main zip file if needed
        with zipfile.ZipFile(self.data_path, "r") as main_zip:
            with tempfile.TemporaryDirectory() as temp_dir:
                main_zip.extractall(temp_dir)

                # Iterate over each building sequence label zip file inside the main directory
                for bldg_zip_name in main_zip.namelist():
                    bldg_route_label = format_name(bldg_zip_name)
                    # The last character is the route label a,b,c,d
                    bldg, start_type, end_type, route = bldg_route_label
                    if bldg not in self.building_names:
                        self.building_names.append(bldg)

                    # Extract the building sequence label zip file

                    bldg_zip_path = os.path.join(temp_dir, bldg_zip_name)
                    with zipfile.ZipFile(bldg_zip_path, "r") as bldg_zip:
                        bldg_temp_dir = os.path.join(
                            temp_dir, " ".join(bldg_route_label)
                        )
                        bldg_zip.extractall(bldg_temp_dir)

                        # Iterate over path folders
                        for path_folder in tqdm(
                            sorted(os.listdir(bldg_temp_dir)),
                            desc=f"Loading {bldg}",
                            unit="path",
                        ):
                            if path_folder.startswith("path"):
                                path_images = []
                                path_folder_full = os.path.join(
                                    bldg_temp_dir, path_folder
                                )

                                for frame in range(
                                    self.num_frame
                                ):  # Assuming 30 frames per path
                                    img_filename = f"panoramic_{frame:02d}.png"
                                    img_path = os.path.join(
                                        path_folder_full, img_filename
                                    )
                                    if os.path.exists(img_path):
                                        img_array = read_image(img_path).astype(
                                            np.float64
                                        )
                                        path_images.append(img_array)
                                        # Update the sums for mean and std calculation
                                        sum_pixels += img_array.sum()
                                        sum_pixels_squared += (img_array**2).sum()
                                        pixel_count += img_array.size

                                # Only consider complete sequences with 30 frames
                                if len(path_images) != self.num_frame:
                                    print(
                                        f"Error: {bldg}, Route {route}, Type {start_type} to {end_type},Path {path_folder} does not have 30 images."
                                    )
                                    return
                                else:
                                    self.data.append(
                                        {
                                            "images": np.concatenate(
                                                path_images, axis=0
                                            ),
                                            "path": int(
                                                re.search(r"\d+", path_folder).group()
                                            ),
                                            "route": route,
                                            "bldg": bldg,
                                            "start_type": start_type,
                                            "end_type": end_type,
                                        }
                                    )
        self.mean = sum_pixels / pixel_count
        variance = (sum_pixels_squared / pixel_count) - (self.mean**2)
        if variance < 0:
            if np.isclose(variance, 0):
                self.std = 0
            else:
                raise ValueError(f"Calculated negative variance: {variance}")
        else:
            self.std = np.sqrt(variance)

    def __getitem__(self, index):
        item = self.data[index]
        imgs = item["images"]
        t_imgs = imgs.reshape(-1, 1, 30, 60)  # num_frame, C, H, W
        # Apply transform if provided
        if self.transform:
            t_imgs = self.transform(t_imgs)
        t_imgs = np.stack(t_imgs, axis=0)
        t_imgs = torch.from_numpy(t_imgs).float()
        # normalize
        t_imgs = (t_imgs - self.mean) / self.std
        (C, H, W) = t_imgs[0].size()
        t_imgs = t_imgs.view(self.num_seq, self.seq_len, C, H, W).transpose(
            1, 2
        )  # num_seq,C,seq_len,H,W

        # Return data as a dictionary
        return {
            "t_imgs": t_imgs,
            "imgs": imgs,
            "path": item["path"],
            "route": item["route"],
            "bldg": item["bldg"],
            "start_type": item["start_type"],
            "end_type": item["end_type"],
        }

    def split_data(self, seed):
        # Split the data into training, validation, and test sets
        np.random.seed(seed)
        np.random.shuffle(self.data)  # Shuffle the data

        train_split = int(len(self.data) * 0.7)
        val_split = int(len(self.data) * 0.85)

        if self.mode == "train":
            self.data = self.data[:train_split]
        elif self.mode == "val":
            self.data = self.data[train_split:val_split]
        elif self.mode == "test":
            self.data = self.data[val_split:]
        elif self.mode == "predict":
            self.data = self.data[::]  # Keep all data for prediction mode

    def __len__(self):
        return len(self.data)

    def find_indices(self, bldg_name, sequence_type, route, path_number):
        """
        Finds the indices of the data items that match the given building, route, and path number.

        :param bldg_name: The name of the building (formatted as 'caracalla baths', for example).
        :param sequence_type: The tuple of start and end type such as (room,passage)
        :param route: The route label (a single character like 'a', 'b', etc.).
        :param path_number: The path number (an integer).
        :return: A list of indices that match the criteria.
        """
        indices = []
        for idx, item in enumerate(self.data):
            if (
                item["bldg"].lower() == bldg_name.lower()
                and item["route"].lower() == route.lower()
                and item["path"] == path_number
                and item["start_type"] == sequence_type[0]
                and item["end_type"] == sequence_type[1]
            ):
                indices.append(idx)
        return indices

    def plot_one_sequence(self, idx):
        """take idx of the data , plot the sequence"""
        path = self[idx]["path"]
        imgs = self[idx]["imgs"]
        bldg = self[idx]["bldg"]
        route = self[idx]["route"]
        sequence_type = f"{self[idx]['start_type']} to {self[idx]['end_type']}"
        for img in imgs:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f"{bldg}_{sequence_type}_route {route}_path {path}")
            plt.imshow(img.reshape(30, 60), cmap="gray", vmin=0, vmax=255)
            plt.show()
