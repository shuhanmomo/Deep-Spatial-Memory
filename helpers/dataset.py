import os
import numpy as np
import torch
from torch.utils.data import Dataset
import zipfile
import tempfile
import re
from tqdm import tqdm
from utils import read_image
import matplotlib.pyplot as plt


def format_name(zip_file_name):
    """
    reads zip file name and format it into lower case
    """
    formatted_name = zip_file_name.replace(".zip", "").replace("_", " ").lower()
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
        self.load_dataset()

    def load_dataset(self):
        # Extract the main zip file if needed
        with zipfile.ZipFile(self.data_path, "r") as main_zip:
            with tempfile.TemporaryDirectory() as temp_dir:
                main_zip.extractall(temp_dir)

                # Iterate over each building sequence label zip file inside the main directory
                for bldg_zip_name in main_zip.namelist():
                    bldg_route_label = format_name(bldg_zip_name)
                    route = bldg_route_label[
                        -1
                    ]  # The last character is the route label a,b,c,d
                    bldg = bldg_route_label[:-2]
                    if bldg not in self.building_names:
                        self.building_names.append(bldg)

                    # Extract the building sequence label zip file
                    bldg_zip_path = os.path.join(temp_dir, bldg_zip_name)
                    with zipfile.ZipFile(bldg_zip_path, "r") as bldg_zip:
                        bldg_temp_dir = os.path.join(temp_dir, bldg_route_label)
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
                                        img_array = read_image(img_path)
                                        path_images.append(img_array)

                                # Only consider complete sequences with 30 frames
                                if len(path_images) != self.num_frame:
                                    print(
                                        f"Error: {bldg}, Route {route}, Path {path_folder} does not have 30 images."
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
                                        }
                                    )

    def __getitem__(self, index):
        item = self.data[index]
        imgs = item["images"]

        # Apply transform if provided
        if self.transform:
            t_imgs = self.transform(imgs)

        # Convert images to PyTorch tensor
        else:
            t_imgs = torch.from_numpy(imgs).float()

        t_imgs = t_imgs.unsqueeze(1)  # num_frame, C,H,W
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
        }

    def __len__(self):
        return len(self.data)

    def find_indices(self, bldg_name, route, path_number):
        """
        Finds the indices of the data items that match the given building, route, and path number.

        :param bldg_name: The name of the building (formatted as 'caracalla baths', for example).
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
            ):
                indices.append(idx)
        return indices

    def plot_one_sequence(self, idx):
        """take idx of the data , plot the sequence"""
        path = self[idx]["path"]
        imgs = self[idx]["imgs"]
        bldg = self[idx]["bldg"]
        route = self[idx]["route"]
        for img in imgs:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title(f"{bldg} route {route} path {path}")
            plt.imshow(img.reshape(30, 60), cmap="gray", vmin=0, vmax=255)
            plt.show()
