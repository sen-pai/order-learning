from typing import Optional, Tuple, List, Dict
import random
import numpy as np
from PIL import Image, ImageDraw


import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from literals import *


def circle(
    output_path: str = "ignore",
    channels_first: bool = True,
    fill_color: Tuple = (255, 255, 255),
) -> Tuple[np.array]:
    """
    Create a circle in the center of the image, circle is filled with the rgb values provided.

    Args:
        output_path (str, optional): required if the image is to be saved. Defaults to "ignore".
        channels_first (bool, optional): pytorch needs. Defaults to True.
        fill_color (Tuple, optional): color. Defaults to (255, 255, 255).

    Returns:
        Tuple[np.array]: array of the image
    """

    image = Image.new("RGB", IMAGE_SIZE, BG_COLOR)
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, 50, 50), fill=fill_color)

    if output_path != "ignore":
        image.save(output_path)

    if channels_first:
        return np.moveaxis(np.array(image), -1, 0)

    return np.array(image)


def get_comp_tuple(
    type: str = "equal", index_i: int = -1, index_j: int = -1
) -> Tuple[np.array, np.array, torch.tensor]:
    """
    Args:
        type (str, ): three options for type = equal, greater, lesser. Defaults to "equal".
        index_i (int): Needed if type is greater or lesser, else randomly chosen
        index_j (int): same as index_i

    Returns:
        Tuple[np.array, np.array, torch.tensor]: x_i array, x_j array, one_hot
    """

    if type == "equal":
        one_hot = F.one_hot(torch.tensor(1), 3)

        assert index_i >= 0, "check index_i for type = equal"
        x_i = circle(fill_color=GREEN_COLORS[index_i][0])
        x_j = circle(fill_color=GREEN_COLORS[index_i][0])

        return x_i, x_j, one_hot

    elif type == "greater":
        one_hot = F.one_hot(torch.tensor(2), 3)

        assert index_i >= 0, "check index_i for type = greater"
        assert index_j >= 0, "check index_j for type = greater"

    elif type == "lesser":
        one_hot = F.one_hot(torch.tensor(0), 3)

        assert index_i >= 0, "check index_i for type = lesser"
        assert index_j >= 0, "check index_j for type = lesser"

    x_i = circle(fill_color=GREEN_COLORS[index_i][0])
    x_j = circle(fill_color=GREEN_COLORS[index_j][0])

    return x_i, x_j, one_hot


class ColorGradientMain(Dataset):
    def __init__(
        self, dataset_len: int = len(GREEN_COLORS), normalize: bool = True
    ) -> None:
        self.dataset_len = dataset_len
        self.normalize = normalize

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int, image_save_path: str = "ignore"):

        color, rank = GREEN_COLORS[index]
        green_circle = circle(image_save_path, fill_color=color)

        if self.normalize:
            return green_circle / 255.0, rank

        return green_circle, rank


class ColorGradientComp(Dataset):
    def __init__(
        self, dataset_len: int = 40, normalize: bool = True, scheme: Dict = EXTREMES
    ) -> None:
        self.dataset_len = dataset_len
        self.normalize = normalize
        self.types = ["equal", "greater", "lesser"]
        self.comp_items = []

        # create a fixed list of comp items
        for _ in range(self.dataset_len):
            t = random.sample(self.types, 1)[0]

            if t == "equal":
                index_i = random.sample(
                    scheme[random.sample(["upper", "lower"], 1)[0]], 1
                )[0]
                self.comp_items.append(get_comp_tuple(t, index_i))
            elif t == "lesser":
                index_i = random.sample(scheme["lower"], 1)[0]
                index_j = random.sample(scheme["upper"], 1)[0]
                self.comp_items.append(get_comp_tuple(t, index_i, index_j))
            elif t == "greater":
                index_j = random.sample(scheme["lower"], 1)[0]
                index_i = random.sample(scheme["upper"], 1)[0]
                self.comp_items.append(get_comp_tuple(t, index_i, index_j))

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, index: int):

        x_i, x_j, one_hot = self.comp_items[index]

        if self.normalize:
            return x_i / 255.0, x_j / 255.0, one_hot

        return x_i, x_j, one_hot


if __name__ == "__main__":

    import os

    # # sanity check, save all the colors
    assets_path = os.path.join(os.getcwd(), "assets")
    for c in GREEN_COLORS:
        circle(
            output_path=os.path.join(assets_path, str(c[1]) + ".png"), fill_color=c[0]
        )

    grad_data = ColorGradientMain()
    print(grad_data.__getitem__(10, os.path.join(assets_path, "check.png")))

    grad_comp = ColorGradientComp()
    print(grad_comp.__getitem__(10))
